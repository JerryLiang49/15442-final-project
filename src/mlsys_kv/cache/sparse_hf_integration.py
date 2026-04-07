"""Phase 12 — Hugging Face sparse **draft** integration boundary.

Why this exists
---------------
Sparse draft shortens ``past_key_values`` on the sequence axis. HuggingFace models assume
physical cache length tracks how they build default ``position_ids`` (via
``past_key_values.get_seq_length()``). Logical timeline length, per-row **global** token
indices, and **physical** row indices into the HF tensor must stay consistent or decode
silently drifts (Phase 11).

Previously those rules lived inside :class:`~mlsys_kv.cache.kv_cache_sparse.KVCacheSparse`
alongside payload accounting. **Phase 12** pulls *all* “model output → shortened KV +
bookkeeping” logic into this module so:

* The integration contract with HF is explicit and localized.
* The retention **policy** (recent / heavy hitters / refresh / scoring knobs from
  :class:`~mlsys_kv.cache.heavy_hitter_selector.SparseRetentionConfig`) stays unchanged but
  is applied only *after* we reconcile logical vs physical lengths.
* Quantized sparse draft (:class:`~mlsys_kv.cache.kv_cache_sparse_quantized.KVCacheSparseQuantized`)
  reuses the same integrator; it only layers quantization **on top of** the gathered FP16
  tensors returned here.

Execution path
--------------
1. Draft ``model.forward`` returns full ``past_key_values`` (possibly **physically short**
   after a prior sparse step).
2. :meth:`SparseHFCacheIntegrator.integrate_cloned_hf_past` consumes a **clone** of that
   object, updates logical length and scores, runs :func:`~mlsys_kv.cache.heavy_hitter_selector.select_retained_token_indices`,
   gathers K/V with :func:`gather_retained_kv_layers`.
3. :class:`~mlsys_kv.cache.kv_cache_sparse.KVCacheSparse` stores the result and exposes it via
   :meth:`~mlsys_kv.cache.kv_cache_base.KVCacheBase.get_attention_kv`.
4. The speculative loop passes explicit ``position_ids`` when the cache reports them
  (:meth:`~mlsys_kv.cache.kv_cache_base.KVCacheBase.position_ids_for_next_queries`).

Stateful components must call :meth:`SparseHFCacheIntegrator.reset` (or
:meth:`~mlsys_kv.cache.kv_cache_sparse.KVCacheSparse.reset`) when starting a **new prompt**
if a cache instance is reused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

from mlsys_kv.cache.hf_kv_clone import (
    past_sequence_length,
    strip_last_position_from_past,
)
from mlsys_kv.cache.heavy_hitter_selector import (
    SparseRetentionConfig,
    attention_mass_from_last_token,
    build_full_length_scores_from_attention_prefix,
    key_norm_token_scores,
    select_retained_token_indices,
)


def scores_tensor_for_sparse_physical_past(
    full_past: Any,
    *,
    logical_len: int,
    prev_retained: list[int],
    prev_logical_len: int,
    prior_scores_cpu: torch.Tensor | None,
) -> torch.Tensor:
    """Build a length-``logical_len`` score vector when ``full_past`` seq dim is **physical** only.

    Rows of ``full_past`` (length ``P``) map to globals ``prev_retained`` (length ``P-1``) plus
    the new token at global ``prev_logical_len``. Unfilled entries stay very negative so the
    selector (with ``eligible_positions``) never requests missing KV rows.
    """
    kn = key_norm_token_scores(full_past)
    p = int(kn.shape[0])
    pr = len(prev_retained)
    out = torch.full((logical_len,), -1e4, dtype=torch.float32)
    if prior_scores_cpu is not None and prior_scores_cpu.shape[0] > 0:
        c = min(int(prior_scores_cpu.shape[0]), max(0, logical_len - 1))
        if c > 0:
            out[:c] = prior_scores_cpu[:c].float()
    if p >= 1 and pr + 1 == p:
        for i, g in enumerate(prev_retained):
            out[g] = kn[i]
        out[prev_logical_len] = kn[p - 1]
    if logical_len >= 1:
        out[logical_len - 1] = torch.tensor(float("inf"), dtype=torch.float32)
    return out


def gather_retained_kv_layers(
    full: Any,
    indices: list[int],
) -> tuple[str, list[tuple[torch.Tensor, torch.Tensor]] | None, list[Any] | None]:
    """Project KV along dim ``-2`` using integer **row** indices into ``full``.

    For a **dense** prefix, row index equals the global token index. After sparse steps,
    ``indices`` are physical row positions (see :meth:`SparseHFCacheIntegrator.integrate_cloned_hf_past`).
    """

    idx_t = torch.tensor(indices, dtype=torch.long)
    if isinstance(full, tuple):
        tuple_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for item in full:
            if item is None:
                continue
            k, v = item[0], item[1]
            d = idx_t.to(device=k.device)
            tuple_kv.append((k.index_select(-2, d), v.index_select(-2, d)))
        return "tuple", tuple_kv, None

    if isinstance(full, DynamicCache):
        dynamic_layers: list[Any] = []
        for layer in full.layers:
            if isinstance(layer, DynamicSlidingWindowLayer):
                if layer.is_initialized and layer.keys is not None and layer.keys.shape[-2] > 0:
                    k, v = layer.keys, layer.values
                    d = idx_t.to(device=k.device)
                    nl = DynamicSlidingWindowLayer(sliding_window=layer.sliding_window)
                    nl.keys = k.index_select(-2, d)
                    nl.values = v.index_select(-2, d)
                    nl.is_initialized = True
                    nl.dtype = nl.keys.dtype
                    nl.device = nl.keys.device
                    nl.cumulative_length = int(nl.keys.shape[-2])
                    nl._sliding_window_tensor = layer._sliding_window_tensor.to(device=nl.device)
                    dynamic_layers.append(nl)
                else:
                    dynamic_layers.append(layer)
            elif isinstance(layer, DynamicLayer):
                if layer.is_initialized and layer.keys is not None and layer.keys.shape[-2] > 0:
                    k, v = layer.keys, layer.values
                    d = idx_t.to(device=k.device)
                    nl = DynamicLayer()
                    nl.keys = k.index_select(-2, d)
                    nl.values = v.index_select(-2, d)
                    nl.is_initialized = True
                    nl.dtype = nl.keys.dtype
                    nl.device = nl.keys.device
                    dynamic_layers.append(nl)
                else:
                    dynamic_layers.append(layer)
            else:
                dynamic_layers.append(layer)
        return "dynamic", None, dynamic_layers

    raise TypeError(f"gather_retained_kv_layers: unsupported past type {type(full)}")


@dataclass(frozen=True)
class SparseIntegrationStepResult:
    """Materialized sparse draft KV for one absorbed HF forward (FP16 tensors)."""

    fmt: str
    tuple_kv: list[tuple[torch.Tensor, torch.Tensor]] | None
    dynamic_layers: list[Any] | None
    retained_global_indices: list[int]
    logical_seq_len: int


class SparseHFCacheIntegrator:
    """Single place that turns an HF ``past_key_values`` clone into shortened draft KV + bookkeeping.

    **Policy** is unchanged: same :class:`~mlsys_kv.cache.heavy_hitter_selector.SparseRetentionConfig`
    (recent window, heavy-hitter budget, refresh interval, scoring mode). This class only
    sequences scoring, eligibility masking, and gather so HF physical length never poisons
    logical length.
    """

    __slots__ = (
        "_config",
        "_model",
        "_logical_seq_len",
        "_retained_global",
        "_token_scores_cpu",
        "_append_calls",
        "_last_token",
        "_cumulative_refresh_s",
        "_refresh_events",
        "_sum_sparsity",
        "_sparsity_samples",
    )

    def __init__(self, config: SparseRetentionConfig, model: PreTrainedModel | None = None) -> None:
        self._config = config
        self._model = model
        self.reset()

    def reset(self) -> None:
        """Drop all draft-step state (new prompt or reused cache instance)."""
        self._logical_seq_len = 0
        self._retained_global: list[int] = []
        self._token_scores_cpu: torch.Tensor | None = None
        self._append_calls = 0
        self._last_token = None
        self._cumulative_refresh_s = 0.0
        self._refresh_events = 0
        self._sum_sparsity = 0.0
        self._sparsity_samples = 0

    def set_forward_note_token(self, token_ids: torch.Tensor | None) -> None:
        """Token(s) about to be decoded; used for optional attention-based scoring."""
        self._last_token = token_ids.detach() if token_ids is not None else None

    @property
    def logical_seq_len(self) -> int:
        return int(self._logical_seq_len)

    @property
    def retained_global_indices(self) -> list[int]:
        return list(self._retained_global)

    @property
    def append_calls(self) -> int:
        return int(self._append_calls)

    @property
    def token_scores_cpu(self) -> torch.Tensor | None:
        return self._token_scores_cpu

    def refresh_stats(self) -> dict[str, float | int]:
        return {
            "cumulative_refresh_time_s": float(self._cumulative_refresh_s),
            "refresh_events": int(self._refresh_events),
            "append_calls": int(self._append_calls),
            "sum_sparsity": float(self._sum_sparsity),
            "sparsity_samples": int(self._sparsity_samples),
        }

    def integrate_cloned_hf_past(self, full: Any) -> SparseIntegrationStepResult:
        """Absorb one decoded step from an HF ``past_key_values`` object (caller must pass a clone).

        Raises:
            RuntimeError: If physical row count does not match ``prev_retained + new`` globals.
        """
        prev_r = len(self._retained_global)
        prev_L = int(self._logical_seq_len)
        prev_retained = list(self._retained_global)
        phys = past_sequence_length(full)

        if prev_L == 0 and prev_r == 0:
            logical_L = phys
            global_per_phys = tuple(range(phys))
            eligible: frozenset[int] | None = None
        else:
            logical_L = prev_L + (phys - prev_r)
            global_per_phys = tuple(prev_retained + [prev_L])
            if len(global_per_phys) != phys:
                raise RuntimeError(
                    f"sparse KV row/global mismatch: P={phys} rows vs {len(global_per_phys)} globals"
                )
            eligible = frozenset(global_per_phys)

        row_for_global = {g: i for i, g in enumerate(global_per_phys)}
        self._logical_seq_len = logical_L

        need_refresh = self._token_scores_cpu is None or (
            self._append_calls % self._config.refresh_interval == 0
        )
        if need_refresh:
            t0 = time.perf_counter()
            if phys >= logical_L:
                self._token_scores_cpu = self._compute_scores_cpu(full, logical_L).cpu()
            else:
                self._token_scores_cpu = scores_tensor_for_sparse_physical_past(
                    full,
                    logical_len=logical_L,
                    prev_retained=prev_retained,
                    prev_logical_len=prev_L,
                    prior_scores_cpu=self._token_scores_cpu,
                ).cpu()
            self._cumulative_refresh_s += time.perf_counter() - t0
            self._refresh_events += 1
        elif self._token_scores_cpu is not None:
            old = self._token_scores_cpu
            if old.shape[0] < logical_L:
                padded = torch.zeros(logical_L, dtype=old.dtype, device=old.device)
                c = int(old.shape[0])
                padded[:c] = old
                padded[logical_L - 1] = torch.tensor(float("inf"), dtype=old.dtype, device=old.device)
                self._token_scores_cpu = padded
            elif old.shape[0] > logical_L:
                self._token_scores_cpu = old[:logical_L].clone()

        assert self._token_scores_cpu is not None
        idx = select_retained_token_indices(
            logical_L,
            self._token_scores_cpu,
            recent_window=self._config.recent_window,
            heavy_hitter_budget=self._config.heavy_hitter_budget,
            eligible_positions=eligible,
        )
        self._retained_global = idx

        assert len(idx) <= logical_L
        if logical_L > 0 and idx:
            assert max(idx) < logical_L
        if logical_L > 0:
            self._sum_sparsity += 1.0 - (len(idx) / float(logical_L))
            self._sparsity_samples += 1

        phys_rows = [row_for_global[g] for g in idx]
        fmt, tup, dyn = gather_retained_kv_layers(full, phys_rows)
        self._append_calls += 1

        return SparseIntegrationStepResult(
            fmt=fmt,
            tuple_kv=tup,
            dynamic_layers=dyn,
            retained_global_indices=list(idx),
            logical_seq_len=logical_L,
        )

    def _compute_scores_cpu(self, full_past: Any, L: int) -> torch.Tensor:
        if L == 0:
            return torch.zeros(0)

        use_attention = (
            self._config.scoring == "attention"
            and self._model is not None
            and self._last_token is not None
            and L > 1
        )
        if use_attention:
            try:
                prefix = strip_last_position_from_past(full_past)
                if past_sequence_length(prefix) == 0:
                    return key_norm_token_scores(full_past)
                s_pre = attention_mass_from_last_token(self._model, prefix, self._last_token)
                return build_full_length_scores_from_attention_prefix(s_pre, total_len=L)
            except Exception:
                return key_norm_token_scores(full_past)

        return key_norm_token_scores(full_past)
