"""Sparse draft KV: retain a recent window plus attention-ranked heavy hitters (SnapKV-style)."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

from mlsys_kv.cache.hf_kv_clone import (
    clone_past_key_values,
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
from mlsys_kv.cache.kv_cache_base import KVCacheBase


def gather_retained_kv_layers(
    full: Any,
    indices: list[int],
) -> tuple[str, list[tuple[torch.Tensor, torch.Tensor]] | None, list[Any] | None]:
    """Project KV onto **sorted retained** token indices (FP16 tensors).

    Shared by :class:`KVCacheSparse` and :class:`KVCacheSparseQuantized` so the
    **sparsify-first** step is identical before optional quantization.
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


class KVCacheSparse(KVCacheBase):
    """Draft-only cache storing **gathered** K/V for a retained token subset.

    The **verifier** remains full FP16; approximation lives only on the draft path.
    :meth:`get_attention_kv` returns a dense HF-style ``past_key_values`` object
    whose sequence length equals ``len(retained_indices)`` (positions appear in
    **sorted global token order**, so causality among kept tokens is preserved).

    **Memory accounting**

    * **payload**: bytes of stored key/value tensors after ``index_select``.
    * **metadata**: global retained indices (``int64``), optional score buffer
      (``float32`` per position), and fixed config-sized bookkeeping omitted
      (refresh counters live in :meth:`stats`, not duplicated in bytes).

    **Metrics**

    Refresh CPU/GPU time is accumulated in ``cumulative_refresh_time_s``;
    average sparsity (``1 - retained/full``) over appends is reported in :meth:`stats`.
    """

    __slots__ = (
        "_config",
        "_model",
        "_fmt",
        "_tuple_kv",
        "_dynamic_layers",
        "_full_seq_len",
        "_retained_indices",
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
        self._fmt: str | None = None
        self._tuple_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._dynamic_layers: list[Any] | None = None

        self._full_seq_len: int = 0
        self._retained_indices: list[int] = []
        self._token_scores_cpu: torch.Tensor | None = None

        self._append_calls: int = 0
        self._last_token: torch.Tensor | None = None

        self._cumulative_refresh_s: float = 0.0
        self._refresh_events: int = 0

        self._sum_sparsity: float = 0.0
        self._sparsity_samples: int = 0

    def note_forward_token(self, token_ids: torch.Tensor) -> None:
        """Record the token id tensor ``[B,1]`` that will be fed on the next draft forward.

        Used only for attention scoring on refresh; safe to omit for snapshot-only appends.
        """

        self._last_token = token_ids.detach()

    def append_from_forward_output(self, past_key_values: Any) -> None:
        full = clone_past_key_values(past_key_values)
        if full is None:
            self._fmt = None
            self._tuple_kv = None
            self._dynamic_layers = None
            self._full_seq_len = 0
            self._retained_indices = []
            return

        L = past_sequence_length(full)
        self._full_seq_len = L

        need_refresh = self._token_scores_cpu is None or (
            self._append_calls % self._config.refresh_interval == 0
        )
        if need_refresh:
            t0 = time.perf_counter()
            self._token_scores_cpu = self._compute_scores_cpu(full, L).cpu()
            self._cumulative_refresh_s += time.perf_counter() - t0
            self._refresh_events += 1
        elif self._token_scores_cpu is not None:
            # Periodic schedule: grow or trim the cached score vector to length L without
            # rerunning attention (newest index is assumed important).
            old = self._token_scores_cpu
            if old.shape[0] < L:
                padded = torch.zeros(L, dtype=old.dtype, device=old.device)
                c = int(old.shape[0])
                padded[:c] = old
                padded[L - 1] = torch.tensor(float("inf"), dtype=old.dtype, device=old.device)
                self._token_scores_cpu = padded
            elif old.shape[0] > L:
                self._token_scores_cpu = old[:L].clone()

        assert self._token_scores_cpu is not None
        idx = select_retained_token_indices(
            L,
            self._token_scores_cpu,
            recent_window=self._config.recent_window,
            heavy_hitter_budget=self._config.heavy_hitter_budget,
        )
        self._retained_indices = idx
        if L > 0:
            self._sum_sparsity += 1.0 - (len(idx) / float(L))
            self._sparsity_samples += 1

        fmt, tup, dyn = gather_retained_kv_layers(full, idx)
        self._fmt = fmt
        self._tuple_kv = tup
        self._dynamic_layers = dyn
        self._append_calls += 1

    def _compute_scores_cpu(self, full_past: Any, L: int) -> torch.Tensor:
        """Return length-``L`` float scores on the current KV device (caller moves to CPU if desired)."""
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

    def get_attention_kv(self) -> Any | None:
        if self._fmt is None:
            return None
        if self._fmt == "tuple":
            assert self._tuple_kv is not None
            return tuple(self._tuple_kv)
        assert self._dynamic_layers is not None
        out = DynamicCache()
        out.layers = list(self._dynamic_layers)
        return out

    def _payload_and_meta_bytes(self) -> tuple[int, int]:
        payload = 0
        if self._tuple_kv:
            for k, v in self._tuple_kv:
                payload += int(k.numel()) * k.element_size()
                payload += int(v.numel()) * v.element_size()
        if self._dynamic_layers:
            for layer in self._dynamic_layers:
                if isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer)):
                    if layer.is_initialized and layer.keys is not None:
                        k, v = layer.keys, layer.values
                        payload += int(k.numel()) * k.element_size()
                        payload += int(v.numel()) * v.element_size()

        metadata = 0
        if self._retained_indices:
            metadata += len(self._retained_indices) * 8  # int64 index list (logical)
        if self._token_scores_cpu is not None:
            metadata += int(self._token_scores_cpu.numel()) * int(
                self._token_scores_cpu.element_size()
            )

        # Fixed refresh metadata (counts are in stats; reserve 3 int64)
        metadata += 3 * 8
        return int(payload), int(metadata)

    def memory_bytes(self) -> int:
        p, m = self._payload_and_meta_bytes()
        return p + m

    def stats(self) -> dict[str, Any]:
        retained = len(self._retained_indices)
        mean_sparsity = (
            (self._sum_sparsity / self._sparsity_samples) if self._sparsity_samples > 0 else 0.0
        )
        payload, metadata = self._payload_and_meta_bytes()
        dense_bytes_est = 0
        if self._full_seq_len > 0 and retained > 0:
            scale = self._full_seq_len / float(retained)
            dense_bytes_est = int(payload * scale) if retained else payload
        return {
            "type": "KVCacheSparse",
            "selection_scope": "token_level_shared_across_heads_per_layer",
            "recent_window": int(self._config.recent_window),
            "heavy_hitter_budget": int(self._config.heavy_hitter_budget),
            "refresh_interval_decode_steps": int(self._config.refresh_interval),
            "scoring_mode": self._config.scoring,
            "full_sequence_length": int(self._full_seq_len),
            "retained_sequence_length": int(retained),
            "sparsity_ratio_point_estimate": (
                (1.0 - retained / float(self._full_seq_len)) if self._full_seq_len > 0 else 0.0
            ),
            "mean_sparsity_ratio_over_appends": float(mean_sparsity),
            "payload_bytes": int(payload),
            "metadata_bytes": int(metadata),
            "memory_bytes_logical": int(payload + metadata),
            "dense_fp16_kv_bytes_est": int(dense_bytes_est),
            "refresh_events": int(self._refresh_events),
            "append_calls": int(self._append_calls),
            "cumulative_refresh_time_s": float(self._cumulative_refresh_s),
            "num_layers": (
                len(self._tuple_kv)
                if self._tuple_kv
                else (len(self._dynamic_layers) if self._dynamic_layers else 0)
            ),
        }
