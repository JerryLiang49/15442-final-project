"""Sparse draft KV: retain a recent window plus attention-ranked heavy hitters (SnapKV-style).

**Phase 11 — positional correctness**

After eviction, physical KV length ``R`` is shorter than logical length ``L``. Explicit
``position_ids`` for the next query must start at ``L`` (see
:meth:`~mlsys_kv.cache.kv_cache_base.KVCacheBase.position_ids_for_next_queries`), wired in
:func:`~mlsys_kv.decoding.speculative.propose_draft_tokens`.

**Phase 12 — HF integration boundary**

Selection policy knobs are unchanged (:class:`~mlsys_kv.cache.heavy_hitter_selector.SparseRetentionConfig`).
All logic that reconciles HuggingFace ``past_key_values`` **physical** length with **logical**
sequence length, builds score vectors, selects globals, and gathers rows lives in
:class:`~mlsys_kv.cache.sparse_hf_integration.SparseHFCacheIntegrator`. This class is a thin
:class:`~mlsys_kv.cache.kv_cache_base.KVCacheBase` adapter that stores the last materialized
FP16 KV and forwards lifecycle hooks (e.g. :meth:`reset`).
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

from mlsys_kv.cache.hf_kv_clone import clone_past_key_values
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.sparse_hf_integration import (
    SparseHFCacheIntegrator,
    gather_retained_kv_layers,
)

# Re-export for callers that imported gather from this module (benchmarks, tests).
__all__ = ("KVCacheSparse", "gather_retained_kv_layers")


class KVCacheSparse(KVCacheBase):
    """Draft-only cache: integrator shortens HF KV; this class holds FP16 tensors + metrics."""

    __slots__ = ("_config", "_model", "_integrator", "_fmt", "_tuple_kv", "_dynamic_layers")

    def __init__(self, config: SparseRetentionConfig, model: PreTrainedModel | None = None) -> None:
        self._config = config
        self._model = model
        self._integrator = SparseHFCacheIntegrator(config, model=model)
        self._fmt: str | None = None
        self._tuple_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._dynamic_layers: list[Any] | None = None

    def reset(self) -> None:
        """Clear sparse integrator state and materialized KV (e.g. new prompt, reused instance)."""
        self._integrator.reset()
        self._fmt = None
        self._tuple_kv = None
        self._dynamic_layers = None

    @property
    def hf_integrator(self) -> SparseHFCacheIntegrator:
        """Phase 12 integration object (logical length, scores, retained globals)."""
        return self._integrator

    @property
    def logical_seq_len(self) -> int:
        return self._integrator.logical_seq_len

    @property
    def retained_global_indices(self) -> list[int]:
        return self._integrator.retained_global_indices

    def position_ids_for_next_queries(
        self,
        query_length: int,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self._fmt is None:
            return None
        start = self._integrator.logical_seq_len
        row = torch.arange(query_length, device=device, dtype=torch.long) + int(start)
        return row.unsqueeze(0).expand(batch_size, -1)

    def note_forward_token(self, token_ids: torch.Tensor) -> None:
        self._integrator.set_forward_note_token(token_ids)

    def append_from_forward_output(self, past_key_values: Any) -> None:
        full = clone_past_key_values(past_key_values)
        if full is None:
            self.reset()
            return

        result = self._integrator.integrate_cloned_hf_past(full)
        self._fmt = result.fmt
        self._tuple_kv = result.tuple_kv
        self._dynamic_layers = result.dynamic_layers

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
        idx = self._integrator.retained_global_indices
        if idx:
            metadata += len(idx) * 8
        sc = self._integrator.token_scores_cpu
        if sc is not None:
            metadata += int(sc.numel()) * int(sc.element_size())
        metadata += 3 * 8
        return int(payload), int(metadata)

    def memory_bytes(self) -> int:
        p, m = self._payload_and_meta_bytes()
        return p + m

    def stats(self) -> dict[str, Any]:
        retained = len(self._integrator.retained_global_indices)
        rs = self._integrator.refresh_stats()
        mean_sparsity = (
            (rs["sum_sparsity"] / rs["sparsity_samples"]) if rs["sparsity_samples"] > 0 else 0.0
        )
        logical = self._integrator.logical_seq_len
        payload, metadata = self._payload_and_meta_bytes()
        dense_bytes_est = 0
        if logical > 0 and retained > 0:
            scale = logical / float(retained)
            dense_bytes_est = int(payload * scale) if retained else payload
        return {
            "type": "KVCacheSparse",
            "selection_scope": "token_level_shared_across_heads_per_layer",
            "recent_window": int(self._config.recent_window),
            "heavy_hitter_budget": int(self._config.heavy_hitter_budget),
            "refresh_interval_decode_steps": int(self._config.refresh_interval),
            "scoring_mode": self._config.scoring,
            "full_sequence_length": int(logical),
            "retained_sequence_length": int(retained),
            "sparsity_ratio_point_estimate": (
                (1.0 - retained / float(logical)) if logical > 0 else 0.0
            ),
            "mean_sparsity_ratio_over_appends": float(mean_sparsity),
            "payload_bytes": int(payload),
            "metadata_bytes": int(metadata),
            "memory_bytes_logical": int(payload + metadata),
            "dense_fp16_kv_bytes_est": int(dense_bytes_est),
            "refresh_events": int(rs["refresh_events"]),
            "append_calls": int(rs["append_calls"]),
            "cumulative_refresh_time_s": float(rs["cumulative_refresh_time_s"]),
            "num_layers": (
                len(self._tuple_kv)
                if self._tuple_kv
                else (len(self._dynamic_layers) if self._dynamic_layers else 0)
            ),
            "logical_seq_len_full": int(logical),
            "next_query_position_start": int(logical),
            "physical_retained_kv_len": int(retained),
            "sparse_hf_integration": "SparseHFCacheIntegrator",
        }
