"""Joint draft KV: **sparsify first** (recent + heavy hitters), then **INT8-quantize** retained K/V only.

**Composition order** (required for report / benchmarks)

1. Clone full HF ``past_key_values`` from one decode step.
2. Run the same token-level retention policy as :class:`~mlsys_kv.cache.kv_cache_sparse.KVCacheSparse`
   (scores, periodic refresh, ``select_retained_token_indices``).
3. :func:`~mlsys_kv.cache.sparse_hf_integration.gather_retained_kv_layers` — gather K/V to retained length ``R``.
4. Apply :func:`~mlsys_kv.cache.quantization.symmetric_quantize_int8` **per tensor** (each gathered
   ``keys`` / ``values``), identical to :class:`~mlsys_kv.cache.kv_cache_quantized.KVCacheQuantized`,
   storing **only** ``int8`` codes + **FP32 scale** (×2 per layer for K and V).

**Memory accounting** (:meth:`memory_bytes`, :meth:`stats`)

* **payload_bytes_int8**: retained key/value quantization codes (one byte per stored element).
* **metadata_bytes_sparse**: global ``int64`` retained index list, optional FP32 score buffer, fixed
  refresh bookkeeping (see sparse cache).
* **metadata_bytes_quant**: two FP32 scales per quantized layer, plus sliding-window auxiliary tensor
  bytes when present (matches quant-only cache).

The **verifier** stays full FP16; lossy draft does not change committed tokens under greedy verification.

**Phase 11:** Same explicit ``position_ids`` contract as :class:`~mlsys_kv.cache.kv_cache_sparse.KVCacheSparse`
(see that module docstring).

**Phase 12:** Sparsification uses :class:`~mlsys_kv.cache.sparse_hf_integration.SparseHFCacheIntegrator`
(same policy knobs); this class only **quantizes** the FP16 tensors produced by that integrator.

**Phase 13:** The quantized payload is **memory-only** — :meth:`get_attention_kv` dequantizes before HF
attention (see :mod:`mlsys_kv.cache.kv_quant_semantics`).

Combined **runtime** overhead in experiments: ``cumulative_refresh_time_s`` (selector) +
``cumulative_dequant_time_s`` (rebuild ``past_key_values`` for the next forward), surfaced via
:class:`~mlsys_kv.decoding.speculative.SpeculativeMetrics` and :meth:`stats`.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

from mlsys_kv.cache.hf_kv_clone import clone_past_key_values, _clone_dynamic_layer
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.kv_cache_int4 import _Int4KVPair, _Int4SlidePair
from mlsys_kv.cache.kv_quant_semantics import memory_only_quant_stats_fragment
from mlsys_kv.cache.kv_cache_quantized import _QuantKVPair, _QuantSlidePair
from mlsys_kv.cache.sparse_hf_integration import SparseHFCacheIntegrator
from mlsys_kv.cache.quantization import symmetric_dequantize_int8, symmetric_quantize_int8
from mlsys_kv.cache.quantization_int4 import (
    _DEFAULT_GROUP,
    symmetric_dequantize_int4_grouped_packed,
    symmetric_quantize_int4_grouped_packed,
)


class KVCacheSparseQuantized(KVCacheBase):
    """Draft-only: :class:`SparseHFCacheIntegrator` → symmetric INT8/INT4 on retained tensors only."""

    __slots__ = (
        "_config",
        "_model",
        "_integrator",
        "_fmt",
        "_tuple_entries",
        "_dynamic_entries",
        "_cumulative_dequant_s",
        "_use_int4",
    )

    def __init__(
        self,
        config: SparseRetentionConfig,
        model: PreTrainedModel | None = None,
        *,
        use_int4: bool = False,
    ) -> None:
        self._config = config
        self._model = model
        self._use_int4 = bool(use_int4)
        self._integrator = SparseHFCacheIntegrator(config, model=model)
        self._fmt: str | None = None
        self._tuple_entries: list[_QuantKVPair | _Int4KVPair] | None = None
        self._dynamic_entries: list[
            _QuantKVPair | _QuantSlidePair | _Int4KVPair | _Int4SlidePair | Any
        ] | None = None
        self._cumulative_dequant_s: float = 0.0

    def reset(self) -> None:
        self._integrator.reset()
        self._fmt = None
        self._tuple_entries = None
        self._dynamic_entries = None
        self._cumulative_dequant_s = 0.0

    @property
    def hf_integrator(self) -> SparseHFCacheIntegrator:
        return self._integrator

    @property
    def logical_seq_len(self) -> int:
        return self._integrator.logical_seq_len

    @property
    def retained_global_indices(self) -> list[int]:
        return self._integrator.retained_global_indices

    def note_forward_token(self, token_ids: torch.Tensor) -> None:
        self._integrator.set_forward_note_token(token_ids)

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

    def append_from_forward_output(self, past_key_values: Any) -> None:
        full = clone_past_key_values(past_key_values)
        if full is None:
            self.reset()
            return

        res = self._integrator.integrate_cloned_hf_past(full)
        fmt = res.fmt
        tup_fp16 = res.tuple_kv
        dyn_fp16 = res.dynamic_layers
        self._fmt = fmt
        if fmt == "tuple":
            self._dynamic_entries = None
            assert tup_fp16 is not None
            self._tuple_entries = []
            for k, v in tup_fp16:
                if self._use_int4:
                    pk, sk, pad_k, ok = symmetric_quantize_int4_grouped_packed(k, _DEFAULT_GROUP)
                    pv, sv, pad_v, ov = symmetric_quantize_int4_grouped_packed(v, _DEFAULT_GROUP)
                    self._tuple_entries.append(
                        _Int4KVPair(
                            pk=pk,
                            pv=pv,
                            scale_k=sk,
                            scale_v=sv,
                            pad_k=pad_k,
                            pad_v=pad_v,
                            orig_shape_k=ok,
                            orig_shape_v=ov,
                            group_size=int(_DEFAULT_GROUP),
                            kv_dtype=k.dtype,
                            kv_device=k.device,
                        )
                    )
                else:
                    qk, sk = symmetric_quantize_int8(k)
                    qv, sv = symmetric_quantize_int8(v)
                    self._tuple_entries.append(
                        _QuantKVPair(
                            qk=qk,
                            qv=qv,
                            scale_k=sk,
                            scale_v=sv,
                            kv_dtype=k.dtype,
                            kv_device=k.device,
                        )
                    )
        else:
            assert fmt == "dynamic"
            self._tuple_entries = None
            assert dyn_fp16 is not None
            self._dynamic_entries = []
            for layer in dyn_fp16:
                if isinstance(layer, DynamicSlidingWindowLayer):
                    if layer.is_initialized and layer.keys is not None:
                        k, v = layer.keys, layer.values
                        if self._use_int4:
                            pk, sk, pad_k, ok = symmetric_quantize_int4_grouped_packed(k, _DEFAULT_GROUP)
                            pv, sv, pad_v, ov = symmetric_quantize_int4_grouped_packed(v, _DEFAULT_GROUP)
                            self._dynamic_entries.append(
                                _Int4SlidePair(
                                    pk=pk,
                                    pv=pv,
                                    scale_k=sk,
                                    scale_v=sv,
                                    pad_k=pad_k,
                                    pad_v=pad_v,
                                    orig_shape_k=ok,
                                    orig_shape_v=ov,
                                    group_size=int(_DEFAULT_GROUP),
                                    kv_dtype=k.dtype,
                                    kv_device=k.device,
                                    cumulative_length=int(layer.cumulative_length),
                                    sliding_window=int(layer.sliding_window),
                                    sliding_window_tensor_cpu=layer._sliding_window_tensor.detach().cpu(),
                                )
                            )
                        else:
                            qk, sk = symmetric_quantize_int8(k)
                            qv, sv = symmetric_quantize_int8(v)
                            self._dynamic_entries.append(
                                _QuantSlidePair(
                                    qk=qk,
                                    qv=qv,
                                    scale_k=sk,
                                    scale_v=sv,
                                    kv_dtype=k.dtype,
                                    kv_device=k.device,
                                    cumulative_length=int(layer.cumulative_length),
                                    sliding_window=int(layer.sliding_window),
                                    sliding_window_tensor_cpu=layer._sliding_window_tensor.detach().cpu(),
                                )
                            )
                    else:
                        self._dynamic_entries.append(_clone_dynamic_layer(layer))
                elif isinstance(layer, DynamicLayer):
                    if layer.is_initialized and layer.keys is not None:
                        k, v = layer.keys, layer.values
                        if self._use_int4:
                            pk, sk, pad_k, ok = symmetric_quantize_int4_grouped_packed(k, _DEFAULT_GROUP)
                            pv, sv, pad_v, ov = symmetric_quantize_int4_grouped_packed(v, _DEFAULT_GROUP)
                            self._dynamic_entries.append(
                                _Int4KVPair(
                                    pk=pk,
                                    pv=pv,
                                    scale_k=sk,
                                    scale_v=sv,
                                    pad_k=pad_k,
                                    pad_v=pad_v,
                                    orig_shape_k=ok,
                                    orig_shape_v=ov,
                                    group_size=int(_DEFAULT_GROUP),
                                    kv_dtype=k.dtype,
                                    kv_device=k.device,
                                )
                            )
                        else:
                            qk, sk = symmetric_quantize_int8(k)
                            qv, sv = symmetric_quantize_int8(v)
                            self._dynamic_entries.append(
                                _QuantKVPair(
                                    qk=qk,
                                    qv=qv,
                                    scale_k=sk,
                                    scale_v=sv,
                                    kv_dtype=k.dtype,
                                    kv_device=k.device,
                                )
                            )
                    else:
                        self._dynamic_entries.append(_clone_dynamic_layer(layer))
                else:
                    self._dynamic_entries.append(_clone_dynamic_layer(layer))

    def _ephemeral_attention_kv_rebuild_bytes_est(self) -> int:
        """High-precision K/V bytes built during :meth:`get_attention_kv` (not stored)."""
        from math import prod

        total_elems = 0
        for ent in self._tuple_entries or []:
            if isinstance(ent, _Int4KVPair):
                total_elems += prod(ent.orig_shape_k) + prod(ent.orig_shape_v)
            else:
                total_elems += ent.qk.numel() + ent.qv.numel()
        for ent in self._dynamic_entries or []:
            if isinstance(ent, (_Int4KVPair, _Int4SlidePair)):
                total_elems += prod(ent.orig_shape_k) + prod(ent.orig_shape_v)
            elif isinstance(ent, (_QuantKVPair, _QuantSlidePair)):
                total_elems += ent.qk.numel() + ent.qv.numel()
        return int(total_elems * 2)

    def get_attention_kv(self) -> Any | None:
        # Phase 13 (memory-only quant): full dequant rebuild for standard HF attention.
        if self._fmt is None:
            return None
        t0 = time.perf_counter()
        try:
            if self._fmt == "tuple":
                assert self._tuple_entries is not None
                layers: list[tuple[torch.Tensor, torch.Tensor]] = []
                for ent in self._tuple_entries:
                    if isinstance(ent, _Int4KVPair):
                        k = symmetric_dequantize_int4_grouped_packed(
                            ent.pk,
                            ent.scale_k,
                            original_shape=ent.orig_shape_k,
                            pad_amt=ent.pad_k,
                            group_size=ent.group_size,
                            out_dtype=ent.kv_dtype,
                            out_device=ent.kv_device,
                        )
                        v = symmetric_dequantize_int4_grouped_packed(
                            ent.pv,
                            ent.scale_v,
                            original_shape=ent.orig_shape_v,
                            pad_amt=ent.pad_v,
                            group_size=ent.group_size,
                            out_dtype=ent.kv_dtype,
                            out_device=ent.kv_device,
                        )
                    else:
                        k = symmetric_dequantize_int8(
                            ent.qk, ent.scale_k, out_dtype=ent.kv_dtype, out_device=ent.kv_device
                        )
                        v = symmetric_dequantize_int8(
                            ent.qv, ent.scale_v, out_dtype=ent.kv_dtype, out_device=ent.kv_device
                        )
                    layers.append((k, v))
                return tuple(layers)

            assert self._dynamic_entries is not None
            new_cache = DynamicCache()
            new_layers: list[Any] = []
            for entry in self._dynamic_entries:
                if isinstance(entry, _Int4SlidePair):
                    k = symmetric_dequantize_int4_grouped_packed(
                        entry.pk,
                        entry.scale_k,
                        original_shape=entry.orig_shape_k,
                        pad_amt=entry.pad_k,
                        group_size=entry.group_size,
                        out_dtype=entry.kv_dtype,
                        out_device=entry.kv_device,
                    )
                    v = symmetric_dequantize_int4_grouped_packed(
                        entry.pv,
                        entry.scale_v,
                        original_shape=entry.orig_shape_v,
                        pad_amt=entry.pad_v,
                        group_size=entry.group_size,
                        out_dtype=entry.kv_dtype,
                        out_device=entry.kv_device,
                    )
                    nl = DynamicSlidingWindowLayer(sliding_window=entry.sliding_window)
                    nl.keys = k
                    nl.values = v
                    nl.is_initialized = True
                    nl.dtype = k.dtype
                    nl.device = k.device
                    nl.cumulative_length = entry.cumulative_length
                    nl._sliding_window_tensor = entry.sliding_window_tensor_cpu.to(device=k.device)
                    new_layers.append(nl)
                elif isinstance(entry, _QuantSlidePair):
                    k = symmetric_dequantize_int8(
                        entry.qk,
                        entry.scale_k,
                        out_dtype=entry.kv_dtype,
                        out_device=entry.kv_device,
                    )
                    v = symmetric_dequantize_int8(
                        entry.qv,
                        entry.scale_v,
                        out_dtype=entry.kv_dtype,
                        out_device=entry.kv_device,
                    )
                    nl = DynamicSlidingWindowLayer(sliding_window=entry.sliding_window)
                    nl.keys = k
                    nl.values = v
                    nl.is_initialized = True
                    nl.dtype = k.dtype
                    nl.device = k.device
                    nl.cumulative_length = entry.cumulative_length
                    nl._sliding_window_tensor = entry.sliding_window_tensor_cpu.to(device=k.device)
                    new_layers.append(nl)
                elif isinstance(entry, _Int4KVPair):
                    k = symmetric_dequantize_int4_grouped_packed(
                        entry.pk,
                        entry.scale_k,
                        original_shape=entry.orig_shape_k,
                        pad_amt=entry.pad_k,
                        group_size=entry.group_size,
                        out_dtype=entry.kv_dtype,
                        out_device=entry.kv_device,
                    )
                    v = symmetric_dequantize_int4_grouped_packed(
                        entry.pv,
                        entry.scale_v,
                        original_shape=entry.orig_shape_v,
                        pad_amt=entry.pad_v,
                        group_size=entry.group_size,
                        out_dtype=entry.kv_dtype,
                        out_device=entry.kv_device,
                    )
                    nl = DynamicLayer()
                    nl.keys = k
                    nl.values = v
                    nl.is_initialized = True
                    nl.dtype = k.dtype
                    nl.device = k.device
                    new_layers.append(nl)
                elif isinstance(entry, _QuantKVPair):
                    k = symmetric_dequantize_int8(
                        entry.qk, entry.scale_k, out_dtype=entry.kv_dtype, out_device=entry.kv_device
                    )
                    v = symmetric_dequantize_int8(
                        entry.qv, entry.scale_v, out_dtype=entry.kv_dtype, out_device=entry.kv_device
                    )
                    nl = DynamicLayer()
                    nl.keys = k
                    nl.values = v
                    nl.is_initialized = True
                    nl.dtype = k.dtype
                    nl.device = k.device
                    new_layers.append(nl)
                else:
                    new_layers.append(entry)
            new_cache.layers = new_layers
            return new_cache
        finally:
            self._cumulative_dequant_s += time.perf_counter() - t0

    def _sparse_metadata_bytes(self) -> int:
        meta = 0
        idx = self._integrator.retained_global_indices
        if idx:
            meta += len(idx) * 8
        sc = self._integrator.token_scores_cpu
        if sc is not None:
            meta += int(sc.numel()) * int(sc.element_size())
        meta += 3 * 8
        return int(meta)

    def _quant_metadata_bytes(self) -> int:
        payload_ignored, meta = self._quant_payload_and_meta()
        return int(meta)

    def _quant_payload_and_meta(self) -> tuple[int, int]:
        """Quant payload bytes + metadata (scales, slide aux)."""
        payload = 0
        metadata = 0
        float32_scale_bytes = 4

        if self._tuple_entries:
            for ent in self._tuple_entries:
                if isinstance(ent, _Int4KVPair):
                    payload += int(ent.pk.numel()) + int(ent.pv.numel())
                    metadata += int(ent.scale_k.numel() * 4 + ent.scale_v.numel() * 4)
                else:
                    payload += int(ent.qk.numel()) + int(ent.qv.numel())
                    metadata += 2 * float32_scale_bytes

        if self._dynamic_entries:
            for entry in self._dynamic_entries:
                if isinstance(entry, _Int4SlidePair):
                    payload += int(entry.pk.numel()) + int(entry.pv.numel())
                    metadata += int(entry.scale_k.numel() * 4 + entry.scale_v.numel() * 4)
                    st = entry.sliding_window_tensor_cpu
                    metadata += int(st.numel() * st.element_size())
                elif isinstance(entry, _Int4KVPair):
                    payload += int(entry.pk.numel()) + int(entry.pv.numel())
                    metadata += int(entry.scale_k.numel() * 4 + entry.scale_v.numel() * 4)
                elif isinstance(entry, _QuantSlidePair):
                    payload += int(entry.qk.numel()) + int(entry.qv.numel())
                    st = entry.sliding_window_tensor_cpu
                    metadata += 2 * float32_scale_bytes + int(st.numel() * st.element_size())
                elif isinstance(entry, _QuantKVPair):
                    payload += int(entry.qk.numel()) + int(entry.qv.numel())
                    metadata += 2 * float32_scale_bytes

        return int(payload), int(metadata)

    def memory_bytes(self) -> int:
        p, qmeta = self._quant_payload_and_meta()
        sm = self._sparse_metadata_bytes()
        return int(p + qmeta + sm)

    def stats(self) -> dict[str, Any]:
        retained = len(self._integrator.retained_global_indices)
        rs = self._integrator.refresh_stats()
        mean_sparsity = (
            (rs["sum_sparsity"] / rs["sparsity_samples"]) if rs["sparsity_samples"] > 0 else 0.0
        )
        logical = self._integrator.logical_seq_len
        payload_int8, metadata_quant = self._quant_payload_and_meta()
        metadata_sparse = self._sparse_metadata_bytes()
        n_layers = (
            len(self._tuple_entries)
            if self._tuple_entries
            else (len(self._dynamic_entries) if self._dynamic_entries else 0)
        )

        seq_len: int | None = None
        if self._tuple_entries and self._tuple_entries:
            t0 = self._tuple_entries[0]
            if isinstance(t0, _Int4KVPair):
                if t0.pk.dim() >= 2:
                    seq_len = int(t0.pk.shape[-2])
            elif t0.qk.dim() >= 2:
                seq_len = int(t0.qk.shape[-2])
        elif self._dynamic_entries:
            for e in self._dynamic_entries:
                if isinstance(e, _Int4KVPair):
                    if e.pk.dim() >= 2:
                        seq_len = int(e.pk.shape[-2])
                    break
                if isinstance(e, _QuantKVPair):
                    if e.qk.dim() >= 2:
                        seq_len = int(e.qk.shape[-2])
                    break

        dense_fp16_est = 0
        if logical > 0 and retained > 0:
            if self._use_int4:
                p_fp16 = payload_int8 * 4
            else:
                p_fp16 = payload_int8 * 2
            dense_fp16_est = int(p_fp16 * (logical / float(retained)))

        scheme = (
            "symmetric_int4_per_group_packed_kv" if self._use_int4 else "symmetric_int8_per_tensor_kv"
        )
        qbits = 4 if self._use_int4 else 8
        out: dict[str, Any] = {
            "type": "KVCacheSparseQuantized",
            "composition_order": "sparsify_then_quantize_retained",
            "quant_scheme": scheme,
            "quantization_kv_bits": qbits,
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
            "metadata_bytes_sparse": int(metadata_sparse),
            "metadata_bytes_quant": int(metadata_quant),
            "metadata_bytes": int(metadata_sparse + metadata_quant),
            "memory_bytes_logical": int(payload_int8 + metadata_sparse + metadata_quant),
            "dense_fp16_kv_bytes_est": int(dense_fp16_est),
            "num_layers": int(n_layers),
            "sequence_length_est": seq_len,
            "refresh_events": int(rs["refresh_events"]),
            "append_calls": int(rs["append_calls"]),
            "cumulative_refresh_time_s": float(rs["cumulative_refresh_time_s"]),
            "cumulative_dequant_time_s": float(self._cumulative_dequant_s),
            "next_query_position_start": int(logical),
            "physical_retained_kv_len": int(retained),
            "logical_seq_len_full": int(logical),
            "sparse_hf_integration": "SparseHFCacheIntegrator",
            **memory_only_quant_stats_fragment(
                ephemeral_attention_kv_rebuild_bytes_est=self._ephemeral_attention_kv_rebuild_bytes_est(),
            ),
        }
        if self._use_int4:
            out["payload_bytes_uint8_packed"] = int(payload_int8)
            out["payload_bytes_int8"] = 0
        else:
            out["payload_bytes_int8"] = int(payload_int8)
        return out
