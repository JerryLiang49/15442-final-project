"""INT4 packed **draft** KV — **Phase 13: memory-only quantization** (see :mod:`mlsys_kv.cache.kv_quant_semantics`).

Narrow-bit **storage** on append; :meth:`KVCacheInt4Packed.get_attention_kv` **dequantizes** to
model dtype for standard HF attention (no native INT4 matmul path here).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import torch

from mlsys_kv.cache.hf_kv_clone import clone_past_key_values, _clone_dynamic_layer
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.kv_quant_semantics import (
    ephemeral_fp16_kv_bytes_from_int4_orig_shapes,
    memory_only_quant_stats_fragment,
)
from mlsys_kv.cache.quantization_int4 import (
    _DEFAULT_GROUP,
    symmetric_dequantize_int4_grouped_packed,
    symmetric_quantize_int4_grouped_packed,
)

from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer


Format = Literal["dynamic", "tuple"]


@dataclass
class _Int4KVPair:
    """One layer's keys/values as packed ``uint8`` + per-group FP32 scales on CPU."""

    pk: torch.Tensor
    pv: torch.Tensor
    scale_k: torch.Tensor
    scale_v: torch.Tensor
    pad_k: int
    pad_v: int
    orig_shape_k: tuple[int, ...]
    orig_shape_v: tuple[int, ...]
    group_size: int
    kv_dtype: torch.dtype
    kv_device: torch.device


@dataclass
class _Int4SlidePair(_Int4KVPair):
    cumulative_length: int
    sliding_window: int
    sliding_window_tensor_cpu: torch.Tensor


class KVCacheInt4Packed(KVCacheBase):
    """Draft-only: INT4 **storage**; each forward rebuilds FP16/BF16 KV for HF attention."""

    __slots__ = ("_fmt", "_dynamic_entries", "_tuple_entries", "_cumulative_dequant_s", "_group_size")

    def __init__(self, group_size: int = _DEFAULT_GROUP) -> None:
        if group_size % 2 != 0:
            raise ValueError("group_size must be even")
        self._group_size = int(group_size)
        self._fmt: Format | None = None
        self._dynamic_entries: list[_Int4KVPair | _Int4SlidePair | Any] | None = None
        self._tuple_entries: list[_Int4KVPair] | None = None
        self._cumulative_dequant_s: float = 0.0

    def append_from_forward_output(self, past_key_values: Any) -> None:
        cloned = clone_past_key_values(past_key_values)
        if cloned is None:
            self._fmt = None
            self._dynamic_entries = None
            self._tuple_entries = None
            return

        if isinstance(cloned, DynamicCache):
            self._fmt = "dynamic"
            self._tuple_entries = None
            self._dynamic_entries = []
            for layer in cloned.layers:
                if isinstance(layer, DynamicSlidingWindowLayer):
                    if layer.is_initialized and layer.keys is not None:
                        k, v = layer.keys, layer.values
                        pk, sk, pad_k, ok = symmetric_quantize_int4_grouped_packed(k, self._group_size)
                        pv, sv, pad_v, ov = symmetric_quantize_int4_grouped_packed(v, self._group_size)
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
                                group_size=self._group_size,
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
                        pk, sk, pad_k, ok = symmetric_quantize_int4_grouped_packed(k, self._group_size)
                        pv, sv, pad_v, ov = symmetric_quantize_int4_grouped_packed(v, self._group_size)
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
                                group_size=self._group_size,
                                kv_dtype=k.dtype,
                                kv_device=k.device,
                            )
                        )
                    else:
                        self._dynamic_entries.append(_clone_dynamic_layer(layer))
                else:
                    self._dynamic_entries.append(_clone_dynamic_layer(layer))
            return

        if isinstance(cloned, tuple):
            self._fmt = "tuple"
            self._dynamic_entries = None
            self._tuple_entries = []
            for item in cloned:
                if item is None:
                    continue
                k, v = item[0], item[1]
                pk, sk, pad_k, ok = symmetric_quantize_int4_grouped_packed(k, self._group_size)
                pv, sv, pad_v, ov = symmetric_quantize_int4_grouped_packed(v, self._group_size)
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
                        group_size=self._group_size,
                        kv_dtype=k.dtype,
                        kv_device=k.device,
                    )
                )
            return

        raise TypeError(f"KVCacheInt4Packed: unsupported past type {type(cloned)}")

    def get_attention_kv(self) -> Any | None:
        # Phase 13 (memory-only): dequant path — HF attention still uses high-precision K/V.
        if self._fmt is None:
            return None
        t0 = time.perf_counter()
        try:
            if self._fmt == "tuple":
                assert self._tuple_entries is not None
                layers: list[tuple[torch.Tensor, torch.Tensor]] = []
                for ent in self._tuple_entries:
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
                else:
                    new_layers.append(entry)
            new_cache.layers = new_layers
            return new_cache
        finally:
            self._cumulative_dequant_s += time.perf_counter() - t0

    def _memory_breakdown(self) -> tuple[int, int]:
        payload = 0
        metadata = 0

        def _acc_int4(ent: _Int4KVPair) -> None:
            nonlocal payload, metadata
            payload += int(ent.pk.numel()) + int(ent.pv.numel())
            metadata += int(ent.scale_k.numel() * 4 + ent.scale_v.numel() * 4)

        if self._tuple_entries:
            for ent in self._tuple_entries:
                _acc_int4(ent)

        if self._dynamic_entries:
            for entry in self._dynamic_entries:
                if isinstance(entry, _Int4SlidePair):
                    _acc_int4(entry)
                    st = entry.sliding_window_tensor_cpu
                    metadata += int(st.numel() * st.element_size())
                elif isinstance(entry, _Int4KVPair):
                    _acc_int4(entry)

        return payload, metadata

    def memory_bytes(self) -> int:
        p, m = self._memory_breakdown()
        return int(p + m)

    def stats(self) -> dict[str, Any]:
        payload, metadata = self._memory_breakdown()
        seq_len: int | None = None
        n_layers = 0
        if self._tuple_entries:
            n_layers = len(self._tuple_entries)
            if self._tuple_entries and self._tuple_entries[0].pk.dim() >= 2:
                seq_len = int(self._tuple_entries[0].pk.shape[-2])
        elif self._dynamic_entries:
            n_layers = len(self._dynamic_entries)
            for e in self._dynamic_entries:
                if isinstance(e, _Int4KVPair):
                    if e.pk.dim() >= 2:
                        seq_len = int(e.pk.shape[-2])
                    break

        ephemeral = ephemeral_fp16_kv_bytes_from_int4_orig_shapes(
            tuple_entries=self._tuple_entries,
            dynamic_entries=self._dynamic_entries,
        )
        return {
            "type": "KVCacheInt4Packed",
            "quant_scheme": "symmetric_int4_per_group_packed_kv",
            "quantization_kv_bits": 4,
            "group_size": int(self._group_size),
            "num_layers": n_layers,
            "sequence_length_est": seq_len,
            "payload_bytes_uint8_packed": int(payload),
            "metadata_bytes": int(metadata),
            "memory_bytes_logical": int(payload + metadata),
            "cumulative_dequant_time_s": float(self._cumulative_dequant_s),
            **memory_only_quant_stats_fragment(
                ephemeral_attention_kv_rebuild_bytes_est=ephemeral,
            ),
        }
