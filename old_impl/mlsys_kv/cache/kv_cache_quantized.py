"""INT8 **draft** KV cache — **Phase 13: memory-only quantization**.

**Quantize** on :meth:`append_from_forward_output` (narrow-bit **storage**).

**Dequantize** inside :meth:`get_attention_kv` so every ``model.forward`` still receives standard
Hugging Face FP16/BF16 ``past_key_values``. Attention therefore does **not** run on INT8 tensors;
report **memory** savings and **dequant overhead**, not implied attention speedups from KV width.

See :mod:`mlsys_kv.cache.kv_quant_semantics` for benchmark vocabulary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import torch

from mlsys_kv.cache.hf_kv_clone import clone_past_key_values, _clone_dynamic_layer
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.kv_quant_semantics import (
    ephemeral_fp16_kv_bytes_same_shape_as_int8_codes,
    memory_only_quant_stats_fragment,
)
from mlsys_kv.cache.quantization import symmetric_dequantize_int8, symmetric_quantize_int8

from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer


Format = Literal["dynamic", "tuple"]


@dataclass
class _QuantKVPair:
    """One layer’s keys/values in INT8 + per-tensor scales (CPU scalars)."""

    qk: torch.Tensor
    qv: torch.Tensor
    scale_k: torch.Tensor  # 0-dim float32 CPU
    scale_v: torch.Tensor
    kv_dtype: torch.dtype
    kv_device: torch.device


@dataclass
class _QuantSlidePair(_QuantKVPair):
    cumulative_length: int
    sliding_window: int
    sliding_window_tensor_cpu: torch.Tensor


class KVCacheQuantized(KVCacheBase):
    """Draft-only cache: symmetric INT8 **storage**; HF attention sees dequantized K/V each step."""

    __slots__ = (
        "_fmt",
        "_dynamic_entries",
        "_tuple_entries",
        "_cumulative_dequant_s",
    )

    def __init__(self) -> None:
        self._fmt: Format | None = None
        self._dynamic_entries: list[_QuantKVPair | _QuantSlidePair | Any] | None = None
        self._tuple_entries: list[_QuantKVPair] | None = None
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
                        qk, sk = symmetric_quantize_int8(layer.keys)
                        qv, sv = symmetric_quantize_int8(layer.values)
                        self._dynamic_entries.append(
                            _QuantSlidePair(
                                qk=qk,
                                qv=qv,
                                scale_k=sk,
                                scale_v=sv,
                                kv_dtype=layer.keys.dtype,
                                kv_device=layer.keys.device,
                                cumulative_length=int(layer.cumulative_length),
                                sliding_window=int(layer.sliding_window),
                                sliding_window_tensor_cpu=layer._sliding_window_tensor.detach().cpu(),
                            )
                        )
                    else:
                        self._dynamic_entries.append(_clone_dynamic_layer(layer))
                elif isinstance(layer, DynamicLayer):
                    if layer.is_initialized and layer.keys is not None:
                        qk, sk = symmetric_quantize_int8(layer.keys)
                        qv, sv = symmetric_quantize_int8(layer.values)
                        self._dynamic_entries.append(
                            _QuantKVPair(
                                qk=qk,
                                qv=qv,
                                scale_k=sk,
                                scale_v=sv,
                                kv_dtype=layer.keys.dtype,
                                kv_device=layer.keys.device,
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
            return

        raise TypeError(f"KVCacheQuantized: unsupported past type {type(cloned)}")

    def get_attention_kv(self) -> Any | None:
        # Phase 13 (memory-only): this is where compressed KV becomes high-precision tensors again.
        # There is no runtime-accelerated attention on INT8 in this codebase.
        if self._fmt is None:
            return None
        t0 = time.perf_counter()
        try:
            if self._fmt == "tuple":
                assert self._tuple_entries is not None
                layers: list[tuple[torch.Tensor, torch.Tensor]] = []
                for ent in self._tuple_entries:
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
                if isinstance(entry, _QuantSlidePair):
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

    def memory_bytes(self) -> int:
        payload, meta = self._memory_breakdown()
        return int(payload + meta)

    def _memory_breakdown(self) -> tuple[int, int]:
        """Return ``(payload_bytes, metadata_bytes)``."""
        payload = 0
        metadata = 0
        float32_scale_bytes = 4

        if self._tuple_entries:
            for ent in self._tuple_entries:
                payload += int(ent.qk.numel()) + int(ent.qv.numel())
                metadata += 2 * float32_scale_bytes

        if self._dynamic_entries:
            for entry in self._dynamic_entries:
                # ``_QuantSlidePair`` is a subclass of ``_QuantKVPair`` — check slide first.
                if isinstance(entry, _QuantSlidePair):
                    payload += int(entry.qk.numel()) + int(entry.qv.numel())
                    st = entry.sliding_window_tensor_cpu
                    metadata += 2 * float32_scale_bytes + int(st.numel() * st.element_size())
                elif isinstance(entry, _QuantKVPair):
                    payload += int(entry.qk.numel()) + int(entry.qv.numel())
                    metadata += 2 * float32_scale_bytes

        return payload, metadata

    def stats(self) -> dict[str, Any]:
        payload, metadata = self._memory_breakdown()
        seq_len: int | None = None
        n_layers = 0
        if self._tuple_entries:
            n_layers = len(self._tuple_entries)
            if self._tuple_entries:
                dq = self._tuple_entries[0]
                if dq.qk.dim() >= 2:
                    seq_len = int(dq.qk.shape[-2])
        elif self._dynamic_entries:
            n_layers = len(self._dynamic_entries)
            for e in self._dynamic_entries:
                if isinstance(e, _QuantKVPair):
                    if e.qk.dim() >= 2:
                        seq_len = int(e.qk.shape[-2])
                    break

        ephemeral = ephemeral_fp16_kv_bytes_same_shape_as_int8_codes(payload)
        return {
            "type": "KVCacheQuantized",
            "quant_scheme": "symmetric_int8_per_tensor_kv",
            "num_layers": n_layers,
            "sequence_length_est": seq_len,
            "payload_bytes_int8": int(payload),
            "metadata_bytes": int(metadata),
            "memory_bytes_logical": int(payload + metadata),
            "cumulative_dequant_time_s": float(self._cumulative_dequant_s),
            **memory_only_quant_stats_fragment(
                ephemeral_attention_kv_rebuild_bytes_est=ephemeral,
            ),
        }
