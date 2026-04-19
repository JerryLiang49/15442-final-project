"""Draft KV backend selection (verifier always uses full FP16 — see :class:`~mlsys_kv.cache.kv_cache_fp16.KVCacheFP16`)."""

from __future__ import annotations

from enum import Enum


class DraftCacheMode(str, Enum):
    """Which draft cache implementation to attach to the speculative draft path.

    ``quant_only`` and ``sparse_quant`` use **memory-only** narrow-bit KV storage (Phase 13):
    attention still runs on dequantized tensors — see :mod:`mlsys_kv.cache.kv_quant_semantics`.
    """

    FP16 = "fp16"
    QUANT_ONLY = "quant_only"
    SPARSE_ONLY = "sparse_only"
    SPARSE_QUANT = "sparse_quant"

    @classmethod
    def from_string(cls, value: str) -> DraftCacheMode:
        v = (value or "").strip().lower()
        for m in cls:
            if m.value == v:
                return m
        raise ValueError(f"Unknown draft cache mode: {value!r}; expected one of {[e.value for e in cls]}")
