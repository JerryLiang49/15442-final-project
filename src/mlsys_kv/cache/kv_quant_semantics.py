"""Explicit **draft KV quantization semantics** (Phase 13).

This project currently implements only **memory-only** draft quantization: keys and values are
stored in narrow-bit form, but :meth:`~mlsys_kv.cache.kv_cache_base.KVCacheBase.get_attention_kv`
**dequantizes** back to the model’s high-precision dtype before each ``model.forward``, so
standard Hugging Face attention still runs on FP16/BF16 tensors.

**Do not** claim end-to-end decode speedups from INT8/INT4 KV **storage alone** without also
measuring (or implementing) a path where attention **natively** consumes packed KV — that would
be :attr:`DraftKVQuantizationSemantics.RUNTIME_ACCELERATED` (not implemented here).

Extensibility
-------------
Future work can add :attr:`DraftKVQuantizationSemantics.RUNTIME_ACCELERATED`, wire caches to pass
packed KV into a custom attention kernel, and flip the benchmark metadata fields below without
renaming this module’s vocabulary.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final

# --- Single source of truth for what the codebase actually does today ---

IMPLEMENTED_DRAFT_KV_QUANTIZATION_SEMANTICS: Final[str] = "memory_only"
"""Value persisted on metrics / stats rows for all quantized draft caches in this repo."""


class DraftKVQuantizationSemantics(str, Enum):
    """How draft KV quantization relates to the attention matmul."""

    MEMORY_ONLY = "memory_only"
    """Store KV narrow-bit; rebuild high-precision K/V for each forward (current implementation)."""

    RUNTIME_ACCELERATED = "runtime_accelerated"
    """Attention would consume compressed KV directly (placeholder; not implemented)."""


def memory_only_quant_stats_fragment(
    *,
    ephemeral_attention_kv_rebuild_bytes_est: int,
) -> dict[str, Any]:
    """Uniform Phase 13 fields for :meth:`KVCacheBase.stats` on memory-only quantized draft caches.

    Payload and scale/metadata byte totals remain on each cache’s existing keys
    (e.g. ``payload_bytes_int8``, ``metadata_bytes``).

    ``ephemeral_attention_kv_rebuild_bytes_est`` approximates **temporary** high-precision K+V bytes
    allocated while building HF ``past_key_values`` inside :meth:`~mlsys_kv.cache.kv_cache_base.KVCacheBase.get_attention_kv`.
    """
    return {
        "kv_quantization_semantics": IMPLEMENTED_DRAFT_KV_QUANTIZATION_SEMANTICS,
        "runtime_accelerated_quant_attention": False,
        "attention_consumes_dequantized_kv": True,
        "ephemeral_attention_kv_rebuild_bytes_est": int(ephemeral_attention_kv_rebuild_bytes_est),
        "claim_decode_speedup_from_kv_quant_alone": False,
    }


def ephemeral_fp16_kv_bytes_same_shape_as_int8_codes(payload_int8_element_bytes: int) -> int:
    """INT8 stores one byte per K/V element; FP16 attention tensors use ~2x that byte count."""
    return int(payload_int8_element_bytes) * 2


def ephemeral_fp16_kv_bytes_from_int4_orig_shapes(
    *,
    tuple_entries: list[Any] | None,
    dynamic_entries: list[Any] | None,
) -> int:
    """Sum ``(numel K + numel V) * 2`` from INT4 entries’ original shapes (FP16/BF16 width)."""
    total_elems = 0

    def acc(ent: Any) -> None:
        nonlocal total_elems
        ok = getattr(ent, "orig_shape_k", None)
        ov = getattr(ent, "orig_shape_v", None)
        if ok is not None and ov is not None:
            from math import prod

            total_elems += int(prod(ok)) + int(prod(ov))

    if tuple_entries:
        for ent in tuple_entries:
            acc(ent)
    if dynamic_entries:
        for ent in dynamic_entries:
            if getattr(ent, "pk", None) is not None:
                acc(ent)

    return int(total_elems * 2)
