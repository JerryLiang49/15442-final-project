"""Construct draft :class:`~mlsys_kv.cache.kv_cache_base.KVCacheBase` instances by mode.

Sparse modes delegate HF/cache reconciliation to
:class:`~mlsys_kv.cache.sparse_hf_integration.SparseHFCacheIntegrator` (Phase 12). Reusing a
sparse cache object across prompts requires :meth:`~mlsys_kv.cache.kv_cache_base.KVCacheBase.reset`.
"""

from __future__ import annotations

from typing import Any

from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.hf_kv_clone import clone_past_key_values
from mlsys_kv.cache.kv_cache_fp16 import KVCacheFP16
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.kv_cache_int4 import KVCacheInt4Packed
from mlsys_kv.cache.kv_cache_quantized import KVCacheQuantized
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse
from mlsys_kv.cache.kv_cache_sparse_quantized import KVCacheSparseQuantized


def create_draft_cache(
    mode: DraftCacheMode,
    *,
    model: Any | None = None,
    sparse_config: SparseRetentionConfig | None = None,
    kv_quant_bits: int = 8,
) -> KVCacheBase:
    """Return a **new empty** draft cache for ``mode``.

    Args:
        mode: Draft backend selector.
        model: Optional HF model (required for attention scoring in sparse mode).
        sparse_config: Optional :class:`SparseRetentionConfig` for ``sparse_only``
            and ``sparse_quant``.

    Raises:
        NotImplementedError: For unknown modes.
    """
    if mode is DraftCacheMode.FP16:
        return KVCacheFP16()
    if mode is DraftCacheMode.QUANT_ONLY:
        if int(kv_quant_bits) == 4:
            return KVCacheInt4Packed()
        return KVCacheQuantized()
    if mode is DraftCacheMode.SPARSE_ONLY:
        cfg = sparse_config or SparseRetentionConfig()
        return KVCacheSparse(cfg, model=model)
    if mode is DraftCacheMode.SPARSE_QUANT:
        cfg = sparse_config or SparseRetentionConfig()
        return KVCacheSparseQuantized(cfg, model=model, use_int4=(int(kv_quant_bits) == 4))
    raise NotImplementedError(f"Unknown draft cache mode: {mode!r}")


def draft_cache_from_verifier_snapshot(
    mode: DraftCacheMode,
    verifier_past: Any,
    *,
    model: Any | None = None,
    sparse_config: SparseRetentionConfig | None = None,
    kv_quant_bits: int = 8,
) -> KVCacheBase:
    """Build a draft cache whose internal state matches a **clone** of verifier HF ``past_key_values``."""
    cache = create_draft_cache(
        mode, model=model, sparse_config=sparse_config, kv_quant_bits=kv_quant_bits
    )
    cache.append_from_forward_output(clone_past_key_values(verifier_past))
    return cache
