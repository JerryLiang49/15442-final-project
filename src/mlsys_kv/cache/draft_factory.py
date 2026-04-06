"""Construct draft :class:`~mlsys_kv.cache.kv_cache_base.KVCacheBase` instances by mode."""

from __future__ import annotations

from typing import Any

from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.hf_kv_clone import clone_past_key_values
from mlsys_kv.cache.kv_cache_fp16 import KVCacheFP16
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.kv_cache_quantized import KVCacheQuantized
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse


def create_draft_cache(
    mode: DraftCacheMode,
    *,
    model: Any | None = None,
    sparse_config: SparseRetentionConfig | None = None,
) -> KVCacheBase:
    """Return a **new empty** draft cache for ``mode``.

    Args:
        mode: Draft backend selector.
        model: Optional HF model (required for attention scoring in sparse mode).
        sparse_config: Optional :class:`SparseRetentionConfig` for ``sparse_only``.

    Raises:
        NotImplementedError: For ``sparse_quant`` when not implemented.
    """
    if mode is DraftCacheMode.FP16:
        return KVCacheFP16()
    if mode is DraftCacheMode.QUANT_ONLY:
        return KVCacheQuantized()
    if mode is DraftCacheMode.SPARSE_ONLY:
        cfg = sparse_config or SparseRetentionConfig()
        return KVCacheSparse(cfg, model=model)
    if mode is DraftCacheMode.SPARSE_QUANT:
        raise NotImplementedError(
            "Draft cache mode 'sparse_quant' is not implemented yet (sparse + quant hybrid)."
        )
    raise NotImplementedError(f"Unknown draft cache mode: {mode!r}")


def draft_cache_from_verifier_snapshot(
    mode: DraftCacheMode,
    verifier_past: Any,
    *,
    model: Any | None = None,
    sparse_config: SparseRetentionConfig | None = None,
) -> KVCacheBase:
    """Build a draft cache whose internal state matches a **clone** of verifier HF ``past_key_values``."""
    cache = create_draft_cache(mode, model=model, sparse_config=sparse_config)
    cache.append_from_forward_output(clone_past_key_values(verifier_past))
    return cache
