"""Bridge to Phase L cache mutation helpers (quantize CF1 → hist).

The implementation lives in :mod:`cache.kv_cache_ops` to stay next to
:class:`cache.hierarchical_kv_store.HierarchicalKVStore` and avoid import cycles.
"""

from __future__ import annotations

from cache.kv_cache_ops import append_quantized_cf1_to_hist, quantize_cf1_fp16_to_int4

__all__ = [
    "append_quantized_cf1_to_hist",
    "quantize_cf1_fp16_to_int4",
]
