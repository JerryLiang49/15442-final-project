"""Phase 11 debug: log how sparse draft KV lines up with Hugging Face ``position_ids`` / cache length.

Call :func:`audit_sparse_draft_state` from tests or exploratory scripts when
``R < L`` (physical retained length vs logical full length).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mlsys_kv.cache.hf_kv_trim import verifier_cache_seq_len_hf
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse
from mlsys_kv.cache.kv_cache_sparse_quantized import KVCacheSparseQuantized


@dataclass
class SparseKVPositionAudit:
    """Snapshot for one decode step (batch size 1)."""

    logical_full_len: int
    physical_hf_cache_len: int | None
    retained_indices_count: int
    retained_global_index_min: int | None
    retained_global_index_max: int | None
    next_query_position_ids: list[int] | None
    message: str


def _next_pos_list(
    cache: KVCacheSparse | KVCacheSparseQuantized, query_length: int, device: torch.device
) -> list[int] | None:
    t = cache.position_ids_for_next_queries(
        query_length, batch_size=1, device=device
    )
    if t is None:
        return None
    return [int(x) for x in t[0].tolist()]


def audit_sparse_draft_state(
    cache: KVCacheSparse | KVCacheSparseQuantized,
    *,
    past_for_model: Any | None,
    query_length: int = 1,
    device: torch.device | None = None,
) -> SparseKVPositionAudit:
    """Record logical vs physical lengths and intended ``position_ids`` row.

    Args:
        cache: Sparse or sparse-quant draft cache after :meth:`append_from_forward_output`.
        past_for_model: Value from :meth:`KVCacheBase.get_attention_kv` (may be ``None``).
        query_length: Decode step length (usually ``1``).
        device: Device for position tensor; defaults to CPU.
    """
    dev = device or torch.device("cpu")
    phys: int | None = None
    if past_for_model is not None:
        try:
            phys = int(verifier_cache_seq_len_hf(past_for_model))
        except Exception:
            phys = None

    idx = cache.retained_global_indices
    idx_min = min(idx) if idx else None
    idx_max = max(idx) if idx else None

    L = int(cache.logical_seq_len)
    R = len(idx)
    pos_l = _next_pos_list(cache, query_length, dev)

    parts = [
        f"logical_L={L}",
        f"retained_R={R}",
        f"physical_hf_cache_len={phys}",
        f"retained_global_range=[{idx_min},{idx_max}]",
        f"next_query_position_ids={pos_l}",
    ]
    if L != R and phys == R:
        parts.append(
            "NOTE: HF cache get_seq_length is physical R; explicit position_ids must start at L."
        )

    return SparseKVPositionAudit(
        logical_full_len=L,
        physical_hf_cache_len=phys,
        retained_indices_count=R,
        retained_global_index_min=idx_min,
        retained_global_index_max=idx_max,
        next_query_position_ids=pos_l,
        message="; ".join(parts),
    )
