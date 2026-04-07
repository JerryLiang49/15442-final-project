"""Crop Hugging Face ``past_key_values`` to a **prefix** of absolute sequence length.

**Phase 10 invariant**

After speculative **block** verification, ``past_key_values`` may reflect **all** ``K`` draft
positions. On partial rejection we must keep only positions ``0 .. L0+j-1`` where ``L0`` is
the verifier length at **round start** and ``j`` is the count of **accepted** draft tokens.
We then run **one** forward on the **correction** token so the cache ends at ``L0+j`` tokens
with KV matching greedy continuation.

**Hugging Face / Transformers**

* :class:`transformers.cache_utils.DynamicCache` — uses per-layer ``crop(max_length)``, which
  retains key/value slices ``[..., :max_length, :]`` (see ``DynamicLayer.crop`` in Transformers
  **≥ 4.40** style APIs used by this project).
* ``DynamicSlidingWindowLayer.crop`` **raises** if the logical sequence has already crossed the
  sliding window (states are no longer fully reconstructible). Callers must **catch**
  ``ValueError`` and fall back (e.g. replay from round-start ``past_key_values``).

**Batch size**

Only **batch 1** speculative decoding is supported; cropped tensors are not validated per-batch.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer

from mlsys_kv.cache.hf_kv_clone import clone_past_key_values, past_sequence_length

logger = logging.getLogger(__name__)


def past_contains_sliding_window_layer(past: Any) -> bool:
    """Return True if any layer is a HuggingFace sliding-window dynamic layer."""
    if isinstance(past, DynamicCache):
        return any(isinstance(layer, DynamicSlidingWindowLayer) for layer in past.layers)
    return False


def crop_verifier_past_to_seq_len(past: Any, target_seq_len: int) -> Any:
    """Return a **clone** of ``past`` cropped so the verifier cache holds prefix ``0..target_seq_len-1``.

    **Postcondition (standard ``DynamicLayer`` / tuple caches):**
    ``past_sequence_length(result) == target_seq_len``.

    Args:
        past: HF ``past_key_values`` (``DynamicCache`` or ``tuple`` of per-layer ``(k,v)``).
        target_seq_len: Desired **absolute** sequence length stored in KV (non-negative).

    Returns:
        New cache object (clone + crop).

    Raises:
        ValueError: Propagated from ``DynamicSlidingWindowLayer.crop`` when cropping is unsafe.
        TypeError: Unsupported ``past`` layout.
    """
    if target_seq_len < 0:
        raise ValueError(f"target_seq_len must be >= 0, got {target_seq_len}")

    cloned = clone_past_key_values(past)
    if cloned is None:
        raise TypeError("Cannot crop None past_key_values")

    if isinstance(cloned, DynamicCache):
        cur = cloned.get_seq_length(0)
        if cur < target_seq_len:
            raise ValueError(
                f"cannot crop to length {target_seq_len}: cache only has length {cur}"
            )
        cloned.crop(target_seq_len)
        after = cloned.get_seq_length(0)
        if after != target_seq_len:
            logger.warning(
                "crop_verifier_past_to_seq_len: expected length %s after crop, got %s "
                "(possible sliding-window / mixed-cache layout)",
                target_seq_len,
                after,
            )
        return cloned

    if isinstance(cloned, tuple):
        out: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for item in cloned:
            if item is None:
                out.append(None)
                continue
            k_t, v_t = item[0], item[1]
            if k_t.shape[-2] < target_seq_len:
                raise ValueError(
                    f"cannot crop to length {target_seq_len}: layer has length {k_t.shape[-2]}"
                )
            nk = k_t[..., :target_seq_len, :].contiguous()
            nv = v_t[..., :target_seq_len, :].contiguous()
            out.append((nk, nv))
        result = tuple(out)
        if past_sequence_length(result) != target_seq_len:
            raise RuntimeError(
                f"tuple crop invariant failed: wanted {target_seq_len}, "
                f"got {past_sequence_length(result)}"
            )
        return result

    raise TypeError(f"Unsupported past_key_values type for crop: {type(past)}")


def verifier_cache_seq_len_hf(past: Any) -> int:
    """Best-effort logical KV length (batch 1); prefers :meth:`DynamicCache.get_seq_length`."""
    if past is None:
        return 0
    if isinstance(past, DynamicCache):
        return int(past.get_seq_length(0))
    return past_sequence_length(past)
