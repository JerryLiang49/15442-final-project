"""Crop Hugging Face ``past_key_values`` to a committed prefix length."""

from __future__ import annotations

import logging
from typing import Any

import torch

from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer

from .hf_kv_clone import clone_past_key_values, past_sequence_length

logger = logging.getLogger(__name__)


def past_contains_sliding_window_layer(past: Any) -> bool:
    """True if any layer uses HF sliding-window dynamic cache."""
    if isinstance(past, DynamicCache):
        return any(isinstance(layer, DynamicSlidingWindowLayer) for layer in past.layers)
    return False


def verifier_cache_seq_len_hf(past: Any) -> int:
    """Logical KV length (batch 1)."""
    if past is None:
        return 0
    if isinstance(past, DynamicCache):
        return int(past.get_seq_length(0))
    return past_sequence_length(past)


def crop_verifier_past_to_seq_len(past: Any, target_seq_len: int) -> Any:
    """Clone and crop ``past`` so KV holds absolute positions ``0 .. target_seq_len - 1``."""
    if target_seq_len < 0:
        raise ValueError(f"target_seq_len must be >= 0, got {target_seq_len}")

    cloned = clone_past_key_values(past)
    if cloned is None:
        raise TypeError("Cannot crop None past_key_values")

    if isinstance(cloned, DynamicCache):
        cur = cloned.get_seq_length(0)
        if cur < target_seq_len:
            raise ValueError(f"cannot crop to length {target_seq_len}: cache only has length {cur}")
        cloned.crop(target_seq_len)
        after = cloned.get_seq_length(0)
        if after != target_seq_len:
            logger.warning(
                "crop_verifier_past_to_seq_len: expected length %s after crop, got %s",
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
                f"tuple crop invariant failed: wanted {target_seq_len}, got {past_sequence_length(result)}"
            )
        return result

    raise TypeError(f"Unsupported past_key_values type for crop: {type(cloned)}")
