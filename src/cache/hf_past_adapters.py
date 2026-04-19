"""Convert between Hugging Face ``past_key_values`` and hierarchical KV views (no custom kernels)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedConfig
from transformers.cache_utils import DynamicCache

from .hierarchical_kv_store import HierarchicalKVView


def _iter_kv_pairs(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Flatten HF ``past_key_values`` into ``(key, value)`` pairs per layer (order 0..L-1)."""
    if past_key_values is None:
        return []
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    key_cache = getattr(past_key_values, "key_cache", None)
    val_cache = getattr(past_key_values, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(val_cache, list):
        for k, v in zip(key_cache, val_cache):
            if k is None or v is None:
                continue
            pairs.append((k, v))
        return pairs

    for layer in past_key_values:
        if layer is None:
            continue
        pairs.append((layer[0], layer[1]))
    return pairs


def hf_past_to_layer_lists(past_key_values: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split HF past into ``layers_k``, ``layers_v`` (each length == num_layers)."""
    pairs = _iter_kv_pairs(past_key_values)
    layers_k = [p[0] for p in pairs]
    layers_v = [p[1] for p in pairs]
    return layers_k, layers_v


def hierarchical_view_to_tuple_past(view: HierarchicalKVView) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Legacy tuple ``past_key_values`` (older Transformers). Prefer :func:`hierarchical_view_to_dynamic_cache`."""
    if len(view.layers_k) != len(view.layers_v):
        raise ValueError("layers_k / layers_v length mismatch")
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in zip(view.layers_k, view.layers_v):
        out.append((k.contiguous(), v.contiguous()))
    return tuple(out)


def hierarchical_view_to_dynamic_cache(
    view: HierarchicalKVView,
    *,
    config: PreTrainedConfig | None = None,
) -> DynamicCache:
    """``DynamicCache`` for modern ``model(..., past_key_values=...)`` (e.g. GPT-2)."""
    tup = hierarchical_view_to_tuple_past(view)
    ddp = tuple((k, v) for k, v in tup)
    return DynamicCache(ddp_cache_data=ddp, config=config)


def hierarchical_view_to_past_key_values(
    view: HierarchicalKVView,
    *,
    config: PreTrainedConfig | None = None,
) -> DynamicCache | None:
    """``None`` if zero-length cache; else :class:`~transformers.cache_utils.DynamicCache`."""
    if not view.layers_k:
        return None
    if int(view.layers_k[0].shape[2]) == 0:
        return None
    return hierarchical_view_to_dynamic_cache(view, config=config)


def extract_last_token_kv_per_layer(past_key_values: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """For each layer, return K/V with time dim 1 (last absolute position)."""
    pairs = _iter_kv_pairs(past_key_values)
    nk = [p[0][..., -1:, :].contiguous() for p in pairs]
    nv = [p[1][..., -1:, :].contiguous() for p in pairs]
    return nk, nv
