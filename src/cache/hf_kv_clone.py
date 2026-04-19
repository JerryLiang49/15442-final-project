"""Clone Hugging Face ``past_key_values`` for independent draft/verifier paths."""

from __future__ import annotations

from typing import Any

import torch

from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer


def clone_past_key_values(past: Any) -> Any:
    """Deep-clone KV cache tensors (non-leaf safe)."""
    if past is None:
        return None

    if isinstance(past, tuple):
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for item in past:
            if item is None:
                continue
            k, v = item[0], item[1]
            out.append((k.detach().clone(), v.detach().clone()))
        return tuple(out)

    if isinstance(past, DynamicCache):
        new_cache = DynamicCache()
        new_layers: list[Any] = []
        for layer in past.layers:
            new_layers.append(_clone_dynamic_layer(layer))
        new_cache.layers = new_layers
        return new_cache

    raise TypeError(f"Unsupported past_key_values type: {type(past)}")


def past_sequence_length(past: Any) -> int:
    """Key-sequence length from first initialized KV layer."""
    if past is None:
        return 0
    if isinstance(past, tuple):
        for item in past:
            if item is None:
                continue
            k = item[0]
            if k is not None and k.ndim >= 2 and k.shape[-2] > 0:
                return int(k.shape[-2])
        return 0
    if isinstance(past, DynamicCache):
        for layer in past.layers:
            if isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer)):
                if layer.is_initialized and layer.keys is not None and layer.keys.shape[-2] > 0:
                    return int(layer.keys.shape[-2])
        return 0
    raise TypeError(f"Unsupported past_key_values type: {type(past)}")


def _clone_dynamic_layer(layer: Any) -> Any:
    if isinstance(layer, DynamicSlidingWindowLayer):
        nl = DynamicSlidingWindowLayer(sliding_window=layer.sliding_window)
        if layer.is_initialized and layer.keys is not None and layer.keys.numel() > 0:
            nl.keys = layer.keys.detach().clone()
            nl.values = layer.values.detach().clone()
            nl.is_initialized = True
            nl.dtype = nl.keys.dtype
            nl.device = nl.keys.device
            nl.cumulative_length = int(layer.cumulative_length)
            nl._sliding_window_tensor = layer._sliding_window_tensor.to(device=nl.device)
        return nl

    if isinstance(layer, DynamicLayer):
        nl = DynamicLayer()
        if layer.is_initialized and layer.keys is not None and layer.keys.numel() > 0:
            nl.keys = layer.keys.detach().clone()
            nl.values = layer.values.detach().clone()
            nl.is_initialized = True
            nl.dtype = nl.keys.dtype
            nl.device = nl.keys.device
        return nl

    raise TypeError(f"Unsupported cache layer type: {type(layer)}")
