"""Clone Hugging Face ``past_key_values`` without ``deepcopy`` (non-leaf tensors)."""

from __future__ import annotations

from typing import Any

import torch

from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer


def clone_past_key_values(past: Any) -> Any:
    """Deep-clone KV cache tensors for independent draft/verifier paths.

    Supports:
        * ``tuple`` of ``(key, value)`` per layer (legacy).
        * :class:`transformers.cache_utils.DynamicCache` with ``DynamicLayer`` / sliding layers.

    Raises:
        TypeError: If the cache layout is not recognized.
    """
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
    """Return key-sequence length ``L`` (``...,-2``) from first initialized KV layer."""
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


def strip_last_position_from_past(past: Any) -> Any:
    """Drop the most recent KV position from each layer (decode step undo for scoring).

    If a layer has length ``0`` or ``1``, returns that layer with length ``0`` KV
    (caller must avoid running a forward that requires non-empty prefix when invalid).
    """
    if past is None:
        return None
    if isinstance(past, tuple):
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for item in past:
            if item is None:
                continue
            k, v = item[0], item[1]
            if k.shape[-2] <= 1:
                out.append((k[..., :0, :], v[..., :0, :]))
            else:
                out.append((k[..., :-1, :], v[..., :-1, :]))
        return tuple(out)
    if isinstance(past, DynamicCache):
        new_cache = DynamicCache()
        new_layers: list[Any] = []
        for layer in past.layers:
            new_layers.append(_strip_last_dynamic_layer(layer))
        new_cache.layers = new_layers
        return new_cache
    raise TypeError(f"Unsupported past_key_values type: {type(past)}")


def _strip_last_dynamic_layer(layer: Any) -> Any:
    if isinstance(layer, DynamicSlidingWindowLayer):
        nl = DynamicSlidingWindowLayer(sliding_window=layer.sliding_window)
        if not layer.is_initialized or layer.keys is None or layer.keys.numel() == 0:
            return nl
        k, v = layer.keys, layer.values
        if k.shape[-2] <= 1:
            nl.keys = k[..., :0, :]
            nl.values = v[..., :0, :]
            nl.cumulative_length = 0
        else:
            nl.keys = k[..., :-1, :]
            nl.values = v[..., :-1, :]
            nl.cumulative_length = max(0, int(layer.cumulative_length) - 1)
        nl.is_initialized = nl.keys.shape[-2] > 0
        if nl.is_initialized:
            nl.dtype = nl.keys.dtype
            nl.device = nl.keys.device
            nl._sliding_window_tensor = layer._sliding_window_tensor.to(device=nl.device)
        return nl

    if isinstance(layer, DynamicLayer):
        nl = DynamicLayer()
        if not layer.is_initialized or layer.keys is None or layer.keys.numel() == 0:
            return nl
        k, v = layer.keys, layer.values
        if k.shape[-2] <= 1:
            nl.keys = k[..., :0, :]
            nl.values = v[..., :0, :]
        else:
            nl.keys = k[..., :-1, :]
            nl.values = v[..., :-1, :]
        nl.is_initialized = nl.keys.shape[-2] > 0
        if nl.is_initialized:
            nl.dtype = nl.keys.dtype
            nl.device = nl.keys.device
        return nl

    raise TypeError(f"Unsupported cache layer type: {type(layer)}")


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
