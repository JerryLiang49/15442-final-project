"""Tests for HF KV prefix crop (Phase 10)."""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from mlsys_kv.cache.hf_kv_clone import past_sequence_length
from mlsys_kv.cache.hf_kv_trim import crop_verifier_past_to_seq_len, verifier_cache_seq_len_hf


def _layer_with_len(seq: int) -> DynamicLayer:
    layer = DynamicLayer()
    layer.keys = torch.zeros(1, 2, seq, 8)
    layer.values = torch.zeros(1, 2, seq, 8)
    layer.is_initialized = True
    layer.dtype = layer.keys.dtype
    layer.device = layer.keys.device
    return layer


def test_crop_dynamic_cache_shortens() -> None:
    cache = DynamicCache()
    cache.layers = [_layer_with_len(10), _layer_with_len(10)]
    cropped = crop_verifier_past_to_seq_len(cache, 4)
    assert isinstance(cropped, DynamicCache)
    assert verifier_cache_seq_len_hf(cropped) == 4
    assert cropped.layers[0].keys.shape[-2] == 4


def test_crop_tuple_matches_length() -> None:
    k = torch.randn(1, 2, 10, 8)
    v = torch.randn(1, 2, 10, 8)
    past = ((k, v),)
    out = crop_verifier_past_to_seq_len(past, 3)
    assert past_sequence_length(out) == 3
    assert out[0][0].shape[-2] == 3


def test_crop_dynamic_idempotent_when_already_short() -> None:
    cache = DynamicCache()
    cache.layers = [_layer_with_len(5)]
    out = crop_verifier_past_to_seq_len(cache, 5)
    assert verifier_cache_seq_len_hf(out) == 5


def test_crop_raises_when_target_longer_than_cache() -> None:
    cache = DynamicCache()
    cache.layers = [_layer_with_len(3)]
    with pytest.raises(ValueError):
        crop_verifier_past_to_seq_len(cache, 99)
