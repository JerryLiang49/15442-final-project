"""Regression tests for hierarchical + dense speculative decoding."""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from cache.hf_past_adapters import hierarchical_view_to_past_key_values

from decoding.speculative_dense_hierarchical import (
    SpeculativeDecoderDenseHierarchical,
    format_hierarchical_debug_line,
)


@pytest.fixture(scope="module")
def tiny_gpt2():
    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=128,
        vocab_size=50257,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return model, tok


def test_hierarchical_matches_autoregressive_short(tiny_gpt2) -> None:
    model, tok = tiny_gpt2
    prompt = "Hello"
    max_new = 16
    dec = SpeculativeDecoderDenseHierarchical(
        model,
        tok,
        gamma=4,
        G=8,
        verify_match_autoregressive=True,
    )
    r = dec.decode(prompt, max_new_tokens=max_new)
    assert r.full_token_ids.shape[1] == tok(prompt, return_tensors="pt")["input_ids"].shape[1] + max_new
    assert r.last_round is not None
    line = format_hierarchical_debug_line(r.last_round)
    assert "hist=" in line and "cf1=" in line
    dbg = r.mgr.instrumentation_dict()
    assert "cf1_len" in dbg and "hist_len" in dbg


def test_target_view_without_cf2_matches_when_cf2_empty(tiny_gpt2) -> None:
    model, tok = tiny_gpt2
    dec = SpeculativeDecoderDenseHierarchical(model, tok, gamma=2, G=16)
    inputs = tok("Hi there", return_tensors="pt")
    with torch.inference_mode():
        pre = model(**inputs, use_cache=True)
    from decoding.speculative_dense_hierarchical import sync_hierarchical_store_from_hf_past

    sync_hierarchical_store_from_hf_past(dec.mgr, pre.past_key_values)
    t0 = dec.mgr.target_view_without_cf2()
    t1 = dec.mgr.target_view()
    assert len(t0.layers_k) == len(t1.layers_k)
    for a, b in zip(t0.layers_k, t1.layers_k):
        assert torch.equal(a, b)


def test_regression_various_gamma(tiny_gpt2) -> None:
    model, tok = tiny_gpt2
    for g in (1, 2, 5):
        dec = SpeculativeDecoderDenseHierarchical(
            model, tok, gamma=g, G=8, verify_match_autoregressive=True
        )
        dec.decode("The cat", max_new_tokens=24)


def test_view_to_past_none_on_empty() -> None:
    from cache.hierarchical_kv_store import HierarchicalKVStore
    from cache.recent_buffer import RecentBufferManager

    st = HierarchicalKVStore(num_layers=1, num_heads=1, head_dim=4, batch_size=1, G=4, device=torch.device("cpu"))
    mgr = RecentBufferManager(st)
    assert hierarchical_view_to_past_key_values(mgr.draft_view(), config=None) is None
