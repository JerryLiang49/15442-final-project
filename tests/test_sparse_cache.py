"""Unit tests for sparse draft retention helpers (no speculative loop)."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlsys_kv.cache.hf_kv_clone import (
    clone_past_key_values,
    past_sequence_length,
    strip_last_position_from_past,
)
from mlsys_kv.cache.heavy_hitter_selector import (
    SparseRetentionConfig,
    build_full_length_scores_from_attention_prefix,
    key_norm_token_scores,
    select_retained_token_indices,
)
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse


def test_select_retained_preserves_recent_window() -> None:
    L = 20
    scores = torch.arange(L, dtype=torch.float32)
    idx = select_retained_token_indices(
        L, scores, recent_window=5, heavy_hitter_budget=3
    )
    assert idx == sorted(set(idx))
    assert idx[-5:] == list(range(15, 20))
    assert all(i in idx for i in range(15, 20))


@pytest.mark.slow
def test_strip_last_matches_seq_len() -> None:
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    ids = tok("Hello", return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    L = past_sequence_length(past)
    stripped = strip_last_position_from_past(clone_past_key_values(past))
    assert past_sequence_length(stripped) == max(0, L - 1)


@pytest.mark.slow
def test_kv_cache_sparse_stats_and_memory() -> None:
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    ids = tok("Once upon a time there was a little sparse test", return_tensors="pt")[
        "input_ids"
    ]
    cfg = SparseRetentionConfig(recent_window=4, heavy_hitter_budget=4, refresh_interval=1)
    cache = KVCacheSparse(cfg, model=model)
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    cache.note_forward_token(ids[:, -1:])
    cache.append_from_forward_output(out.past_key_values)
    st = cache.stats()
    assert st["type"] == "KVCacheSparse"
    assert st["retained_sequence_length"] <= st["full_sequence_length"]
    assert st["memory_bytes_logical"] == st["payload_bytes"] + st["metadata_bytes"]
    assert st["refresh_events"] >= 1


def test_build_full_length_scores() -> None:
    pre = torch.tensor([0.1, 0.5, 0.2])
    full = build_full_length_scores_from_attention_prefix(pre, total_len=4)
    assert full.shape[0] == 4
    assert torch.isinf(full[-1])


def test_key_norm_matches_length() -> None:
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    ids = tok("Hi", return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    s = key_norm_token_scores(out.past_key_values)
    assert s.shape[0] == ids.shape[1]
