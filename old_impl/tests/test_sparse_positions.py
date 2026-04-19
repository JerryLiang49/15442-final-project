"""Phase 11: sparse draft physical KV length R vs logical length L and position_ids."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.hf_kv_trim import verifier_cache_seq_len_hf
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse
from mlsys_kv.cache.sparse_position_audit import audit_sparse_draft_state
from mlsys_kv.decoding.speculative import propose_draft_tokens


@pytest.mark.slow
def test_sparse_explicit_position_ids_when_physical_shorter_than_logical() -> None:
    """HF default uses past.get_seq_length() == R; next query must use position L, not R."""
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    prompt = (
        "Once upon a time there was a little sparse positional test "
        "with enough tokens to force retention smaller than full length."
    )
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    cfg = SparseRetentionConfig(
        recent_window=4,
        heavy_hitter_budget=2,
        refresh_interval=1,
        scoring="key_norm",
    )
    cache = KVCacheSparse(cfg, model=model)
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    cache.note_forward_token(ids[:, -1:])
    cache.append_from_forward_output(out.past_key_values)

    L = cache.stats()["logical_seq_len_full"]
    past = cache.get_attention_kv()
    assert past is not None
    R = verifier_cache_seq_len_hf(past)
    assert L > 5
    assert R < L, "test requires evicted cache; bump prompt length or tighten budget"

    next_tok = torch.tensor([[42]], dtype=torch.long)
    with torch.inference_mode():
        logits_default = model(
            input_ids=next_tok,
            past_key_values=past,
            use_cache=True,
        ).logits
        pos = cache.position_ids_for_next_queries(
            1, batch_size=1, device=next_tok.device
        )
        assert pos is not None and int(pos[0, 0].item()) == L
        logits_explicit = model(
            input_ids=next_tok,
            past_key_values=past,
            use_cache=True,
            position_ids=pos,
        ).logits

    assert not torch.allclose(
        logits_default, logits_explicit, rtol=0.0, atol=1e-3
    ), "default HF positions should not match explicit L when R < L"


@pytest.mark.slow
def test_audit_sparse_draft_state_fields() -> None:
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    ids = tok("Hello sparse audit " * 6, return_tensors="pt")["input_ids"]
    cfg = SparseRetentionConfig(
        recent_window=3, heavy_hitter_budget=2, refresh_interval=1, scoring="key_norm"
    )
    cache = KVCacheSparse(cfg, model=model)
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    cache.append_from_forward_output(out.past_key_values)
    past = cache.get_attention_kv()
    rep = audit_sparse_draft_state(
        cache, past_for_model=past, query_length=1, device=ids.device
    )
    assert rep.logical_full_len == cache.stats()["logical_seq_len_full"]
    assert rep.physical_hf_cache_len == verifier_cache_seq_len_hf(past)
    assert rep.next_query_position_ids == [rep.logical_full_len]


@pytest.mark.slow
def test_propose_draft_tokens_passes_explicit_positions_for_sparse() -> None:
    """Regression: draft loop must not rely on HF get_seq_length() when R < L."""
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    prompt = (
        "The quick brown fox " * 4
        + "jumps over the lazy dog " * 4
        + "again and again for sparse draft steps."
    )
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    cfg = SparseRetentionConfig(
        recent_window=4, heavy_hitter_budget=2, refresh_interval=1, scoring="key_norm"
    )
    with torch.inference_mode():
        pre = model(ids, use_cache=True)
    start_logits = pre.logits[:, -1, :].clone()

    cache = KVCacheSparse(cfg, model=model)
    cache.append_from_forward_output(pre.past_key_values)

    k = 3
    proposals = propose_draft_tokens(
        model, draft_cache=cache, start_logits=start_logits, k=k
    )
    assert len(proposals) == k
    st = cache.stats()
    assert st["logical_seq_len_full"] == ids.shape[1] + k
    assert st["physical_retained_kv_len"] < st["logical_seq_len_full"]
