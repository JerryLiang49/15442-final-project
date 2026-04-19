"""Tests for dense self-speculative decoding (greedy, lossless vs AR)."""

from __future__ import annotations

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoding.speculative_dense import (
    SpeculativeDecoderDense,
    first_mismatch_index_greedy,
    greedy_correction_token,
    verify_block_and_commit,
)
from decoding.autoregressive import decode_greedy_autoregressive, model_device


def _tiny_logits(vocab: int, winner: int) -> torch.Tensor:
    """One-hot style logits so argmax is ``winner``."""
    t = torch.full((1, vocab), -1.0e9)
    t[0, winner] = 0.0
    return t


def test_full_acceptance_all_gamma_match() -> None:
    """All draft tokens match verifier predictions (j == gamma)."""
    gamma = 4
    vocab = 20
    proposals = torch.tensor([[3, 5, 7, 11]], dtype=torch.long)
    start = _tiny_logits(vocab, 3)
    block = torch.stack(
        [_tiny_logits(vocab, 5), _tiny_logits(vocab, 7), _tiny_logits(vocab, 11), _tiny_logits(vocab, 0)],
        dim=1,
    )
    j = first_mismatch_index_greedy(start, block, proposals)
    assert j == gamma


def test_mismatch_at_position_zero() -> None:
    """First draft token disagrees with ``start_logits``."""
    gamma = 3
    vocab = 50
    proposals = torch.tensor([[9, 5, 5]], dtype=torch.long)
    start = _tiny_logits(vocab, 3)  # verifier wants 3, draft proposes 9 at j=0
    block = torch.randn(1, gamma, vocab)
    j = first_mismatch_index_greedy(start, block, proposals)
    assert j == 0
    corr = greedy_correction_token(start, block, 0)
    assert int(corr[0, 0].item()) == 3


def test_mismatch_in_the_middle() -> None:
    """First mismatch at j=2."""
    gamma = 4
    vocab = 30
    proposals = torch.tensor([[1, 2, 99, 4]], dtype=torch.long)
    start = _tiny_logits(vocab, 1)
    b0 = _tiny_logits(vocab, 2)
    b1 = _tiny_logits(vocab, 3)  # verifier predicts 3 at position after token 2, draft has 99
    b2 = _tiny_logits(vocab, 0)
    b3 = _tiny_logits(vocab, 0)
    block = torch.stack([b0, b1, b2, b3], dim=1)
    j = first_mismatch_index_greedy(start, block, proposals)
    assert j == 2
    corr = greedy_correction_token(start, block, j)
    assert int(corr[0, 0].item()) == 3


def test_mismatch_at_last_draft_position() -> None:
    """All but last draft token match; last fails."""
    gamma = 3
    vocab = 40
    proposals = torch.tensor([[2, 3, 7]], dtype=torch.long)
    start = _tiny_logits(vocab, 2)
    b0 = _tiny_logits(vocab, 3)
    b1 = _tiny_logits(vocab, 8)  # after p_2=7 verifier wants 8
    block = torch.stack([b0, b1, _tiny_logits(vocab, 0)], dim=1)
    j = first_mismatch_index_greedy(start, block, proposals)
    assert j == 2
    corr = greedy_correction_token(start, block, j)
    assert int(corr[0, 0].item()) == 8


@pytest.mark.slow
def test_serving_mode_bit_exact_vs_reference_orchestration_gpt2() -> None:
    """Serving path (single verifier clone + draft wrapper reuse) matches legacy double-clone tokens."""
    model_name = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = "The capital of France is"
    max_new = 24
    gamma = 4
    legacy = SpeculativeDecoderDense(
        model,
        tok,
        gamma=gamma,
        verify_match_autoregressive=False,
        serving_mode=False,
        legacy_double_clone_verifier=True,
    )
    serving = SpeculativeDecoderDense(
        model,
        tok,
        gamma=gamma,
        verify_match_autoregressive=False,
        serving_mode=True,
        legacy_double_clone_verifier=False,
    )
    out_l = legacy.decode(prompt, max_new_tokens=max_new)
    out_s = serving.decode(prompt, max_new_tokens=max_new)
    assert torch.equal(out_l.full_token_ids, out_s.full_token_ids)


@pytest.mark.slow
def test_regression_matches_greedy_autoregressive_gpt2() -> None:
    """End-to-end: dense speculative matches manual greedy decode (GPT-2)."""
    model_name = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = model_device(model)
    _ = device

    prompt = "The capital of France is"
    max_new = 16
    for gamma in (1, 2, 4):
        dec = SpeculativeDecoderDense(model, tok, gamma=gamma, verify_match_autoregressive=True)
        out = dec.decode(prompt, max_new_tokens=max_new)
        ref = decode_greedy_autoregressive(
            model, tok, prompt, max_new_tokens=max_new, warmup=False, trial_index=0
        )
        assert torch.equal(out.full_token_ids, ref.full_token_ids), f"gamma={gamma}"


@pytest.mark.slow
def test_verify_block_commit_matches_ar_single_round_equivalence() -> None:
    """One speculative round with gamma=1 reduces to greedy next token (GPT-2)."""
    model_name = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = model_device(model)

    prompt = "Hello"
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.inference_mode():
        pre = model(input_ids=input_ids, use_cache=True, past_key_values=None)
    past = pre.past_key_values
    start_logits = pre.logits[:, -1, :].clone()
    expected_next = int(start_logits.argmax(dim=-1)[0].item())

    from decoding.speculative_dense import draft_greedy_proposals, draft_cache_from_verifier_snapshot
    from cache.hf_kv_clone import clone_past_key_values

    draft = draft_cache_from_verifier_snapshot(past)
    props = draft_greedy_proposals(model, draft_cache=draft, start_logits=start_logits, gamma=1)
    vp = clone_past_key_values(past)
    committed, new_past, new_logits, n_acc, rej = verify_block_and_commit(
        model,
        verifier_past_at_round_start=vp,
        start_logits=start_logits,
        proposals=props,
        debug=False,
    )
    assert len(committed) == 1
    assert int(committed[0][0, 0].item()) == expected_next
    assert n_acc == 1
    assert not rej
