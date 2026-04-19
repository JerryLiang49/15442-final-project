"""Unit checks for manual greedy decoding vs ``model.generate``."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.memory import reset_peak_memory_stats
from decoding.autoregressive import (
    decode_greedy_autoregressive,
    reference_greedy_generate_ids,
)


@pytest.mark.slow
def test_greedy_decode_matches_generate_gpt2() -> None:
    """Greedy manual decoding should match ``model.generate`` for GPT-2."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = "The meaning of life is"
    max_new = 9
    device = next(model.parameters()).device
    reset_peak_memory_stats(device)

    ref_ids = reference_greedy_generate_ids(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new,
    )
    manual = decode_greedy_autoregressive(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new,
        warmup=False,
        trial_index=0,
    )

    assert manual.full_token_ids.dtype == torch.long
    assert ref_ids.shape == manual.full_token_ids.shape
    assert torch.equal(ref_ids, manual.full_token_ids)


@pytest.mark.slow
def test_greedy_decode_zero_new_tokens_gpt2() -> None:
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = "Hi"
    manual = decode_greedy_autoregressive(
        model,
        tokenizer,
        prompt,
        max_new_tokens=0,
        warmup=False,
        trial_index=0,
    )
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    assert manual.new_token_ids.numel() == 0
    assert torch.equal(manual.full_token_ids, inputs)
    assert manual.metrics.new_tokens_generated == 0
    assert manual.metrics.decode_step_times_s == []
