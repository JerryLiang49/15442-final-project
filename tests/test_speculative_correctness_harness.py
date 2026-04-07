"""Regression harness: speculative decode must match greedy AR (multiple prompts × draft modes)."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.decoding.autoregressive import decode_greedy_autoregressive
from mlsys_kv.decoding.speculative import SpeculativeDecoder


@pytest.fixture(scope="module")
def gpt2_small():
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return model, tok


SPARSE_CFG = SparseRetentionConfig(
    recent_window=8,
    heavy_hitter_budget=8,
    refresh_interval=2,
    scoring="key_norm",
)


@pytest.mark.slow
@pytest.mark.parametrize("prompt", ["The number one is", "Once upon a time", "a" * 20])
@pytest.mark.parametrize(
    "draft_mode,extra",
    [
        (DraftCacheMode.FP16, {}),
        (DraftCacheMode.QUANT_ONLY, {"kv_quant_bits": 8}),
        (DraftCacheMode.SPARSE_ONLY, {"sparse_config": SPARSE_CFG}),
        (DraftCacheMode.SPARSE_QUANT, {"sparse_config": SPARSE_CFG, "kv_quant_bits": 8}),
    ],
)
@pytest.mark.parametrize("k", [2, 4])
def test_speculative_matches_ar_multi(
    gpt2_small,
    prompt: str,
    draft_mode: DraftCacheMode,
    extra: dict,
    k: int,
) -> None:
    model, tok = gpt2_small
    n = 12
    dec = SpeculativeDecoder(
        model,
        tok,
        k=k,
        draft_mode=draft_mode,
        verify_match=True,
        debug_speculative=True,
        **extra,
    )
    spec = dec.decode(prompt, max_new_tokens=n)
    ar = decode_greedy_autoregressive(
        model, tok, prompt, max_new_tokens=n, warmup=False, trial_index=0
    )
    assert torch.equal(spec.full_token_ids, ar.full_token_ids)
