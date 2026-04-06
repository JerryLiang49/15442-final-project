"""Self-speculative decoding matches greedy autoregressive (uncompressed KV)."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.draft_factory import create_draft_cache
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse
from mlsys_kv.cache.kv_cache_sparse_quantized import KVCacheSparseQuantized
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


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 4])
def test_speculative_matches_greedy_gpt2(gpt2_small, k: int) -> None:
    model, tok = gpt2_small
    prompt = "The number one is"
    n = 15
    dec = SpeculativeDecoder(
        model,
        tok,
        k=k,
        draft_mode=DraftCacheMode.FP16,
        verbose=False,
        verify_match=True,
    )
    spec = dec.decode(prompt, max_new_tokens=n)
    ar = decode_greedy_autoregressive(
        model, tok, prompt, max_new_tokens=n, warmup=False, trial_index=0
    )
    assert torch.equal(spec.full_token_ids, ar.full_token_ids)


@pytest.mark.slow
def test_speculative_zero_new_tokens(gpt2_small) -> None:
    model, tok = gpt2_small
    dec = SpeculativeDecoder(model, tok, k=2, draft_mode=DraftCacheMode.FP16, verify_match=True)
    r = dec.decode("Hi", max_new_tokens=0)
    assert r.new_token_ids.numel() == 0
    assert r.metrics.total_rounds == 0
    assert r.metrics.acceptance_rate == 0.0
    assert r.metrics.draft_cache_mode == "fp16"


@pytest.mark.slow
def test_speculative_high_acceptance_self_model(gpt2_small) -> None:
    model, tok = gpt2_small
    dec = SpeculativeDecoder(model, tok, k=4, draft_mode=DraftCacheMode.FP16, verify_match=True)
    r = dec.decode("Once upon a time", max_new_tokens=24)
    assert r.metrics.acceptance_rate == 1.0
    assert r.metrics.rejection_events == 0


@pytest.mark.slow
def test_fp16_draft_metrics_reproducible_gpt2(gpt2_small) -> None:
    """Refactor regression: acceptance structure unchanged for FP16 draft (excluding wall time)."""
    model, tok = gpt2_small
    prompt = "When the sun"
    n = 20
    k = 3

    def _run() -> dict:
        dec = SpeculativeDecoder(
            model,
            tok,
            k=k,
            draft_mode=DraftCacheMode.FP16,
            verify_match=True,
        )
        m = dec.decode(prompt, max_new_tokens=n).metrics.to_jsonable()
        m.pop("total_runtime_s", None)
        m.pop("draft_dequant_time_s_total", None)
        m.pop("draft_refresh_time_s_total", None)
        m.pop("draft_mean_sparsity_ratio", None)
        m.pop("draft_quantization_kv_bits", None)
        m.pop("draft_cache_end_stats", None)
        return m

    a = _run()
    b = _run()
    assert a == b


def test_create_draft_cache_sparse_only() -> None:
    c = create_draft_cache(DraftCacheMode.SPARSE_ONLY, model=None)
    assert isinstance(c, KVCacheSparse)


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2])
def test_speculative_sparse_matches_greedy_gpt2(gpt2_small, k: int) -> None:
    model, tok = gpt2_small
    prompt = "The number one is"
    n = 15
    cfg = SparseRetentionConfig(
        recent_window=8,
        heavy_hitter_budget=8,
        refresh_interval=2,
        scoring="key_norm",
    )
    dec = SpeculativeDecoder(
        model,
        tok,
        k=k,
        draft_mode=DraftCacheMode.SPARSE_ONLY,
        verbose=False,
        verify_match=True,
        sparse_config=cfg,
    )
    spec = dec.decode(prompt, max_new_tokens=n)
    ar = decode_greedy_autoregressive(
        model, tok, prompt, max_new_tokens=n, warmup=False, trial_index=0
    )
    assert torch.equal(spec.full_token_ids, ar.full_token_ids)
    assert spec.metrics.draft_mean_sparsity_ratio >= 0.0


def test_create_draft_cache_sparse_quant() -> None:
    c = create_draft_cache(DraftCacheMode.SPARSE_QUANT, model=None)
    assert isinstance(c, KVCacheSparseQuantized)


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2])
def test_speculative_sparse_quant_matches_greedy_gpt2(gpt2_small, k: int) -> None:
    model, tok = gpt2_small
    prompt = "The number one is"
    n = 15
    cfg = SparseRetentionConfig(
        recent_window=8,
        heavy_hitter_budget=8,
        refresh_interval=2,
        scoring="key_norm",
    )
    dec = SpeculativeDecoder(
        model,
        tok,
        k=k,
        draft_mode=DraftCacheMode.SPARSE_QUANT,
        verbose=False,
        verify_match=True,
        sparse_config=cfg,
    )
    spec = dec.decode(prompt, max_new_tokens=n)
    ar = decode_greedy_autoregressive(
        model, tok, prompt, max_new_tokens=n, warmup=False, trial_index=0
    )
    assert torch.equal(spec.full_token_ids, ar.full_token_ids)
    assert spec.metrics.draft_cache_end_stats is not None
    assert spec.metrics.draft_cache_end_stats["type"] == "KVCacheSparseQuantized"
    assert spec.metrics.draft_cache_end_stats["composition_order"] == "sparsify_then_quantize_retained"
    assert spec.metrics.draft_quantization_kv_bits == 8
    assert spec.metrics.draft_dequant_time_s_total > 0.0
    assert spec.metrics.draft_refresh_time_s_total >= 0.0
