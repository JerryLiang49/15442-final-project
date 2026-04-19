"""Phase 14 benchmark gate — run before Phase 15 sweeps.

    pytest -m benchmark_gate

Validates: speculative modes match greedy AR, frozen metrics JSON shape, acceptance rates,
verifier KV length, quantization semantics, sparse cache reset when reusing a draft instance.

Keep prompts and ``max_new_tokens`` small so this stays CI-friendly with ``--slow``.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlsys_kv.benchmarks.experiment_schema import (
    DRAFT_CACHE_MODE_VALUES,
    SPECULATIVE_METRICS_JSON_KEYS,
    assert_speculative_metrics_jsonable_shape,
)
from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.hf_kv_trim import verifier_cache_seq_len_hf
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse
from mlsys_kv.cache.kv_quant_semantics import IMPLEMENTED_DRAFT_KV_QUANTIZATION_SEMANTICS
from mlsys_kv.decoding.autoregressive import decode_greedy_autoregressive
from mlsys_kv.decoding.speculative import SpeculativeDecoder

# --- Prompt suite (short / medium / longer) ---

PHASE14_PROMPT_SHORT = "Hi."
PHASE14_PROMPT_MEDIUM = "The number one is"
PHASE14_PROMPT_LONG = "Once upon a time there was a benchmark gate. " * 3

PHASE14_PROMPTS = (PHASE14_PROMPT_SHORT, PHASE14_PROMPT_MEDIUM, PHASE14_PROMPT_LONG)

PHASE14_SPARSE_CFG = SparseRetentionConfig(
    recent_window=8,
    heavy_hitter_budget=8,
    refresh_interval=2,
    scoring="key_norm",
)

PHASE14_MAX_NEW_TOKENS = 10


@pytest.fixture(scope="module")
def gpt2_small():
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return model, tok


def _mode_extra(draft_mode: DraftCacheMode) -> dict:
    if draft_mode is DraftCacheMode.FP16:
        return {}
    if draft_mode is DraftCacheMode.QUANT_ONLY:
        return {"kv_quant_bits": 8}
    if draft_mode is DraftCacheMode.SPARSE_ONLY:
        return {"sparse_config": PHASE14_SPARSE_CFG}
    if draft_mode is DraftCacheMode.SPARSE_QUANT:
        return {"sparse_config": PHASE14_SPARSE_CFG, "kv_quant_bits": 8}
    raise AssertionError(draft_mode)


@pytest.mark.benchmark_gate
@pytest.mark.slow
def test_phase14_autoregressive_baseline(gpt2_small) -> None:
    """Sweep mode ``autoregressive``: greedy decode runs and returns consistent length."""
    model, tok = gpt2_small
    for prompt in PHASE14_PROMPTS:
        out = decode_greedy_autoregressive(
            model,
            tok,
            prompt,
            max_new_tokens=PHASE14_MAX_NEW_TOKENS,
            warmup=False,
            trial_index=0,
        )
        expect = tok(prompt, return_tensors="pt")["input_ids"].shape[1] + PHASE14_MAX_NEW_TOKENS
        assert out.full_token_ids.shape[1] == expect


@pytest.mark.benchmark_gate
@pytest.mark.slow
@pytest.mark.parametrize("prompt", PHASE14_PROMPTS)
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize(
    "draft_mode",
    [
        DraftCacheMode.FP16,
        DraftCacheMode.QUANT_ONLY,
        DraftCacheMode.SPARSE_ONLY,
        DraftCacheMode.SPARSE_QUANT,
    ],
)
def test_phase14_speculative_matches_ar_metrics_verifier_and_schema(
    gpt2_small,
    prompt: str,
    k: int,
    draft_mode: DraftCacheMode,
) -> None:
    model, tok = gpt2_small
    n = PHASE14_MAX_NEW_TOKENS
    extra = _mode_extra(draft_mode)

    dec = SpeculativeDecoder(
        model,
        tok,
        k=k,
        draft_mode=draft_mode,
        verify_match=True,
        debug_speculative=True,
        **extra,
    )
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"]
    expect_len = int(prompt_ids.shape[1]) + n
    spec = dec.decode(prompt, max_new_tokens=n)
    # ``verify_match=True`` already compared to greedy AR inside the decoder.
    assert spec.full_token_ids.shape[1] == expect_len, (
        f"mode={draft_mode.value} k={k} seq_len mismatch for prompt chars={len(prompt)}"
    )

    m = spec.metrics
    assert m.draft_cache_mode in DRAFT_CACHE_MODE_VALUES
    assert 0.0 <= m.acceptance_rate <= 1.0
    assert m.total_rounds >= 1
    assert m.total_new_tokens == n
    if m.total_draft_proposals > 0:
        assert 0 <= m.total_accepted_tokens <= m.total_draft_proposals

    j = m.to_jsonable()
    assert_speculative_metrics_jsonable_shape(j)

    ver_past = spec.verifier_kv.get_attention_kv()
    assert ver_past is not None
    assert verifier_cache_seq_len_hf(ver_past) == int(spec.full_token_ids.shape[1])

    if draft_mode in (DraftCacheMode.QUANT_ONLY, DraftCacheMode.SPARSE_QUANT):
        assert j["draft_kv_quantization_semantics"] == IMPLEMENTED_DRAFT_KV_QUANTIZATION_SEMANTICS
        assert j["draft_runtime_accelerated_quant_attention"] is False
        assert m.draft_cache_end_stats is not None
        assert m.draft_cache_end_stats.get("kv_quantization_semantics") == "memory_only"
    else:
        assert j["draft_kv_quantization_semantics"] is None
        assert j["draft_runtime_accelerated_quant_attention"] is False


@pytest.mark.benchmark_gate
@pytest.mark.slow
def test_phase14_two_sequential_decodes_same_decoder_instance(gpt2_small) -> None:
    """Reuse :class:`SpeculativeDecoder` for a second prompt — no stale state (fresh caches per decode)."""
    model, tok = gpt2_small
    dec = SpeculativeDecoder(
        model,
        tok,
        k=3,
        draft_mode=DraftCacheMode.SPARSE_ONLY,
        verify_match=True,
        sparse_config=PHASE14_SPARSE_CFG,
    )
    n = 8
    for prompt in (PHASE14_PROMPT_SHORT, PHASE14_PROMPT_MEDIUM):
        p_ids = tok(prompt, return_tensors="pt")["input_ids"]
        spec = dec.decode(prompt, max_new_tokens=n)
        assert spec.full_token_ids.shape[1] == p_ids.shape[1] + n


@pytest.mark.benchmark_gate
@pytest.mark.slow
def test_phase14_sparse_draft_cache_reset_between_prompts_on_reused_cache(gpt2_small) -> None:
    """Reusing a single :class:`KVCacheSparse` requires :meth:`~KVCacheSparse.reset` between fills."""
    model, tok = gpt2_small
    cfg = PHASE14_SPARSE_CFG
    cache = KVCacheSparse(cfg, model=model)
    ids_a = tok(PHASE14_PROMPT_SHORT, return_tensors="pt")["input_ids"]
    ids_b = tok(PHASE14_PROMPT_MEDIUM, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        oa = model(ids_a, use_cache=True)
        ob = model(ids_b, use_cache=True)
    cache.append_from_forward_output(oa.past_key_values)
    assert cache.logical_seq_len > 0
    cache.reset()
    assert cache.logical_seq_len == 0
    cache.append_from_forward_output(ob.past_key_values)
    assert cache.logical_seq_len == int(ids_b.shape[1])


@pytest.mark.benchmark_gate
def test_phase14_frozen_speculative_metrics_key_set_covers_schema() -> None:
    """CI-only: :data:`SPECULATIVE_METRICS_JSON_KEYS` stays aligned with production ``to_jsonable``."""
    # If SpeculativeMetrics gains a field, extend experiment_schema.SPECULATIVE_METRICS_JSON_KEYS.
    from mlsys_kv.decoding.speculative import SpeculativeMetrics

    assert set(SPECULATIVE_METRICS_JSON_KEYS) == set(SpeculativeMetrics.__dataclass_fields__.keys())
