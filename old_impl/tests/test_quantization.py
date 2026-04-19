"""INT8 symmetric quantization helpers and quantized draft cache."""

from __future__ import annotations

import pytest
import torch

from mlsys_kv.cache.kv_cache_quantized import KVCacheQuantized
from mlsys_kv.cache.quantization import symmetric_dequantize_int8, symmetric_quantize_int8
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_symmetric_quantize_dequantize_zero_error_large_tensor() -> None:
    x = torch.linspace(-1.2, 1.2, steps=1024, dtype=torch.float32)
    q, s = symmetric_quantize_int8(x)
    xh = symmetric_dequantize_int8(q, s, out_dtype=torch.float32, out_device=x.device)
    max_err = (x - xh).abs().max().item()
    assert max_err <= (s.item() * 0.51)  # within ~half step


@pytest.mark.slow
def test_kv_cache_quantized_roundtrip_dynamic_cache_gpt2() -> None:
    from transformers.cache_utils import DynamicCache, DynamicLayer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tok = AutoTokenizer.from_pretrained("gpt2")
    inp = tok("hello", return_tensors="pt").input_ids
    out = model(inp, use_cache=True)
    past = out.past_key_values
    assert isinstance(past, DynamicCache)

    cq = KVCacheQuantized()
    cq.append_from_forward_output(past)
    rebuilt = cq.get_attention_kv()
    assert isinstance(rebuilt, DynamicCache)
    assert len(rebuilt.layers) == len(past.layers)

    for i, (old_layer, new_layer) in enumerate(zip(past.layers, rebuilt.layers, strict=True)):
        assert isinstance(old_layer, DynamicLayer) and isinstance(new_layer, DynamicLayer)
        if not old_layer.is_initialized:
            continue
        ok = old_layer.keys.float()
        nk = new_layer.keys.float()
        max_err = (ok - nk).abs().max().item()
        assert max_err < 0.2, f"layer {i} max err {max_err}"


@pytest.mark.slow
def test_kv_cache_quantized_memory_accounts_payload_and_scales() -> None:
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tok = AutoTokenizer.from_pretrained("gpt2")
    inp = tok("a b c", return_tensors="pt").input_ids
    out = model(inp, use_cache=True)
    cq = KVCacheQuantized()
    cq.append_from_forward_output(out.past_key_values)
    st = cq.stats()
    assert st["payload_bytes_int8"] > 0
    assert st["metadata_bytes"] >= 8 * int(st["num_layers"])  # two float32 scales per layer (gpt2)
    assert st["memory_bytes_logical"] == st["payload_bytes_int8"] + st["metadata_bytes"]


@pytest.mark.slow
def test_speculative_quant_draft_matches_greedy() -> None:
    from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
    from mlsys_kv.decoding.autoregressive import decode_greedy_autoregressive
    from mlsys_kv.decoding.speculative import SpeculativeDecoder

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    prompt = "Counting: one, two,"
    n = 18
    dec = SpeculativeDecoder(
        model,
        tok,
        k=3,
        draft_mode=DraftCacheMode.QUANT_ONLY,
        verify_match=True,
    )
    spec = dec.decode(prompt, max_new_tokens=n)
    ar = decode_greedy_autoregressive(
        model, tok, prompt, max_new_tokens=n, warmup=False, trial_index=0
    )
    assert torch.equal(spec.full_token_ids, ar.full_token_ids)
    assert spec.metrics.draft_dequant_time_s_total > 0.0
    assert spec.metrics.draft_cache_end_stats is not None
    assert spec.metrics.draft_cache_end_stats["type"] == "KVCacheQuantized"
    assert spec.metrics.draft_kv_quantization_semantics == "memory_only"
    assert spec.metrics.draft_runtime_accelerated_quant_attention is False
    st = spec.metrics.draft_cache_end_stats
    assert st["kv_quantization_semantics"] == "memory_only"
    assert st["runtime_accelerated_quant_attention"] is False
    assert st["attention_consumes_dequantized_kv"] is True
    assert st["claim_decode_speedup_from_kv_quant_alone"] is False
    assert st["ephemeral_attention_kv_rebuild_bytes_est"] >= st["payload_bytes_int8"]
