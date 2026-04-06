"""Phase 7: joint sparse + quantized draft cache."""

from __future__ import annotations

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlsys_kv.cache.hf_kv_clone import past_sequence_length
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.kv_cache_quantized import KVCacheQuantized
from mlsys_kv.cache.kv_cache_sparse_quantized import KVCacheSparseQuantized


@pytest.mark.slow
def test_sparse_quant_memory_below_quant_only_on_long_prefix() -> None:
    """With real sparsity, joint INT8 payload is smaller than quantizing the full sequence."""
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    text = "word " * 25 + "end"
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    assert past_sequence_length(past) >= 20

    cfg = SparseRetentionConfig(
        recent_window=4,
        heavy_hitter_budget=4,
        refresh_interval=1,
        scoring="key_norm",
    )
    joint = KVCacheSparseQuantized(cfg, model=None)
    joint.note_forward_token(ids[:, -1:])
    joint.append_from_forward_output(past)

    qonly = KVCacheQuantized()
    qonly.append_from_forward_output(past)

    st_j = joint.stats()
    st_q = qonly.stats()
    assert st_j["composition_order"] == "sparsify_then_quantize_retained"
    assert st_j["quantization_kv_bits"] == 8
    assert st_j["metadata_bytes_sparse"] > 0
    assert st_j["metadata_bytes_quant"] > 0
    assert st_j["memory_bytes_logical"] == st_j["payload_bytes_int8"] + st_j["metadata_bytes"]
    assert st_j["retained_sequence_length"] < st_j["full_sequence_length"]
    assert st_j["payload_bytes_int8"] < st_q["payload_bytes_int8"]
