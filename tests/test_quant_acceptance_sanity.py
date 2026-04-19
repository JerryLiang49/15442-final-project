"""Short-prompt sanity check: real KV tensors from a tiny GPT-2 before perf work."""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2Model

from cache.quant_spec_kv import (
    quantize_fp16_kv_to_upper_lower,
    reconstruct_key_target,
    reconstruct_value_target,
)


def test_tiny_gpt2_past_kv_quantize_sanity() -> None:
    """One forward with cache; quantize layer-0 K/V; target reconstruction error bounded."""
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=64,
        vocab_size=100,
    )
    torch.manual_seed(0)
    model = GPT2Model(cfg)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past = out.past_key_values
    assert past is not None
    # Transformers 4.4x+: DynamicCache with per-layer DynamicLayer (.keys / .values).
    layer0 = past.layers[0]
    k0 = layer0.keys
    v0 = layer0.values
    assert k0.dtype in (torch.float16, torch.float32)
    k0 = k0.to(torch.float16)
    v0 = v0.to(torch.float16)
    b, nh, sl, hd = k0.shape
    gs = hd
    if sl % gs != 0:
        pad = gs - (sl % gs)
        k0 = torch.nn.functional.pad(k0, (0, 0, 0, pad))
        v0 = torch.nn.functional.pad(v0, (0, 0, 0, pad))
    kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp = quantize_fp16_kv_to_upper_lower(
        k0, v0, group_size=gs
    )
    rk = reconstruct_key_target(kuq, kus, kuzp, klq, kls, klzp)
    rv = reconstruct_value_target(vuq, vus, vuzp, vlq, vls, vlzp, group_size=gs)
    assert (k0.float() - rk.float()).abs().max().item() < 0.5
    assert (v0.float() - rv.float()).abs().max().item() < 0.5
