"""Unit tests for :mod:`cache.quant_spec_kv` (QuantSpec-style KV quantization)."""

from __future__ import annotations

import pytest
import torch

from cache.quant_spec_kv import (
    pack_int4_pair,
    quantize_fp16_kv_to_upper_lower,
    quantize_key_channelwise_upper_lower,
    quantize_value_tokenwise_upper_lower,
    reconstruct_key_draft,
    reconstruct_key_target,
    reconstruct_value_draft,
    reconstruct_value_target,
    unpack_int4_pair,
)


def test_pack_unpack_int4_roundtrip() -> None:
    lo = torch.randint(0, 16, (2, 3, 4), dtype=torch.int8)
    hi = torch.randint(0, 16, (2, 3, 4), dtype=torch.int8)
    packed = pack_int4_pair(lo, hi)
    assert packed.dtype == torch.uint8
    lo2, hi2 = unpack_int4_pair(packed)
    assert torch.equal(lo, lo2)
    assert torch.equal(hi, hi2)


def test_target_reconstruction_error_small() -> None:
    torch.manual_seed(0)
    b, h, s, d = 1, 2, 8, 8
    gs = 8
    k = torch.randn(b, h, s, d, dtype=torch.float16)
    v = torch.randn(b, h, s, d, dtype=torch.float16)
    kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp = quantize_fp16_kv_to_upper_lower(
        k, v, group_size=gs
    )
    rk = reconstruct_key_target(kuq, kus, kuzp, klq, kls, klzp)
    rv = reconstruct_value_target(vuq, vus, vuzp, vlq, vls, vlzp, group_size=gs)
    assert (k.float() - rk.float()).abs().max().item() < 0.35
    assert (v.float() - rv.float()).abs().max().item() < 0.35


def test_draft_uses_upper_only_target_adds_lower() -> None:
    torch.manual_seed(1)
    gs = 4
    k = torch.randn(1, 1, 8, 8, dtype=torch.float16)
    v = torch.randn(1, 1, 8, 8, dtype=torch.float16)
    kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp = quantize_fp16_kv_to_upper_lower(
        k, v, group_size=gs
    )
    kd = reconstruct_key_draft(kuq, kus, kuzp)
    kt = reconstruct_key_target(kuq, kus, kuzp, klq, kls, klzp)
    vd = reconstruct_value_draft(vuq, vus, vuzp, group_size=gs)
    vt = reconstruct_value_target(vuq, vus, vuzp, vlq, vls, vlzp, group_size=gs)
    err_d_k = (k.float() - kd.float()).abs().mean()
    err_t_k = (k.float() - kt.float()).abs().mean()
    err_d_v = (v.float() - vd.float()).abs().mean()
    err_t_v = (v.float() - vt.float()).abs().mean()
    assert err_t_k < err_d_k
    assert err_t_v < err_d_v
    assert (kt.float() - kd.float()).abs().max().item() > 1e-6
    assert (vt.float() - vd.float()).abs().max().item() > 1e-6


def test_key_channel_axis_independence() -> None:
    """Scales for channel group g reflect only that group's min/max (not other groups)."""
    gs = 4
    d = 8
    k = torch.zeros(1, 1, 2, d, dtype=torch.float32)
    # Group 0 (dims 0..3): small dynamic range; group 1 (dims 4..7): large range.
    k[..., 0] = 0.0
    k[..., 1] = 1.0
    k[..., 2] = 2.0
    k[..., 3] = 3.0
    k[..., 4] = 100.0
    k[..., 5] = 200.0
    k[..., 6] = 300.0
    k[..., 7] = 400.0
    kuq, kus, kuzp, _, _, _ = quantize_key_channelwise_upper_lower(k, group_size=gs)
    assert kus.shape == (1, 1, 2, 2)
    s0 = kus[0, 0, 0, 0].item()
    s1 = kus[0, 0, 0, 1].item()
    assert s1 > s0 * 10


def test_value_token_axis_independence() -> None:
    """Scales for token group g reflect only that group's min/max along S (per channel)."""
    gs = 4
    s = 8
    d = 4
    v = torch.zeros(1, 1, s, d, dtype=torch.float32)
    # Per token group, vary values across the gs timesteps so each channel has nonzero span.
    for t in range(4):
        v[:, :, t, :] = torch.arange(d, dtype=torch.float32) + float(t)
    for t in range(4, 8):
        v[:, :, t, :] = (torch.arange(d, dtype=torch.float32) + float(t)) * 100.0
    vuq, vus, vuzp, _, _, _ = quantize_value_tokenwise_upper_lower(v, group_size=gs)
    assert vus.shape == (1, 1, 2, d)
    assert vus[0, 0, 1, 0].item() > vus[0, 0, 0, 0].item() * 10


def test_logical_view_consistency_same_stored_tensors() -> None:
    """Draft and target read identical codes/metadata; only reconstruction differs."""
    gs = 8
    k = torch.randn(1, 1, 8, 8, dtype=torch.float16)
    v = torch.randn(1, 1, 8, 8, dtype=torch.float16)
    kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp = quantize_fp16_kv_to_upper_lower(
        k, v, group_size=gs
    )
    kd = reconstruct_key_draft(kuq, kus, kuzp)
    kt = reconstruct_key_target(kuq, kus, kuzp, klq, kls, klzp)
    vd = reconstruct_value_draft(vuq, vus, vuzp, group_size=gs)
    vt = reconstruct_value_target(vuq, vus, vuzp, vlq, vls, vlzp, group_size=gs)
    assert kd.shape == kt.shape == k.shape
    assert vd.shape == vt.shape == v.shape


@pytest.mark.parametrize("gs", [1, 2, 4, 8])
def test_group_size_variants(gs: int) -> None:
    if 8 % gs != 0:
        pytest.skip("head_dim 8 must divide gs")
    k = torch.randn(1, 1, 8, 8, dtype=torch.float16)
    v = torch.randn(1, 1, 8, 8, dtype=torch.float16)
    kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp = quantize_fp16_kv_to_upper_lower(
        k, v, group_size=gs
    )
    rk = reconstruct_key_target(kuq, kus, kuzp, klq, kls, klzp)
    rv = reconstruct_value_target(vuq, vus, vuzp, vlq, vls, vlzp, group_size=gs)
    assert (k.float() - rk.float()).abs().max().item() < 0.4
    assert (v.float() - rv.float()).abs().max().item() < 0.4
