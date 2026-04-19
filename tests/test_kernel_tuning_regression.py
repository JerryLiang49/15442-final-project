"""Phase O — fused kernel parity across tuning presets (reference vs Triton × profiles)."""

from __future__ import annotations

import pytest
import torch

from kv_kernels.fused_verifier_block_attention import fused_verifier_block_attention
from kv_kernels.triton_runtime import triton_available
from kv_kernels.tuning import get_preset_config, kernel_tuning_scope, list_tuning_profiles


def _case(device: torch.device) -> tuple:
    torch.manual_seed(1)
    h, d, s_hist, s_rec, gamma = 2, 64, 12, 2, 3
    gs_k, gs_v = 8, 8
    n_gk = d // gs_k
    n_gv = (s_hist + gs_v - 1) // gs_v
    q = torch.randn(1, h, gamma, d, device=device, dtype=torch.float32)
    k_uq = torch.randint(0, 16, (h, s_hist, d), device=device, dtype=torch.int8)
    k_lq = torch.randint(0, 16, (h, s_hist, d), device=device, dtype=torch.int8)
    v_uq = torch.randint(0, 16, (h, s_hist, d), device=device, dtype=torch.int8)
    v_lq = torch.randint(0, 16, (h, s_hist, d), device=device, dtype=torch.int8)
    k_su = torch.randn(h, s_hist, n_gk, device=device) * 0.05
    k_zu = torch.randn(h, s_hist, n_gk, device=device) * 0.05
    k_sl = torch.randn(h, s_hist, n_gk, device=device) * 0.05
    k_zl = torch.randn(h, s_hist, n_gk, device=device) * 0.05
    v_su = torch.randn(h, n_gv, d, device=device) * 0.05
    v_zu = torch.randn(h, n_gv, d, device=device) * 0.05
    v_sl = torch.randn(h, n_gv, d, device=device) * 0.05
    v_zl = torch.randn(h, n_gv, d, device=device) * 0.05
    k_rec = torch.randn(h, s_rec, d, device=device)
    v_rec = torch.randn(h, s_rec, d, device=device)
    k_blk = torch.randn(h, gamma, d, device=device)
    v_blk = torch.randn(h, gamma, d, device=device)
    return (
        q,
        k_uq,
        k_lq,
        k_su,
        k_zu,
        k_sl,
        k_zl,
        v_uq,
        v_lq,
        v_su,
        v_zu,
        v_sl,
        v_zl,
        k_rec,
        v_rec,
        k_blk,
        v_blk,
        gs_k,
        gs_v,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
@pytest.mark.skipif(not triton_available(), reason="triton")
@pytest.mark.parametrize(
    "profile",
    ["default", "verifier_wide", "draft_aggressive"],
)
def test_fused_verifier_triton_matches_reference_across_tuning_profiles(profile: str) -> None:
    dev = torch.device("cuda")
    tup = _case(dev)
    *tensors, gs_k, gs_v = tup
    ref = fused_verifier_block_attention(*tensors, group_size_k=gs_k, group_size_v=gs_v, backend="ref")

    cfg = get_preset_config(profile)
    with kernel_tuning_scope(cfg):
        trit = fused_verifier_block_attention(*tensors, group_size_k=gs_k, group_size_v=gs_v, backend="triton")

    err = (ref.float() - trit.float()).abs().max().item()
    assert err < 0.12, f"profile={profile} max_abs={err}"


def test_list_profiles_contains_defaults() -> None:
    names = list_tuning_profiles()
    for p in ("default", "a10g_balanced", "a100_high_throughput"):
        assert p in names
