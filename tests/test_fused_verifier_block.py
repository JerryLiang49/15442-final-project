"""Fused verifier block attention: reference vs gold matmul, ref vs Triton, logits parity.

**Causal rule** (see module docstring): prefix ``P = S_hist + S_rec``; key index ``k`` is allowed for
query row ``t`` iff ``k <= P + t`` (block-causal within the γ draft slots).

Full speculative decoding sequence parity requires an integration test with a model and
``AttentionKernelDispatch.TRITON_FUSED_VERIFIER``; this file focuses on kernel numerics.
"""

from __future__ import annotations

import pytest
import torch

from kv_kernels.fused_verifier_block_attention import (
    fused_verifier_block_attention,
    fused_verifier_block_attention_reference,
    fused_verifier_block_attention_reference_logits,
)
from kv_kernels.parity_harness import DEFAULT_LAYERWISE_TOLERANCES, assert_tensor_parity
from kv_kernels.triton_runtime import triton_available


def _random_case(
    *,
    h: int,
    d: int,
    s_hist: int,
    s_rec: int,
    gamma: int,
    gs_k: int,
    gs_v: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    n_gk = d // gs_k
    n_gv = (s_hist + gs_v - 1) // gs_v
    q = torch.randn(1, h, gamma, d, device=device, dtype=torch.float32)
    k_uq = torch.randint(0, 16, (h, s_hist, d), device=device, dtype=torch.int8)
    k_lq = torch.randint(0, 16, (h, s_hist, d), device=device, dtype=torch.int8)
    v_uq = torch.randint(0, 16, (h, s_hist, d), device=device, dtype=torch.int8)
    v_lq = torch.randint(0, 16, (h, s_hist, d), device=device, dtype=torch.int8)
    k_su = torch.randn(h, s_hist, n_gk, device=device, dtype=torch.float32) * 0.05
    k_zu = torch.randn(h, s_hist, n_gk, device=device, dtype=torch.float32) * 0.05
    k_sl = torch.randn(h, s_hist, n_gk, device=device, dtype=torch.float32) * 0.05
    k_zl = torch.randn(h, s_hist, n_gk, device=device, dtype=torch.float32) * 0.05
    v_su = torch.randn(h, n_gv, d, device=device, dtype=torch.float32) * 0.05
    v_zu = torch.randn(h, n_gv, d, device=device, dtype=torch.float32) * 0.05
    v_sl = torch.randn(h, n_gv, d, device=device, dtype=torch.float32) * 0.05
    v_zl = torch.randn(h, n_gv, d, device=device, dtype=torch.float32) * 0.05
    k_rec = torch.randn(h, s_rec, d, device=device, dtype=torch.float32)
    v_rec = torch.randn(h, s_rec, d, device=device, dtype=torch.float32)
    k_blk = torch.randn(h, gamma, d, device=device, dtype=torch.float32)
    v_blk = torch.randn(h, gamma, d, device=device, dtype=torch.float32)
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
    )


def _dequant_k_row(k_uq, k_lq, k_su, k_zu, k_sl, k_zl, gs_k: int) -> torch.Tensor:
    from kv_kernels.fused_verifier_block_attention import _dequant_k_target_row

    return _dequant_k_target_row(k_uq, k_lq, k_su, k_zu, k_sl, k_zl, group_size_k=gs_k)


def _dequant_v_row(v_uq, v_lq, s: int, v_su, v_zu, v_sl, v_zl, gs_v: int) -> torch.Tensor:
    from kv_kernels.fused_verifier_block_attention import _dequant_v_target_row

    return _dequant_v_target_row(v_uq, v_lq, s, v_su, v_zu, v_sl, v_zl, group_size_v=gs_v)


def _gold_matmul_attention(
    q: torch.Tensor,
    k_uq_hist: torch.Tensor,
    k_lq_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    k_scale_l: torch.Tensor,
    k_zp_l: torch.Tensor,
    v_uq_hist: torch.Tensor,
    v_lq_hist: torch.Tensor,
    v_scale_u: torch.Tensor,
    v_zp_u: torch.Tensor,
    v_scale_l: torch.Tensor,
    v_zp_l: torch.Tensor,
    k_recent: torch.Tensor,
    v_recent: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    *,
    group_size_k: int,
    group_size_v: int,
) -> torch.Tensor:
    """Explicit matmul + block-causal mask (float32). ``q`` ``[1,H,γ,D]``."""
    b, h, gamma, d = q.shape
    assert b == 1
    s_hist = k_uq_hist.shape[1]
    s_rec = k_recent.shape[1]
    p = s_hist + s_rec
    l_total = p + gamma
    scale_attn = float(d**-0.5)
    out = torch.zeros(1, h, gamma, d, device=q.device, dtype=torch.float32)
    qf = q.to(torch.float32)

    for head in range(h):
        k_mat = torch.zeros(l_total, d, device=q.device, dtype=torch.float32)
        v_mat = torch.zeros(l_total, d, device=q.device, dtype=torch.float32)
        for k in range(l_total):
            if k < s_hist:
                k_mat[k] = _dequant_k_row(
                    k_uq_hist[head, k],
                    k_lq_hist[head, k],
                    k_scale_u[head, k],
                    k_zp_u[head, k],
                    k_scale_l[head, k],
                    k_zp_l[head, k],
                    group_size_k,
                )
                v_mat[k] = _dequant_v_row(
                    v_uq_hist[head, k],
                    v_lq_hist[head, k],
                    k,
                    v_scale_u[head],
                    v_zp_u[head],
                    v_scale_l[head],
                    v_zp_l[head],
                    group_size_v,
                )
            elif k < s_hist + s_rec:
                j = k - s_hist
                k_mat[k] = k_recent[head, j].to(torch.float32)
                v_mat[k] = v_recent[head, j].to(torch.float32)
            else:
                j = k - p
                k_mat[k] = k_block[head, j].to(torch.float32)
                v_mat[k] = v_block[head, j].to(torch.float32)

        for t in range(gamma):
            logits = torch.full((l_total,), float("-inf"), device=q.device, dtype=torch.float32)
            for k in range(l_total):
                if k <= p + t:
                    logits[k] = (qf[0, head, t] * k_mat[k]).sum() * scale_attn
            p_attn = torch.softmax(logits, dim=0)
            out[0, head, t] = (p_attn.unsqueeze(-1) * v_mat).sum(dim=0)

    return out


def test_reference_matches_gold_matmul_cpu() -> None:
    dev = torch.device("cpu")
    (
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
    ) = _random_case(h=2, d=16, s_hist=6, s_rec=3, gamma=4, gs_k=4, gs_v=2, device=dev, seed=3)

    ref = fused_verifier_block_attention_reference(
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
        group_size_k=4,
        group_size_v=2,
    )
    gold = _gold_matmul_attention(
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
        group_size_k=4,
        group_size_v=2,
    )
    torch.testing.assert_close(ref, gold, rtol=1e-5, atol=1e-5)


def test_logits_reference_consistent_cpu() -> None:
    dev = torch.device("cpu")
    (
        q,
        k_uq,
        k_lq,
        k_su,
        k_zu,
        k_sl,
        k_zl,
        _,
        _,
        _,
        _,
        _,
        _,
        k_rec,
        _,
        k_blk,
        _,
    ) = _random_case(h=1, d=8, s_hist=4, s_rec=2, gamma=3, gs_k=4, gs_v=2, device=dev, seed=9)
    logits = fused_verifier_block_attention_reference_logits(
        q,
        k_uq,
        k_lq,
        k_su,
        k_zu,
        k_sl,
        k_zl,
        k_rec,
        k_blk,
        group_size_k=4,
    )
    s_hist = k_uq.shape[1]
    s_rec = k_rec.shape[1]
    p = s_hist + s_rec
    gamma = q.shape[2]
    l_total = p + gamma
    assert logits.shape == (1, 1, gamma, l_total)
    for t in range(gamma):
        for k in range(l_total):
            if k > p + t:
                assert torch.isneginf(logits[0, 0, t, k])


@pytest.mark.parity_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
def test_triton_matches_reference_cuda() -> None:
    dev = torch.device("cuda:0")
    (
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
    ) = _random_case(h=4, d=64, s_hist=12, s_rec=5, gamma=5, gs_k=8, gs_v=4, device=dev, seed=11)

    ref = fused_verifier_block_attention_reference(
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
        group_size_k=8,
        group_size_v=4,
    )
    tri = fused_verifier_block_attention(
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
        group_size_k=8,
        group_size_v=4,
        backend="triton",
    )
    assert_tensor_parity(
        ref,
        tri,
        tolerances=DEFAULT_LAYERWISE_TOLERANCES,
        context="fused_verifier_triton_cuda",
    )

    gamm = int(q.shape[2])
    sh = int(k_uq.shape[1])
    sr = int(k_rec.shape[1])
    bias = torch.randn(gamm, sh + sr + gamm, device=dev, dtype=torch.float32) * 0.02
    ref_b = fused_verifier_block_attention_reference(
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
        group_size_k=8,
        group_size_v=4,
        attn_logit_bias=bias,
    )
    tri_b = fused_verifier_block_attention(
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
        group_size_k=8,
        group_size_v=4,
        backend="triton",
        attn_logit_bias=bias,
    )
    assert_tensor_parity(
        ref_b,
        tri_b,
        tolerances=DEFAULT_LAYERWISE_TOLERANCES,
        context="fused_verifier_triton_cuda_bias",
    )
