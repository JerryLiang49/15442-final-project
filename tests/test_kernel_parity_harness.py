"""Phase M — layerwise parity, metrics, fuzz, and failure triage for fused verifier attention."""

from __future__ import annotations

import pytest
import torch

from kv_kernels.fused_verifier_block_attention import (
    fused_verifier_block_attention,
    fused_verifier_block_attention_reference,
)
from kv_kernels.parity_harness import (
    DEFAULT_LAYERWISE_TOLERANCES,
    ParityTolerances,
    assert_tensor_parity,
    locate_worst_head_token,
    max_relative_error,
    tensor_parity_report,
    triage_fused_verifier_mismatch,
    triage_json_dumps,
)
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


def _call_fused(
    tensors: tuple[torch.Tensor, ...],
    *,
    gs_k: int,
    gs_v: int,
    backend: str,
) -> torch.Tensor:
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
    ) = tensors
    return fused_verifier_block_attention(
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
        group_size_k=gs_k,
        group_size_v=gs_v,
        backend=backend,
    )


def test_tensor_parity_identical() -> None:
    x = torch.randn(2, 3, 4, 5)
    rep = tensor_parity_report(x, x.clone())
    assert rep.passed and rep.max_abs == 0.0 and rep.allclose


def test_tensor_parity_metrics_mismatch() -> None:
    a = torch.ones(3, 3)
    b = torch.ones(3, 3)
    b[0, 0] = 2.0
    rep = tensor_parity_report(a, b, tolerances=ParityTolerances(rtol=0.0, atol=0.1))
    assert not rep.passed
    assert rep.max_abs == pytest.approx(1.0)


def test_max_relative_error_zero_safe() -> None:
    a = torch.tensor([0.0, 1.0])
    b = torch.tensor([0.0, 1.01])
    m = max_relative_error(a, b)
    assert m <= 0.02


def test_locate_worst_head_token() -> None:
    ref = torch.zeros(1, 2, 3, 4)
    cand = torch.zeros(1, 2, 3, 4)
    cand[0, 1, 2, :] = 1.0
    h, t, mx = locate_worst_head_token(ref, cand)
    assert (h, t) == (1, 2)
    assert mx > 0


def test_triage_json_roundtrip() -> None:
    q = torch.randn(1, 2, 3, 8)
    k_uq = torch.randint(-8, 8, (2, 4, 8), dtype=torch.int8)
    rep = tensor_parity_report(q, q + 1.0, tolerances=ParityTolerances(rtol=0.0, atol=0.0))
    tri = triage_fused_verifier_mismatch(
        backend_triton="triton",
        layer_idx=0,
        head_idx=1,
        query_row_t=2,
        gamma=3,
        s_hist=4,
        s_rec=1,
        group_size_k=4,
        group_size_v=2,
        k_uq_hist=k_uq,
        q=q,
        report=rep,
    )
    s = triage_json_dumps(tri)
    assert "parity" in s and "packed_k_upper_tile" in s


def test_cpu_reference_backend_matches_explicit_reference() -> None:
    """``backend='ref'`` must match :func:`fused_verifier_block_attention_reference` (ground truth)."""
    dev = torch.device("cpu")
    t = _random_case(h=2, d=16, s_hist=5, s_rec=2, gamma=3, gs_k=4, gs_v=2, device=dev, seed=7)
    ref = fused_verifier_block_attention_reference(
        *t,
        group_size_k=4,
        group_size_v=2,
    )
    got = _call_fused(t, gs_k=4, gs_v=2, backend="ref")
    assert_tensor_parity(ref, got, tolerances=ParityTolerances(rtol=0.0, atol=0.0), context="ref_dispatch")


@pytest.mark.parity_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
@pytest.mark.parametrize("seed", [0, 1, 2, 4, 11])
@pytest.mark.parametrize(
    "h,d,s_hist,s_rec,gamma,gs_k,gs_v",
    [
        (2, 32, 6, 2, 3, 8, 4),
        (4, 64, 10, 4, 4, 8, 4),
        (1, 128, 8, 0, 5, 8, 8),
    ],
)
def test_cuda_layerwise_fuzz_triton_vs_reference(
    seed: int,
    h: int,
    d: int,
    s_hist: int,
    s_rec: int,
    gamma: int,
    gs_k: int,
    gs_v: int,
) -> None:
    dev = torch.device("cuda:0")
    tensors = _random_case(
        h=h,
        d=d,
        s_hist=s_hist,
        s_rec=s_rec,
        gamma=gamma,
        gs_k=gs_k,
        gs_v=gs_v,
        device=dev,
        seed=seed,
    )
    ref = fused_verifier_block_attention_reference(
        *tensors,
        group_size_k=gs_k,
        group_size_v=gs_v,
    )
    tri = _call_fused(tensors, gs_k=gs_k, gs_v=gs_v, backend="triton")
    try:
        assert_tensor_parity(
            ref,
            tri,
            tolerances=DEFAULT_LAYERWISE_TOLERANCES,
            context=f"seed={seed} h={h} d={d} sh={s_hist} sr={s_rec} g={gamma}",
        )
    except AssertionError as exc:
        rep = tensor_parity_report(ref, tri, tolerances=DEFAULT_LAYERWISE_TOLERANCES)
        wh, wt, _ = locate_worst_head_token(ref, tri)
        triage = triage_fused_verifier_mismatch(
            backend_triton="triton",
            head_idx=int(wh),
            query_row_t=int(wt),
            gamma=gamma,
            s_hist=s_hist,
            s_rec=s_rec,
            group_size_k=gs_k,
            group_size_v=gs_v,
            k_uq_hist=tensors[1],
            k_lq_hist=tensors[2],
            k_scale_u=tensors[3],
            k_zp_u=tensors[4],
            k_block=tensors[14],
            q=tensors[0],
            report=rep,
        )
        raise AssertionError(f"{exc}\n{triage_json_dumps(triage)}") from exc


@pytest.mark.parity_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
def test_cuda_forced_triage_dump_shape() -> None:
    """Sanity: triage bundle contains debug fields without failing parity."""
    dev = torch.device("cuda:0")
    tensors = _random_case(h=2, d=32, s_hist=4, s_rec=1, gamma=2, gs_k=8, gs_v=4, device=dev, seed=99)
    ref = fused_verifier_block_attention_reference(*tensors, group_size_k=8, group_size_v=4)
    tri = _call_fused(tensors, gs_k=8, gs_v=4, backend="triton")
    rep = tensor_parity_report(ref, tri, tolerances=DEFAULT_LAYERWISE_TOLERANCES)
    triage = triage_fused_verifier_mismatch(
        backend_triton="triton",
        layer_idx=0,
        head_idx=0,
        query_row_t=0,
        gamma=2,
        s_hist=4,
        s_rec=1,
        group_size_k=8,
        group_size_v=4,
        k_uq_hist=tensors[1],
        report=rep,
    )
    js = triage_json_dumps(triage)
    assert "max_abs" in js or "parity" in js
