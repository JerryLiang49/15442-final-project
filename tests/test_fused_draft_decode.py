"""Parity: fused draft decode reference vs two-pass ref vs Triton (CUDA).

Layout (see ``kv_kernels.fused_draft_decode`` module docstring):

* ``packed_k`` / ``packed_v``: ``[H, S_hist, D]`` uint8; draft uses **high nibble** only.
* ``k_scale_u`` / ``k_zp_u``: ``[H, S_hist, n_gk]`` with ``n_gk = D // group_size_k``.
* ``v_scale_u`` / ``v_zp_u``: ``[H, n_gv, D]`` with ``n_gv = ceil(S_hist / group_size_v)`` (token groups).
* ``k_recent`` / ``v_recent``: ``[H, S_rec, D]`` float.
* ``q``: ``[1, H, 1, D]``.
"""

from __future__ import annotations

import pytest
import torch

from kv_kernels.fused_draft_decode import (
    fused_draft_decode_attention,
    fused_draft_decode_attention_reference,
    fused_draft_decode_attention_reference_two_pass,
)
from kv_kernels.triton_runtime import triton_available


def _random_inputs(
    *,
    h: int,
    d: int,
    s_hist: int,
    s_rec: int,
    gs_k: int,
    gs_v: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    n_gk = d // gs_k
    n_gv = (s_hist + gs_v - 1) // gs_v
    q = torch.randn(1, h, 1, d, device=device, dtype=torch.float32)
    pk = torch.randint(0, 256, (h, s_hist, d), device=device, dtype=torch.uint8)
    pv = torch.randint(0, 256, (h, s_hist, d), device=device, dtype=torch.uint8)
    ks = torch.randn(h, s_hist, n_gk, device=device, dtype=torch.float32) * 0.1
    kz = torch.randn(h, s_hist, n_gk, device=device, dtype=torch.float32) * 0.1
    vs = torch.randn(h, n_gv, d, device=device, dtype=torch.float32) * 0.1
    vz = torch.randn(h, n_gv, d, device=device, dtype=torch.float32) * 0.1
    kr = torch.randn(h, s_rec, d, device=device, dtype=torch.float32)
    vr = torch.randn(h, s_rec, d, device=device, dtype=torch.float32)
    return q, pk, ks, kz, pv, vs, vz, kr, vr


def test_reference_matches_two_pass_cpu() -> None:
    dev = torch.device("cpu")
    q, pk, ks, kz, pv, vs, vz, kr, vr = _random_inputs(
        h=2, d=16, s_hist=5, s_rec=3, gs_k=4, gs_v=2, device=dev, seed=1
    )
    a = fused_draft_decode_attention_reference(
        q, pk, ks, kz, pv, vs, vz, kr, vr, group_size_k=4, group_size_v=2
    )
    b = fused_draft_decode_attention_reference_two_pass(
        q, pk, ks, kz, pv, vs, vz, kr, vr, group_size_k=4, group_size_v=2
    )
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
def test_triton_matches_reference_cuda() -> None:
    dev = torch.device("cuda:0")
    q, pk, ks, kz, pv, vs, vz, kr, vr = _random_inputs(
        h=4, d=64, s_hist=17, s_rec=5, gs_k=8, gs_v=4, device=dev, seed=42
    )
    ref = fused_draft_decode_attention_reference(
        q, pk, ks, kz, pv, vs, vz, kr, vr, group_size_k=8, group_size_v=4
    )
    tri = fused_draft_decode_attention(
        q,
        pk,
        ks,
        kz,
        pv,
        vs,
        vz,
        kr,
        vr,
        group_size_k=8,
        group_size_v=4,
        backend="triton",
    )
    torch.testing.assert_close(ref, tri, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
def test_triton_matches_reference_cuda_fp16_q() -> None:
    dev = torch.device("cuda:0")
    q, pk, ks, kz, pv, vs, vz, kr, vr = _random_inputs(
        h=2, d=32, s_hist=8, s_rec=2, gs_k=8, gs_v=4, device=dev, seed=7
    )
    q = q.to(torch.float16)
    kr = kr.half()
    vr = vr.half()
    ref = fused_draft_decode_attention_reference(
        q, pk, ks, kz, pv, vs, vz, kr, vr, group_size_k=8, group_size_v=4
    )
    tri = fused_draft_decode_attention(
        q,
        pk,
        ks,
        kz,
        pv,
        vs,
        vz,
        kr,
        vr,
        group_size_k=8,
        group_size_v=4,
        backend="triton",
    )
    torch.testing.assert_close(ref, tri, rtol=1e-3, atol=1e-3)
