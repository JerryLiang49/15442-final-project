"""Kernel correctness (reference vs Triton) and append-pack; CUDA optional."""

from __future__ import annotations

import pytest
import torch

from kv_kernels.reference_attention import (
    qk_scores_draft_upper_only,
    qk_scores_target_upper_plus_lower,
)
from kv_kernels.triton_runtime import triton_available


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
def test_triton_qk_matches_reference() -> None:
    from kv_kernels.integration import validate_qk_kernels_cuda

    validate_qk_kernels_cuda(s=32, d=32, group_size=8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
def test_triton_pack_matches_reference() -> None:
    from cache.quant_spec_kv import pack_int4_pair as pack_ref
    from kv_kernels.triton_pack import pack_int4_pair_triton

    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    lo = torch.randint(0, 16, (256, 64), device=dev, dtype=torch.int8)
    hi = torch.randint(0, 16, (256, 64), device=dev, dtype=torch.int8)
    a = pack_ref(lo.cpu(), hi.cpu()).to(dev)
    b = pack_int4_pair_triton(lo, hi)
    assert torch.equal(a, b)


def test_reference_qk_cpu() -> None:
    s, d, gs = 4, 16, 4
    ng = d // gs
    q = torch.randn(d, dtype=torch.float32)
    packed = torch.randint(0, 256, (s, d), dtype=torch.uint8)
    su = torch.randn(s, ng)
    zu = torch.randn(s, ng)
    sl = torch.randn(s, ng) * 0.1
    zl = torch.randn(s, ng) * 0.1
    o1 = qk_scores_draft_upper_only(q, packed, su, zu, group_size=gs)
    o2 = qk_scores_target_upper_plus_lower(q, packed, su, zu, sl, zl, group_size=gs)
    assert o1.shape == (s,) and o2.shape == (s,)
