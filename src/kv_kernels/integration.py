"""Hooks to validate kernels against reference before enabling Triton in experiments."""

from __future__ import annotations

import torch

from .reference_attention import qk_scores_draft_upper_only, qk_scores_target_upper_plus_lower
from .triton_attention import qk_draft_hist_triton, qk_target_hist_triton
from .triton_runtime import triton_available


def validate_qk_kernels_cuda(
    *,
    s: int = 64,
    d: int = 32,
    group_size: int = 8,
    atol: float = 1e-2,
    rtol: float = 1e-3,
) -> None:
    """Assert Triton Q·K score kernels match reference on random data (CUDA)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Triton kernel validation")
    if not triton_available():
        raise RuntimeError("Triton not installed")
    if d % group_size != 0:
        raise ValueError("d must divide group_size")
    ng = d // group_size
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    q = torch.randn(d, device=dev, dtype=torch.float16)
    packed = torch.randint(0, 256, (s, d), device=dev, dtype=torch.uint8)
    su = torch.randn(s, ng, device=dev, dtype=torch.float16)
    zu = torch.randn(s, ng, device=dev, dtype=torch.float16)
    sl = torch.randn(s, ng, device=dev, dtype=torch.float16) * 0.1
    zl = torch.randn(s, ng, device=dev, dtype=torch.float16) * 0.1

    ref_d = qk_scores_draft_upper_only(q, packed, su, zu, group_size=group_size)
    tri_d = qk_draft_hist_triton(q, packed, su, zu, group_size=group_size)
    if not torch.allclose(ref_d, tri_d, atol=atol, rtol=rtol):
        raise AssertionError(
            f"draft kernel mismatch max|diff|={(ref_d - tri_d).abs().max().item()}"
        )

    ref_t = qk_scores_target_upper_plus_lower(
        q, packed, su, zu, sl, zl, group_size=group_size
    )
    tri_t = qk_target_hist_triton(q, packed, su, zu, sl, zl, group_size=group_size)
    if not torch.allclose(ref_t, tri_t, atol=atol, rtol=rtol):
        raise AssertionError(
            f"target kernel mismatch max|diff|={(ref_t - tri_t).abs().max().item()}"
        )
