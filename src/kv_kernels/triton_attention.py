"""Triton Q·K^T scores against **packed** INT4 history (explicit nibble loads; no full FP16 K materialization).

**Draft** — high nibble + upper ``(scale, zero_point)`` per channel group.

**Target** — low + high nibbles with separate upper/lower metadata (same layout as reference).

This module is **modular**: swap implementations behind :class:`kv_kernels.backend.KVKernelBackend`.
"""

from __future__ import annotations

import torch

from .reference_attention import qk_scores_draft_upper_only, qk_scores_target_upper_plus_lower
from .triton_runtime import require_triton, triton_available
from .tuning import active_kernel_tuning

if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _qk_draft_hist_kernel(
        q_ptr,
        packed_ptr,
        su_ptr,
        zu_ptr,
        out_ptr,
        D: tl.constexpr,
        GS: tl.constexpr,
        stride_packed_s: tl.constexpr,
        stride_su_r: tl.constexpr,
        stride_su_c: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        s = tl.program_id(0)
        acc = 0.0
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            qv = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(packed_ptr + s * stride_packed_s + offs, mask=mask, other=0)
            hi = ((b.to(tl.int32) >> 4) & 15).to(tl.float32)
            g = offs // GS
            su = tl.load(su_ptr + s * stride_su_r + g * stride_su_c, mask=mask, other=0.0).to(tl.float32)
            zu = tl.load(zu_ptr + s * stride_su_r + g * stride_su_c, mask=mask, other=0.0).to(tl.float32)
            kv = hi * su + zu
            acc += tl.sum(tl.where(mask, kv * qv, 0.0))
        tl.store(out_ptr + s, acc)

    @triton.jit
    def _qk_target_hist_kernel(
        q_ptr,
        packed_ptr,
        su_ptr,
        zu_ptr,
        sl_ptr,
        zl_ptr,
        out_ptr,
        D: tl.constexpr,
        GS: tl.constexpr,
        stride_packed_s: tl.constexpr,
        stride_su_r: tl.constexpr,
        stride_su_c: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        s = tl.program_id(0)
        acc = 0.0
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            qv = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(packed_ptr + s * stride_packed_s + offs, mask=mask, other=0)
            lo = (b.to(tl.int32) & 15).to(tl.float32)
            hi = ((b.to(tl.int32) >> 4) & 15).to(tl.float32)
            g = offs // GS
            su = tl.load(su_ptr + s * stride_su_r + g * stride_su_c, mask=mask, other=0.0).to(tl.float32)
            zu = tl.load(zu_ptr + s * stride_su_r + g * stride_su_c, mask=mask, other=0.0).to(tl.float32)
            sl = tl.load(sl_ptr + s * stride_su_r + g * stride_su_c, mask=mask, other=0.0).to(tl.float32)
            zl = tl.load(zl_ptr + s * stride_su_r + g * stride_su_c, mask=mask, other=0.0).to(tl.float32)
            kv = (hi * su + zu) + (lo * sl + zl)
            acc += tl.sum(tl.where(mask, kv * qv, 0.0))
        tl.store(out_ptr + s, acc)

    def qk_draft_hist_triton(
        q: torch.Tensor,
        packed_hist: torch.Tensor,
        k_scale_u: torch.Tensor,
        k_zp_u: torch.Tensor,
        *,
        group_size: int,
        block_d: int | None = None,
        num_warps: int | None = None,
    ) -> torch.Tensor:
        """CUDA Triton kernel: draft scores ``[S]`` (upper nibble path only)."""
        require_triton()
        tun = active_kernel_tuning()
        bd = int(block_d) if block_d is not None else int(tun.qk_hist_block_d)
        nw = int(num_warps) if num_warps is not None else int(tun.qk_hist_num_warps)
        if q.device.type != "cuda":
            raise ValueError("CUDA tensors required")
        if q.dim() != 1:
            raise ValueError("q must be [D]")
        d = int(q.shape[0])
        s = int(packed_hist.shape[0])
        if packed_hist.shape[1] != d:
            raise ValueError("packed_hist must be [S, D]")
        ng = d // group_size
        if ng * group_size != d:
            raise ValueError("D must divide group_size")
        if k_scale_u.shape != (s, ng) or k_zp_u.shape != (s, ng):
            raise ValueError("k_scale_u / k_zp_u must be [S, n_g]")

        out = torch.empty(s, device=q.device, dtype=torch.float32)
        packed = packed_hist.contiguous()
        su = k_scale_u.contiguous().to(torch.float32)
        zu = k_zp_u.contiguous().to(torch.float32)
        q32 = q.contiguous().to(torch.float16)

        stride_ps = packed.stride(0)
        stride_sur = su.stride(0)
        stride_suc = su.stride(1)

        _qk_draft_hist_kernel[(s,)](
            q32,
            packed,
            su,
            zu,
            out,
            D=d,
            GS=group_size,
            stride_packed_s=stride_ps,
            stride_su_r=stride_sur,
            stride_su_c=stride_suc,
            BLOCK_D=bd,
            num_warps=nw,
        )
        return out

    def qk_target_hist_triton(
        q: torch.Tensor,
        packed_hist: torch.Tensor,
        k_scale_u: torch.Tensor,
        k_zp_u: torch.Tensor,
        k_scale_l: torch.Tensor,
        k_zp_l: torch.Tensor,
        *,
        group_size: int,
        block_d: int | None = None,
        num_warps: int | None = None,
    ) -> torch.Tensor:
        """CUDA Triton kernel: target scores ``[S]`` (upper + lower)."""
        require_triton()
        tun = active_kernel_tuning()
        bd = int(block_d) if block_d is not None else int(tun.qk_hist_block_d)
        nw = int(num_warps) if num_warps is not None else int(tun.qk_hist_num_warps)
        if q.device.type != "cuda":
            raise ValueError("CUDA tensors required")
        d = int(q.shape[0])
        s = int(packed_hist.shape[0])
        ng = d // group_size
        out = torch.empty(s, device=q.device, dtype=torch.float32)
        packed = packed_hist.contiguous()
        su = k_scale_u.contiguous().to(torch.float32)
        zu = k_zp_u.contiguous().to(torch.float32)
        sl = k_scale_l.contiguous().to(torch.float32)
        zl = k_zp_l.contiguous().to(torch.float32)
        q32 = q.contiguous().to(torch.float16)

        stride_ps = packed.stride(0)
        stride_sur = su.stride(0)
        stride_suc = su.stride(1)

        _qk_target_hist_kernel[(s,)](
            q32,
            packed,
            su,
            zu,
            sl,
            zl,
            out,
            D=d,
            GS=group_size,
            stride_packed_s=stride_ps,
            stride_su_r=stride_sur,
            stride_su_c=stride_suc,
            BLOCK_D=bd,
            num_warps=nw,
        )
        return out


else:

    def qk_draft_hist_triton(*args, **kwargs):
        raise RuntimeError("Triton not available")

    def qk_target_hist_triton(*args, **kwargs):
        raise RuntimeError("Triton not available")


def qk_draft_dispatch(
    q: torch.Tensor,
    packed_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    *,
    group_size: int,
    backend: str,
) -> torch.Tensor:
    """Dispatch: ``\"ref\"`` | ``\"triton\"``."""
    if backend == "ref":
        return qk_scores_draft_upper_only(
            q, packed_hist, k_scale_u, k_zp_u, group_size=group_size
        )
    if backend == "triton":
        return qk_draft_hist_triton(
            q, packed_hist, k_scale_u, k_zp_u, group_size=group_size
        )
    raise ValueError(f"unknown backend {backend}")


def qk_target_dispatch(
    q: torch.Tensor,
    packed_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    k_scale_l: torch.Tensor,
    k_zp_l: torch.Tensor,
    *,
    group_size: int,
    backend: str,
) -> torch.Tensor:
    if backend == "ref":
        return qk_scores_target_upper_plus_lower(
            q,
            packed_hist,
            k_scale_u,
            k_zp_u,
            k_scale_l,
            k_zp_l,
            group_size=group_size,
        )
    if backend == "triton":
        return qk_target_hist_triton(
            q,
            packed_hist,
            k_scale_u,
            k_zp_u,
            k_scale_l,
            k_zp_l,
            group_size=group_size,
        )
    raise ValueError(f"unknown backend {backend}")
