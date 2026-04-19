"""Triton: pack two INT4 tensors into ``uint8`` (explicit low-bit path, no FP16 staging)."""

from __future__ import annotations

import torch

from .triton_runtime import require_triton, triton_available

if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _pack_int4_pair_kernel(
        lo_ptr,
        hi_ptr,
        out_ptr,
        n: int,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        lo = tl.load(lo_ptr + offs, mask=mask, other=0).to(tl.int32) & 15
        hi = tl.load(hi_ptr + offs, mask=mask, other=0).to(tl.int32) & 15
        packed = (lo | (hi << 4)).to(tl.uint8)
        tl.store(out_ptr + offs, packed, mask=mask)

    def pack_int4_pair_triton(
        low_nibble: torch.Tensor,
        high_nibble: torch.Tensor,
        *,
        block: int = 1024,
    ) -> torch.Tensor:
        """Pack ``low | (high << 4)`` into ``uint8`` (same shape). **CUDA + Triton required.**"""
        require_triton()
        if low_nibble.shape != high_nibble.shape:
            raise ValueError("shape mismatch")
        if low_nibble.device.type != "cuda":
            raise ValueError("pack_int4_pair_triton expects CUDA tensors")
        n = int(low_nibble.numel())
        out = torch.empty_like(low_nibble, dtype=torch.uint8, device=low_nibble.device)
        lo = low_nibble.contiguous().view(-1)
        hi = high_nibble.contiguous().view(-1)
        grid = (triton.cdiv(n, block),)
        _pack_int4_pair_kernel[grid](lo, hi, out.view(-1), n, BLOCK=block)
        return out.view_as(low_nibble)

else:

    def pack_int4_pair_triton(
        low_nibble: torch.Tensor,
        high_nibble: torch.Tensor,
        *,
        block: int = 1024,
    ) -> torch.Tensor:
        raise RuntimeError("Triton not available; install triton and use CUDA")


def append_packed_concat(
    existing: torch.Tensor | None,
    new_packed: torch.Tensor,
) -> torch.Tensor:
    """Concatenate packed chunks along sequence (dim 0): ``[S0, D]`` + ``[S1, D]`` → ``[S0+S1, D]``."""
    if existing is None:
        return new_packed.contiguous()
    return torch.cat([existing, new_packed], dim=0)
