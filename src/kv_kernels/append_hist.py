"""Append historical KV: quantize FP16 chunk → packed ``uint8`` + metadata (reference quant from :mod:`cache.quant_spec_kv`)."""

from __future__ import annotations

import torch

from cache.quant_spec_kv import quantize_key_channelwise_upper_lower

from cache.quant_spec_kv import pack_int4_pair as pack_int4_pair_ref

from .triton_pack import append_packed_concat, pack_int4_pair_triton, triton_available


def quantize_and_pack_key_hist_chunk(
    k_fp16: torch.Tensor,
    *,
    group_size: int,
    pack_backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``packed_uint8 [T,D]``, ``k_uq, k_us, k_uzp, k_lq, k_ls, k_lzp`` with channel-wise shapes.

    ``pack_backend``: ``\"ref\"`` (PyTorch pack) or ``\"triton\"`` (requires CUDA + Triton).
    """
    if k_fp16.dim() != 2:
        raise ValueError("k_fp16 must be [T, D]")
    t, d = k_fp16.shape
    if d % group_size != 0:
        raise ValueError("D must divide group_size")
    x = k_fp16.unsqueeze(0).unsqueeze(0)
    kuq, kus, kuzp, klq, kls, klzp = quantize_key_channelwise_upper_lower(x, group_size=group_size)
    kuq = kuq[0, 0]
    kus = kus[0, 0]
    kuzp = kuzp[0, 0]
    klq = klq[0, 0]
    kls = kls[0, 0]
    klzp = klzp[0, 0]

    lo = klq.to(torch.int32).clamp(0, 15).to(torch.int8)
    hi = kuq.to(torch.int32).clamp(0, 15).to(torch.int8)
    if pack_backend == "ref":
        packed = pack_int4_pair_ref(lo, hi).to(torch.uint8)
    elif pack_backend == "triton":
        if not triton_available() or k_fp16.device.type != "cuda":
            raise ValueError("triton pack requires CUDA + triton")
        packed = pack_int4_pair_triton(lo.to(k_fp16.device), hi.to(k_fp16.device))
    else:
        raise ValueError(pack_backend)

    return packed, kuq, kus, kuzp, klq, kls, klzp


def append_hist_packed_buffer(
    existing_packed: torch.Tensor | None,
    k_fp16_chunk: torch.Tensor,
    *,
    group_size: int,
    pack_backend: str,
) -> torch.Tensor:
    """Append one quantized+packed chunk along sequence (dim 0)."""
    packed, *_ = quantize_and_pack_key_hist_chunk(
        k_fp16_chunk, group_size=group_size, pack_backend=pack_backend
    )
    return append_packed_concat(existing_packed, packed)
