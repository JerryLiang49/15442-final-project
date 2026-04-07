"""Per-group symmetric INT4 with **nibble packing** (two 4-bit codes per :class:`torch.uint8`).

**Grouping**

The last dimension ``D`` of a KV tensor ``[B, H, L, D]`` is padded to a multiple of
``group_size`` (default **32**). Each contiguous group of ``group_size`` elements
gets one **symmetric** scale (same recipe as INT8 but with max value **7**).

**Packing**

Within a group, ``group_size`` signed integers in ``[-7, 7]`` are biased to ``[0, 15]``,
paired along the group axis, and stored as low/high nibbles in ``uint8`` requiring
``group_size // 2`` bytes per group.

**VRAM**

Payload is **half** the naive ``int4`` element count in bytes vs storing one byte per
quantized element (two values per stored byte).
"""

from __future__ import annotations

import torch

_DEFAULT_GROUP = 32


def symmetric_quantize_int4_grouped_packed(
    x: torch.Tensor,
    group_size: int = _DEFAULT_GROUP,
) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, ...]]:
    """Quantize last dimension in ``group_size`` chunks; pack pairs into ``uint8``.

    Returns:
        ``packed`` on the **same device** as ``x``, shape
        ``(*leading, ceil(D/g) * (g//2))`` where ``leading`` = all dims except last.
        ``scales`` float32 **CPU** tensor, shape ``(*leading, ceil(D/g))``,
        one scale per group.
        ``pad_amt`` extra zeros appended to D (0 if already aligned).
        ``original_shape`` full shape of ``x`` (for dequant trim).
    """
    if group_size % 2 != 0 or group_size < 2:
        raise ValueError("group_size must be a positive even integer")
    orig_shape = tuple(int(s) for s in x.shape)
    if orig_shape[-1] % group_size != 0:
        pad_amt = group_size - (orig_shape[-1] % group_size)
        x = torch.nn.functional.pad(x, (0, pad_amt))
    else:
        pad_amt = 0
    *lead, d_padded = x.shape
    g = d_padded // group_size
    xg = x.view(*lead, g, group_size).float()
    amax = xg.abs().amax(dim=-1)
    scale = (amax / 7.0).clamp(min=1e-8)
    q = torch.round(xg / scale.unsqueeze(-1)).clamp(-7, 7).to(torch.int32)
    qv = q.view(*lead, g, group_size // 2, 2)
    n0 = (qv[..., 0] + 8).clamp(0, 15).to(torch.uint8)
    n1 = (qv[..., 1] + 8).clamp(0, 15).to(torch.uint8)
    packed_5d = (n0 | (n1 << 4)).contiguous()
    packed = packed_5d.view(*lead, g * (group_size // 2))
    scale_cpu = scale.detach().cpu().to(torch.float32)
    return packed, scale_cpu, pad_amt, orig_shape


def symmetric_dequantize_int4_grouped_packed(
    packed: torch.Tensor,
    scale_cpu: torch.Tensor,
    *,
    original_shape: tuple[int, ...],
    pad_amt: int,
    group_size: int = _DEFAULT_GROUP,
    out_dtype: torch.dtype,
    out_device: torch.device,
) -> torch.Tensor:
    """Inverse of :func:`symmetric_quantize_int4_grouped_packed`."""
    _ = pad_amt
    *inner, n_bytes = packed.shape
    g = n_bytes // (group_size // 2)
    low = (packed & 0xF).to(torch.int32) - 8
    high = ((packed >> 4) & 0xF).to(torch.int32) - 8
    stacked = torch.stack([low, high], dim=-1)
    q_flat = stacked.reshape(*inner, n_bytes * 2)
    q_groups = q_flat.reshape(*inner, g, group_size)
    s = scale_cpu.to(device=out_device, dtype=torch.float32).unsqueeze(-1)
    xf = q_groups.float() * s
    out_padded = xf.reshape(*inner, g * group_size)
    d_orig = original_shape[-1]
    out = out_padded[..., :d_orig]
    return out.to(dtype=out_dtype, device=out_device)
