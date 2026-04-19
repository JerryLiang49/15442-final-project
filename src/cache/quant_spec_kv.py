"""QuantSpec-style KV quantization (reference Python, correctness-first).

**Axes (paper-style)**

* **Keys** — *channel-wise*: groups partition the **head dimension** ``D`` into chunks of
  ``group_size`` (default ``D`` → one group per token position). Each group gets asymmetric
  INT4 **upper**, then residual ``r = x - x_hat_upper`` gets **lower** INT4.
* **Values** — *token-wise*: groups partition the **sequence** length ``S`` into chunks of
  ``group_size`` tokens. Per group and **per channel** ``d`` we have scales (shape
  ``[B, H, n_token_groups, D]``).

**Asymmetric INT4**

Unsigned 16 levels: for a 1D float group ``x``,

* ``x_min = min(x)``, ``x_max = max(x)``, ``scale = (x_max - x_min) / 15`` (or ``1`` if degenerate),
* ``q = round((x - x_min) / scale).clamp(0, 15)`` stored as ``int8`` in ``[0, 15]``,
* ``x_hat = q.float() * scale + x_min``.

**Upper / lower**

1. Upper: quantize FP16 → ``q_u``, ``scale_u``, ``zp_u`` (``zp`` = ``x_min`` in our parameterization).
2. ``r = x - dequant(q_u, scale_u, zp_u)``.
3. Lower: same asymmetric INT4 on ``r`` (same grouping as K or V for that tensor).

**Views**

* **Draft** — dequantize **upper only** for K and V.
* **Target** — ``dequant(upper) + dequant(lower)`` (residual sum).

**Packing**

Optional packed nibbles: two 4-bit codes per ``uint8`` (see :func:`pack_int4_pair`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

N_LEVELS = 15  # 16 levels 0..15


def asymmetric_int4_quantize(
    x: torch.Tensor,
    *,
    dim_reduce: int | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric INT4 (16 levels) along reduced dims.

    Args:
        x: Tensor to quantize.
        dim_reduce: dims over which min/max are taken (e.g. last dim for channel groups).

    Returns:
        ``q`` same shape as ``x`` (int8 in ``[0, 15]``), ``scale``, ``zero_point`` broadcastable
        to ``x`` (FP32 for numerics; cast by caller).
    """
    xf = x.float()
    x_min = xf.amin(dim=dim_reduce, keepdim=True)
    x_max = xf.amax(dim=dim_reduce, keepdim=True)
    span = (x_max - x_min).clamp_min(1e-6)
    scale = span / float(N_LEVELS)
    q = torch.round((xf - x_min) / scale).clamp(0, 15).to(torch.int8)
    return q, scale.to(x.dtype), x_min.to(x.dtype)


def asymmetric_int4_dequantize(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """``x_hat = q * scale + zero_point`` (``zero_point`` is ``x_min`` from quantize)."""
    return q.float() * scale.float() + zero_point.float()


def pack_int4_pair(low_nibble: torch.Tensor, high_nibble: torch.Tensor) -> torch.Tensor:
    """Pack two INT4 tensors (values 0..15) into ``uint8``: ``low | (high << 4)``.

    ``low_nibble`` and ``high_nibble`` must match shape; last dim must be even to pack pairs
    along last axis, or identical shapes for elementwise pack.
    """
    if low_nibble.shape != high_nibble.shape:
        raise ValueError("shape mismatch")
    lo = low_nibble.to(torch.int16).clamp(0, 15)
    hi = high_nibble.to(torch.int16).clamp(0, 15)
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4_pair(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack ``uint8`` into two INT4 tensors (0..15)."""
    lo = (packed.to(torch.int16) & 15).to(torch.int8)
    hi = ((packed.to(torch.int16) >> 4) & 15).to(torch.int8)
    return lo, hi


def _key_channel_groups(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """``x`` is [B,H,S,D]. Returns grouped view [B,H,S,n_g,gs] and ``n_g``."""
    b, h, s, d = x.shape
    if d % group_size != 0:
        raise ValueError(f"head_dim {d} not divisible by group_size {group_size}")
    n_g = d // group_size
    return x.view(b, h, s, n_g, group_size), n_g


def _value_token_groups(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """``x`` is [B,H,S,D]. Groups along S: [B,H,n_g,gs,D]."""
    b, h, s, d = x.shape
    if s % group_size != 0:
        raise ValueError(f"sequence {s} not divisible by group_size {group_size}")
    n_g = s // group_size
    return x.view(b, h, n_g, group_size, d), n_g


def quantize_key_channelwise_upper_lower(
    k_fp16: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keys: channel-wise groups along ``D``; upper then lower on residual.

    Returns:
        ``k_uq, k_us, k_uzp, k_lq, k_ls, k_lzp`` with ``k_uq`` shape ``[B,H,S,D]`` int8,
        ``k_us``, ``k_uzp`` shape ``[B,H,S,n_g]``.
    """
    b, h, s, d = k_fp16.shape
    if d % group_size != 0:
        raise ValueError("head_dim must be divisible by group_size")
    n_g = d // group_size
    x = k_fp16.view(b, h, s, n_g, group_size)
    # min/max over channel group (last dim)
    k_uq, k_us, k_uzp = asymmetric_int4_quantize(x, dim_reduce=-1)
    k_hat_u = asymmetric_int4_dequantize(k_uq, k_us, k_uzp)
    resid = x - k_hat_u
    k_lq, k_ls, k_lzp = asymmetric_int4_quantize(resid, dim_reduce=-1)
    return (
        k_uq.view(b, h, s, d),
        k_us.squeeze(-1),
        k_uzp.squeeze(-1),
        k_lq.view(b, h, s, d),
        k_ls.squeeze(-1),
        k_lzp.squeeze(-1),
    )


def quantize_value_tokenwise_upper_lower(
    v_fp16: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Values: token-wise groups along ``S``; upper then lower on residual.

    Scales are per ``(b, h, token_group, d)``: shape ``[B, H, n_g, D]``.
    Codes ``v_uq`` shape ``[B,H,S,D]``.
    """
    b, h, s, d = v_fp16.shape
    if s % group_size != 0:
        raise ValueError("sequence length must be divisible by group_size")
    n_g = s // group_size
    x = v_fp16.view(b, h, n_g, group_size, d)
    # min/max over token group (dim 3)
    v_uq, v_us, v_uzp = asymmetric_int4_quantize(x, dim_reduce=3)
    v_hat_u = asymmetric_int4_dequantize(v_uq, v_us, v_uzp)
    resid = x - v_hat_u
    v_lq, v_ls, v_lzp = asymmetric_int4_quantize(resid, dim_reduce=3)
    return (
        v_uq.view(b, h, s, d),
        v_us.squeeze(3),
        v_uzp.squeeze(3),
        v_lq.view(b, h, s, d),
        v_ls.squeeze(3),
        v_lzp.squeeze(3),
    )


def reconstruct_key_draft(k_uq: torch.Tensor, k_us: torch.Tensor, k_uzp: torch.Tensor) -> torch.Tensor:
    """Draft path: upper only; ``k_us`` ``[B,H,S,n_g]`` broadcast to ``D``.

    Returns float32 dequantized tensor (cast to FP16 at integration boundaries).
    """
    b, h, s, d = k_uq.shape
    n_g = k_us.shape[-1]
    gs = d // n_g
    x = k_uq.view(b, h, s, n_g, gs)
    su = k_us.unsqueeze(-1)
    zu = k_uzp.unsqueeze(-1)
    # Output is float (not int8 codes); caller casts to FP16/FP32 as needed.
    return asymmetric_int4_dequantize(x, su, zu).view(b, h, s, d)


def reconstruct_key_target(
    k_uq: torch.Tensor,
    k_us: torch.Tensor,
    k_uzp: torch.Tensor,
    k_lq: torch.Tensor,
    k_ls: torch.Tensor,
    k_lzp: torch.Tensor,
) -> torch.Tensor:
    """Target: upper + lower residual. Returns float32."""
    b, h, s, d = k_uq.shape
    n_g = k_us.shape[-1]
    gs = d // n_g
    ku = k_uq.view(b, h, s, n_g, gs)
    su = k_us.unsqueeze(-1)
    zu = k_uzp.unsqueeze(-1)
    upper = asymmetric_int4_dequantize(ku, su, zu)
    kl = k_lq.view(b, h, s, n_g, gs)
    sl = k_ls.unsqueeze(-1)
    zl = k_lzp.unsqueeze(-1)
    lower = asymmetric_int4_dequantize(kl, sl, zl)
    return (upper + lower).view(b, h, s, d)


def reconstruct_value_draft(
    v_uq: torch.Tensor,
    v_us: torch.Tensor,
    v_uzp: torch.Tensor,
    *,
    group_size: int,
    logical_seq_len: int | None = None,
) -> torch.Tensor:
    """Draft: upper only; ``v_us`` shape ``[B,H,n_g,D]``. Returns float32.

    Stored codes use sequence length ``n_g * group_size`` (padded) when the logical history length is
    not divisible by ``group_size``; pass ``logical_seq_len`` (e.g. ``hist_len``) to trim the
    reconstructed FP16 prefix to match keys.
    """
    b, h, s, d = v_uq.shape
    n_g = v_us.shape[2]
    gs = int(group_size)
    if s != n_g * gs:
        raise ValueError(
            f"value codes length s={s} must equal n_g*group_size ({n_g}*{gs}={n_g * gs}); "
            "fix V storage (prefill/rollover should keep padded length)."
        )
    x = v_uq.view(b, h, n_g, gs, d)
    su = v_us.unsqueeze(3)
    zu = v_uzp.unsqueeze(3)
    out = asymmetric_int4_dequantize(x, su, zu).view(b, h, s, d)
    if logical_seq_len is not None:
        return out[:, :, : int(logical_seq_len), :]
    return out


def reconstruct_value_target(
    v_uq: torch.Tensor,
    v_us: torch.Tensor,
    v_uzp: torch.Tensor,
    v_lq: torch.Tensor,
    v_ls: torch.Tensor,
    v_lzp: torch.Tensor,
    *,
    group_size: int,
    logical_seq_len: int | None = None,
) -> torch.Tensor:
    """Target: upper + lower. Returns float32. See :func:`reconstruct_value_draft` for padding rules."""
    b, h, s, d = v_uq.shape
    n_g = v_us.shape[2]
    gs = int(group_size)
    if s != n_g * gs:
        raise ValueError(
            f"value codes length s={s} must equal n_g*group_size ({n_g}*{gs}={n_g * gs}); "
            "fix V storage (prefill/rollover should keep padded length)."
        )
    x = v_uq.view(b, h, n_g, gs, d)
    su = v_us.unsqueeze(3)
    zu = v_uzp.unsqueeze(3)
    upper = asymmetric_int4_dequantize(x, su, zu)
    xl = v_lq.view(b, h, n_g, gs, d)
    sl = v_ls.unsqueeze(3)
    zl = v_lzp.unsqueeze(3)
    lower = asymmetric_int4_dequantize(xl, sl, zl)
    out = (upper + lower).view(b, h, s, d)
    if logical_seq_len is not None:
        return out[:, :, : int(logical_seq_len), :]
    return out


def quantize_fp16_kv_to_upper_lower(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize historical K/V with distinct K and V pathways."""
    kuq, kus, kuzp, klq, kls, klzp = quantize_key_channelwise_upper_lower(k, group_size=group_size)
    vuq, vus, vuzp, vlq, vls, vlzp = quantize_value_tokenwise_upper_lower(v, group_size=group_size)
    return kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp


@dataclass
class QuantizedHistKV:
    """Packed historical KV for one layer."""

    k_uq: torch.Tensor
    k_us: torch.Tensor
    k_uzp: torch.Tensor
    k_lq: torch.Tensor
    k_ls: torch.Tensor
    k_lzp: torch.Tensor
    v_uq: torch.Tensor
    v_us: torch.Tensor
    v_uzp: torch.Tensor
    v_lq: torch.Tensor
    v_ls: torch.Tensor
    v_lzp: torch.Tensor
    group_size: int
