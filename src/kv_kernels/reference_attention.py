"""Reference FP32 attention scores vs quantized history (for kernel correctness).

**Layouts**

* ``packed_hist`` — ``uint8`` ``[S, D]``: low nibble = lower-INT4 code, high nibble = upper-INT4 code
  (same convention as :func:`cache.quant_spec_kv.pack_int4_pair`).
* ``k_scale_u``, ``k_zp_u`` — ``[S, n_g]`` channel-wise **upper** metadata for keys (``n_g = D // group_size``).
* ``k_scale_l``, ``k_zp_l`` — same shape for **lower** residual.

**Draft path** — uses **high nibble only** + ``(scale_u, zp_u)`` (explicit upper-only read).

**Target path** — ``dequant(upper) + dequant(lower)`` with distinct scales/zero-points.
"""

from __future__ import annotations

import torch


def _broadcast_group(
    t: torch.Tensor,
    *,
    group_size: int,
    d: int,
) -> torch.Tensor:
    """``t`` is ``[S, n_g]`` → ``[S, D]`` by repeating within each group."""
    s, n_g = t.shape
    return t.unsqueeze(-1).expand(s, n_g, group_size).reshape(s, d)


def qk_scores_draft_upper_only(
    q: torch.Tensor,
    packed_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """``q`` ``[D]`` vs history rows; returns logits ``[S]`` (one score per hist position).

    Reads **only** the upper nibble from ``packed_hist`` for the key value.
    """
    if q.dim() != 1:
        raise ValueError("q must be [D]")
    d = int(q.shape[0])
    s = int(packed_hist.shape[0])
    if packed_hist.shape[1] != d:
        raise ValueError("packed_hist must be [S, D]")
    hi = ((packed_hist.to(torch.int32) >> 4) & 15).to(torch.float32)
    su = _broadcast_group(k_scale_u, group_size=group_size, d=d)
    zu = _broadcast_group(k_zp_u, group_size=group_size, d=d)
    k_hat = hi * su + zu
    qf = q.to(torch.float32)
    return (k_hat * qf.unsqueeze(0)).sum(dim=-1)


def qk_scores_target_upper_plus_lower(
    q: torch.Tensor,
    packed_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    k_scale_l: torch.Tensor,
    k_zp_l: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Upper + lower dequant on K, then dot with ``q``."""
    if q.dim() != 1:
        raise ValueError("q must be [D]")
    d = int(q.shape[0])
    lo = (packed_hist.to(torch.int32) & 15).to(torch.float32)
    hi = ((packed_hist.to(torch.int32) >> 4) & 15).to(torch.float32)
    su = _broadcast_group(k_scale_u, group_size=group_size, d=d)
    zu = _broadcast_group(k_zp_u, group_size=group_size, d=d)
    sl = _broadcast_group(k_scale_l, group_size=group_size, d=d)
    zl = _broadcast_group(k_zp_l, group_size=group_size, d=d)
    k_hat = (hi * su + zu) + (lo * sl + zl)
    qf = q.to(torch.float32)
    return (k_hat * qf.unsqueeze(0)).sum(dim=-1)


def pack_upper_lower_int4(
    upper: torch.Tensor,
    lower: torch.Tensor,
) -> torch.Tensor:
    """``uint8`` packed tensor (same shape), convention matching :func:`cache.quant_spec_kv.pack_int4_pair`."""
    from cache.quant_spec_kv import pack_int4_pair

    return pack_int4_pair(lower, upper)
