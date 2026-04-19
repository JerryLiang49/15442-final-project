"""Fused verifier block attention: ``q_len = γ`` over QuantSpec **target** view (upper+lower INT8) + FP16 tail + block.

**Layout (matches :mod:`cache.hierarchical_kv_store` + :mod:`cache.quant_spec_kv`)**

* **Key codes** — separate ``int8`` tensors ``k_uq``, ``k_lq`` shaped ``[H, S_h, D]`` with codes in ``0..15``
  (same semantics as packed nibbles, but **not** merged into one ``uint8`` byte).
* **Value codes** — ``v_uq``, ``v_lq`` shaped ``[H, S_h, D]`` (token-wise upper/lower groups along ``S``).
* **K metadata** — channel groups along ``D``: ``g = d // group_size_k``;
  ``k_scale_u``, ``k_zp_u``, ``k_scale_l``, ``k_zp_l`` each ``[H, S_h, n_gk]``.
* **V metadata** — token groups: ``tg = s // group_size_v``;
  ``v_scale_u``, ``v_zp_u``, ``v_scale_l``, ``v_zp_l`` each ``[H, n_gv, D]``.
* **Recent** — FP16 (or FP32) ``k_recent``, ``v_recent`` shaped ``[H, S_r, D]`` (committed FP16 prefix
  immediately after history).
* **Draft block** — FP16 ``k_block``, ``v_block`` shaped ``[H, γ, D]`` (proposed tokens’ KV for this verify).

**Target reconstruction (per channel / token group)**

* ``k_hat[d] = (hi*su+zu) + (lo*sl+zl)`` with ``hi = k_uq[...,d] & 15``, ``lo = k_lq[...,d] & 15``.
* ``v_hat[d] = (hi*su+zu) + (lo*sl+zl)`` with token group ``tg = s // group_size_v`` for history index ``s``.

**Causal mask (block verification)**

* Prefix length ``P = S_h + S_r`` (history + recent; no draft tokens in prefix).
* Total key positions ``L = P + γ`` (concat along sequence: hist → recent → draft block).
* Query row ``t ∈ {0,…,γ-1}`` (verify position within block) may attend only to keys ``k`` with::

      k <= P + t

  i.e. full prefix plus draft slots ``0..t`` inclusive, **not** future draft positions ``t+1..γ-1``.

**Attention**

* ``logit_k = scale_attn * dot(q[t], K[k])`` with ``scale_attn = D ** -0.5``.
* Softmax over **allowed** keys only (masked causal block attention).

**Scope**

* Primary: ``B = 1``. Multi-batch may fall back to reference.
* No full ``[L, D]`` FP16 KV tensor in global memory on the Triton path (recompute per key in registers).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .triton_runtime import require_triton, triton_available
from .tuning import active_kernel_tuning

# ---------------------------------------------------------------------------
# Reference (may materialize per-row dequant for testing)
# ---------------------------------------------------------------------------


def _dequant_k_target_row(
    k_uq_row: torch.Tensor,
    k_lq_row: torch.Tensor,
    k_su_row: torch.Tensor,
    k_zu_row: torch.Tensor,
    k_sl_row: torch.Tensor,
    k_zl_row: torch.Tensor,
    *,
    group_size_k: int,
) -> torch.Tensor:
    """``[D]`` int8 upper/lower codes → FP32 key row."""
    d = int(k_uq_row.shape[0])
    hi = (k_uq_row.to(torch.int32) & 15).to(torch.float32)
    lo = (k_lq_row.to(torch.int32) & 15).to(torch.float32)
    n_g = d // group_size_k
    g = torch.arange(d, device=k_uq_row.device, dtype=torch.int64) // group_size_k
    su = k_su_row[g].to(torch.float32)
    zu = k_zu_row[g].to(torch.float32)
    sl = k_sl_row[g].to(torch.float32)
    zl = k_zl_row[g].to(torch.float32)
    return hi * su + zu + lo * sl + zl


def _dequant_v_target_row(
    v_uq_row: torch.Tensor,
    v_lq_row: torch.Tensor,
    s: int,
    v_su: torch.Tensor,
    v_zu: torch.Tensor,
    v_sl: torch.Tensor,
    v_zl: torch.Tensor,
    *,
    group_size_v: int,
) -> torch.Tensor:
    """History index ``s``; ``v_*`` are ``[n_gv, D]`` per head."""
    d = int(v_uq_row.shape[0])
    hi = (v_uq_row.to(torch.int32) & 15).to(torch.float32)
    lo = (v_lq_row.to(torch.int32) & 15).to(torch.float32)
    tg = s // group_size_v
    su = v_su[tg].to(torch.float32)
    zu = v_zu[tg].to(torch.float32)
    sl = v_sl[tg].to(torch.float32)
    zl = v_zl[tg].to(torch.float32)
    return hi * su + zu + lo * sl + zl


def fused_verifier_block_attention_reference(
    q: torch.Tensor,
    k_uq_hist: torch.Tensor,
    k_lq_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    k_scale_l: torch.Tensor,
    k_zp_l: torch.Tensor,
    v_uq_hist: torch.Tensor,
    v_lq_hist: torch.Tensor,
    v_scale_u: torch.Tensor,
    v_zp_u: torch.Tensor,
    v_scale_l: torch.Tensor,
    v_zp_l: torch.Tensor,
    k_recent: torch.Tensor,
    v_recent: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    *,
    group_size_k: int,
    group_size_v: int,
    scale_attn: float | None = None,
) -> torch.Tensor:
    """Reference attention output ``[B, H, γ, D]`` (materialized dequant per key row).

    Shapes: ``q`` ``[B, H, γ, D]``; hist int8 ``[H, S_h, D]``; recent/block ``[H, S_r, D]`` and ``[H, γ, D]``.
    """
    if q.dim() != 4:
        raise ValueError("q must be [B, H, γ, D]")
    b, h, gamma, d = q.shape
    if scale_attn is None:
        scale_attn = float(d**-0.5)
    s_hist = int(k_uq_hist.shape[1])
    s_rec = int(k_recent.shape[1])
    p = s_hist + s_rec
    l_total = p + gamma

    out = torch.zeros(b, h, gamma, d, device=q.device, dtype=torch.float32)
    qf = q.to(torch.float32)

    for bi in range(b):
        for head in range(h):
            for t in range(gamma):
                qv = qf[bi, head, t]
                logits = torch.empty(l_total, device=q.device, dtype=torch.float32)
                v_rows: list[torch.Tensor] = []
                for k in range(l_total):
                    if k > p + t:
                        logits[k] = float("-inf")
                        v_rows.append(torch.zeros(d, device=q.device, dtype=torch.float32))
                        continue
                    if k < s_hist:
                        kr_u = k_uq_hist[head, k]
                        kr_l = k_lq_hist[head, k]
                        k_hat = _dequant_k_target_row(
                            kr_u,
                            kr_l,
                            k_scale_u[head, k],
                            k_zp_u[head, k],
                            k_scale_l[head, k],
                            k_zp_l[head, k],
                            group_size_k=group_size_k,
                        )
                        vr_u = v_uq_hist[head, k]
                        vr_l = v_lq_hist[head, k]
                        v_hat = _dequant_v_target_row(
                            vr_u,
                            vr_l,
                            k,
                            v_scale_u[head],
                            v_zp_u[head],
                            v_scale_l[head],
                            v_zp_l[head],
                            group_size_v=group_size_v,
                        )
                    elif k < s_hist + s_rec:
                        j = k - s_hist
                        k_hat = k_recent[head, j].to(torch.float32)
                        v_hat = v_recent[head, j].to(torch.float32)
                    else:
                        j = k - p
                        k_hat = k_block[head, j].to(torch.float32)
                        v_hat = v_block[head, j].to(torch.float32)
                    logits[k] = (qv * k_hat).sum() * scale_attn
                    v_rows.append(v_hat)

                p_attn = torch.softmax(logits, dim=0)
                acc = torch.zeros(d, device=q.device, dtype=torch.float32)
                for k in range(l_total):
                    acc = acc + p_attn[k] * v_rows[k]
                out[bi, head, t] = acc

    return out.to(q.dtype)


def fused_verifier_block_attention_reference_logits(
    q: torch.Tensor,
    k_uq_hist: torch.Tensor,
    k_lq_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    k_scale_l: torch.Tensor,
    k_zp_l: torch.Tensor,
    k_recent: torch.Tensor,
    k_block: torch.Tensor,
    *,
    group_size_k: int,
    scale_attn: float | None = None,
) -> torch.Tensor:
    """Masked logits ``[B, H, γ, L]`` with ``L = S_h + S_r + γ``; masked entries ``-inf``."""
    if q.dim() != 4:
        raise ValueError("q must be [B, H, γ, D]")
    b, h, gamma, dim = q.shape
    if scale_attn is None:
        scale_attn = float(dim**-0.5)
    s_hist = int(k_uq_hist.shape[1])
    s_rec = int(k_recent.shape[1])
    p = s_hist + s_rec
    l_total = p + gamma
    logits = torch.full((b, h, gamma, l_total), float("-inf"), device=q.device, dtype=torch.float32)
    qf = q.to(torch.float32)

    for bi in range(b):
        for head in range(h):
            for t in range(gamma):
                qv = qf[bi, head, t]
                for k in range(l_total):
                    if k > p + t:
                        continue
                    if k < s_hist:
                        k_hat = _dequant_k_target_row(
                            k_uq_hist[head, k],
                            k_lq_hist[head, k],
                            k_scale_u[head, k],
                            k_zp_u[head, k],
                            k_scale_l[head, k],
                            k_zp_l[head, k],
                            group_size_k=group_size_k,
                        )
                    elif k < s_hist + s_rec:
                        k_hat = k_recent[head, k - s_hist].to(torch.float32)
                    else:
                        k_hat = k_block[head, k - p].to(torch.float32)
                    logits[bi, head, t, k] = (qv * k_hat).sum() * scale_attn
    return logits


# ---------------------------------------------------------------------------
# Triton (B=1; one program per head × query row)
# ---------------------------------------------------------------------------

if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_verifier_block_kernel(
        q_ptr,
        out_ptr,
        k_uq_ptr,
        k_lq_ptr,
        k_su_ptr,
        k_zu_ptr,
        k_sl_ptr,
        k_zl_ptr,
        v_uq_ptr,
        v_lq_ptr,
        v_su_ptr,
        v_zu_ptr,
        v_sl_ptr,
        v_zl_ptr,
        k_rec_ptr,
        v_rec_ptr,
        k_blk_ptr,
        v_blk_ptr,
        S_HIST: tl.constexpr,
        S_REC: tl.constexpr,
        GAMMA: tl.constexpr,
        D: tl.constexpr,
        GS_K: tl.constexpr,
        GS_V: tl.constexpr,
        P: tl.constexpr,
        L: tl.constexpr,
        scale_attn: tl.constexpr,
        stride_q_h: tl.constexpr,
        stride_q_t: tl.constexpr,
        stride_q_d: tl.constexpr,
        stride_kh_h: tl.constexpr,
        stride_kh_s: tl.constexpr,
        stride_kh_d: tl.constexpr,
        stride_ksu_h: tl.constexpr,
        stride_ksu_s: tl.constexpr,
        stride_ksu_g: tl.constexpr,
        stride_vs_h: tl.constexpr,
        stride_vs_tg: tl.constexpr,
        stride_vs_d: tl.constexpr,
        stride_kr_h: tl.constexpr,
        stride_kr_s: tl.constexpr,
        stride_kr_d: tl.constexpr,
        stride_kb_h: tl.constexpr,
        stride_kb_t: tl.constexpr,
        stride_kb_d: tl.constexpr,
        stride_out_h: tl.constexpr,
        stride_out_t: tl.constexpr,
        stride_out_d: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per (head, query index t); online softmax over causal keys; two-pass output."""
        head = tl.program_id(0)
        t = tl.program_id(1)

        # --- Pass 1: online softmax over k in [0, L) with mask k <= P + t ---
        m_acc = -1.0e30
        l_acc = 0.0

        for k in range(L):
            if k <= P + t:
                dot_k = 0.0
                for d0 in range(0, D, BLOCK_D):
                    offs = d0 + tl.arange(0, BLOCK_D)
                    mask = offs < D
                    qv = tl.load(
                        q_ptr + head * stride_q_h + t * stride_q_t + offs * stride_q_d,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)

                    if k < S_HIST:
                        ku = tl.load(
                            k_uq_ptr + head * stride_kh_h + k * stride_kh_s + offs * stride_kh_d,
                            mask=mask,
                            other=0,
                        )
                        kl = tl.load(
                            k_lq_ptr + head * stride_kh_h + k * stride_kh_s + offs * stride_kh_d,
                            mask=mask,
                            other=0,
                        )
                        hi = (ku.to(tl.int32) & 15).to(tl.float32)
                        lo = (kl.to(tl.int32) & 15).to(tl.float32)
                        g = offs // GS_K
                        su = tl.load(
                            k_su_ptr + head * stride_ksu_h + k * stride_ksu_s + g * stride_ksu_g,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                        zu = tl.load(
                            k_zu_ptr + head * stride_ksu_h + k * stride_ksu_s + g * stride_ksu_g,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                        sl = tl.load(
                            k_sl_ptr + head * stride_ksu_h + k * stride_ksu_s + g * stride_ksu_g,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                        zl = tl.load(
                            k_zl_ptr + head * stride_ksu_h + k * stride_ksu_s + g * stride_ksu_g,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                        kv = hi * su + zu + lo * sl + zl
                    elif k < S_HIST + S_REC:
                        j = k - S_HIST
                        kv = tl.load(
                            k_rec_ptr + head * stride_kr_h + j * stride_kr_s + offs * stride_kr_d,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                    else:
                        j = k - P
                        kv = tl.load(
                            k_blk_ptr + head * stride_kb_h + j * stride_kb_t + offs * stride_kb_d,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)

                    dot_k += tl.sum(tl.where(mask, qv * kv, 0.0))

                logit = dot_k * scale_attn
                m_new = tl.maximum(m_acc, logit)
                l_acc = l_acc * tl.exp(m_acc - m_new) + tl.exp(logit - m_new)
                m_acc = m_new

        m_final = m_acc
        l_final = l_acc

        # --- Pass 2: weighted V ---
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)

            for k in range(L):
                if k <= P + t:
                    logit = 0.0
                    for dd in range(0, D, BLOCK_D):
                        o2 = dd + tl.arange(0, BLOCK_D)
                        m2 = o2 < D
                        q2 = tl.load(
                            q_ptr + head * stride_q_h + t * stride_q_t + o2 * stride_q_d,
                            mask=m2,
                            other=0.0,
                        ).to(tl.float32)

                        if k < S_HIST:
                            ku = tl.load(
                                k_uq_ptr + head * stride_kh_h + k * stride_kh_s + o2 * stride_kh_d,
                                mask=m2,
                                other=0,
                            )
                            kl = tl.load(
                                k_lq_ptr + head * stride_kh_h + k * stride_kh_s + o2 * stride_kh_d,
                                mask=m2,
                                other=0,
                            )
                            hi = (ku.to(tl.int32) & 15).to(tl.float32)
                            lo = (kl.to(tl.int32) & 15).to(tl.float32)
                            g2 = o2 // GS_K
                            su = tl.load(
                                k_su_ptr + head * stride_ksu_h + k * stride_ksu_s + g2 * stride_ksu_g,
                                mask=m2,
                                other=0.0,
                            ).to(tl.float32)
                            zu = tl.load(
                                k_zu_ptr + head * stride_ksu_h + k * stride_ksu_s + g2 * stride_ksu_g,
                                mask=m2,
                                other=0.0,
                            ).to(tl.float32)
                            sl = tl.load(
                                k_sl_ptr + head * stride_ksu_h + k * stride_ksu_s + g2 * stride_ksu_g,
                                mask=m2,
                                other=0.0,
                            ).to(tl.float32)
                            zl = tl.load(
                                k_zl_ptr + head * stride_ksu_h + k * stride_ksu_s + g2 * stride_ksu_g,
                                mask=m2,
                                other=0.0,
                            ).to(tl.float32)
                            kv2 = hi * su + zu + lo * sl + zl
                        elif k < S_HIST + S_REC:
                            j2 = k - S_HIST
                            kv2 = tl.load(
                                k_rec_ptr + head * stride_kr_h + j2 * stride_kr_s + o2 * stride_kr_d,
                                mask=m2,
                                other=0.0,
                            ).to(tl.float32)
                        else:
                            j2 = k - P
                            kv2 = tl.load(
                                k_blk_ptr + head * stride_kb_h + j2 * stride_kb_t + o2 * stride_kb_d,
                                mask=m2,
                                other=0.0,
                            ).to(tl.float32)

                        logit += tl.sum(tl.where(m2, q2 * kv2, 0.0))

                    logit = logit * scale_attn
                    w = tl.exp(logit - m_final) / l_final

                    if k < S_HIST:
                        vu = tl.load(
                            v_uq_ptr + head * stride_kh_h + k * stride_kh_s + offs * stride_kh_d,
                            mask=mask,
                            other=0,
                        )
                        vl = tl.load(
                            v_lq_ptr + head * stride_kh_h + k * stride_kh_s + offs * stride_kh_d,
                            mask=mask,
                            other=0,
                        )
                        hi_v = (vu.to(tl.int32) & 15).to(tl.float32)
                        lo_v = (vl.to(tl.int32) & 15).to(tl.float32)
                        tg = k // GS_V
                        sv = tl.load(
                            v_su_ptr + head * stride_vs_h + tg * stride_vs_tg + offs * stride_vs_d,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                        zv_u = tl.load(
                            v_zu_ptr + head * stride_vs_h + tg * stride_vs_tg + offs * stride_vs_d,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                        slv = tl.load(
                            v_sl_ptr + head * stride_vs_h + tg * stride_vs_tg + offs * stride_vs_d,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                        zlv = tl.load(
                            v_zl_ptr + head * stride_vs_h + tg * stride_vs_tg + offs * stride_vs_d,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                        vv = hi_v * sv + zv_u + lo_v * slv + zlv
                    elif k < S_HIST + S_REC:
                        jv = k - S_HIST
                        vv = tl.load(
                            v_rec_ptr + head * stride_kr_h + jv * stride_kr_s + offs * stride_kr_d,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)
                    else:
                        jv = k - P
                        vv = tl.load(
                            v_blk_ptr + head * stride_kb_h + jv * stride_kb_t + offs * stride_kb_d,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)

                    acc += w * vv

            tl.store(
                out_ptr + head * stride_out_h + t * stride_out_t + offs * stride_out_d,
                acc,
                mask=mask,
            )

    def fused_verifier_block_attention_triton(
        q: torch.Tensor,
        k_uq_hist: torch.Tensor,
        k_lq_hist: torch.Tensor,
        k_scale_u: torch.Tensor,
        k_zp_u: torch.Tensor,
        k_scale_l: torch.Tensor,
        k_zp_l: torch.Tensor,
        v_uq_hist: torch.Tensor,
        v_lq_hist: torch.Tensor,
        v_scale_u: torch.Tensor,
        v_zp_u: torch.Tensor,
        v_scale_l: torch.Tensor,
        v_zp_l: torch.Tensor,
        k_recent: torch.Tensor,
        v_recent: torch.Tensor,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
        *,
        group_size_k: int,
        group_size_v: int,
        block_d: int | None = None,
        num_warps: int | None = None,
    ) -> torch.Tensor:
        """CUDA Triton fused verifier; ``q`` is ``[1, H, γ, D]``.

        Defaults from :func:`kv_kernels.tuning.active_kernel_tuning` (verifier ``q_len=γ`` profile).
        """
        require_triton()
        if q.device.type != "cuda":
            raise ValueError("CUDA required for fused_verifier_block_attention_triton")
        if q.shape[0] != 1:
            raise ValueError("q must be [1, H, γ, D] for Triton fused verifier")
        tun = active_kernel_tuning()
        bd = int(block_d) if block_d is not None else int(tun.verifier_block_d)
        nw = int(num_warps) if num_warps is not None else int(tun.verifier_num_warps)
        b, h, gamma, d = q.shape
        _ = b
        s_hist = int(k_uq_hist.shape[1])
        s_rec = int(k_recent.shape[1])
        p = s_hist + s_rec
        l_total = p + gamma
        _ = int(v_scale_u.shape[1])
        scale_attn = float(d**-0.5)

        q32 = q.contiguous().to(torch.float32)
        out = torch.empty(1, h, gamma, d, device=q.device, dtype=torch.float32)

        ku = k_uq_hist.contiguous()
        kl = k_lq_hist.contiguous()
        ksu = k_scale_u.contiguous().to(torch.float32)
        kzu = k_zp_u.contiguous().to(torch.float32)
        ksl = k_scale_l.contiguous().to(torch.float32)
        kzl = k_zp_l.contiguous().to(torch.float32)
        vu = v_uq_hist.contiguous()
        vl = v_lq_hist.contiguous()
        vsu = v_scale_u.contiguous().to(torch.float32)
        vzu = v_zp_u.contiguous().to(torch.float32)
        vsl = v_scale_l.contiguous().to(torch.float32)
        vzl = v_zp_l.contiguous().to(torch.float32)
        kr = k_recent.contiguous().to(torch.float32)
        vr = v_recent.contiguous().to(torch.float32)
        kb = k_block.contiguous().to(torch.float32)
        vb = v_block.contiguous().to(torch.float32)

        stride_q_h = q32.stride(1)
        stride_q_t = q32.stride(2)
        stride_q_d = q32.stride(3)
        stride_kh_h = ku.stride(0)
        stride_kh_s = ku.stride(1)
        stride_kh_d = ku.stride(2)
        stride_ksu_h = ksu.stride(0)
        stride_ksu_s = ksu.stride(1)
        stride_ksu_g = ksu.stride(2)
        stride_vs_h = vsu.stride(0)
        stride_vs_tg = vsu.stride(1)
        stride_vs_d = vsu.stride(2)
        stride_kr_h = kr.stride(0)
        stride_kr_s = kr.stride(1)
        stride_kr_d = kr.stride(2)
        stride_kb_h = kb.stride(0)
        stride_kb_t = kb.stride(1)
        stride_kb_d = kb.stride(2)
        stride_out_h = out.stride(1)
        stride_out_t = out.stride(2)
        stride_out_d = out.stride(3)

        _fused_verifier_block_kernel[(h, gamma)](
            q32,
            out,
            ku,
            kl,
            ksu,
            kzu,
            ksl,
            kzl,
            vu,
            vl,
            vsu,
            vzu,
            vsl,
            vzl,
            kr,
            vr,
            kb,
            vb,
            S_HIST=s_hist,
            S_REC=s_rec,
            GAMMA=gamma,
            D=d,
            GS_K=group_size_k,
            GS_V=group_size_v,
            P=p,
            L=l_total,
            scale_attn=scale_attn,
            stride_q_h=stride_q_h,
            stride_q_t=stride_q_t,
            stride_q_d=stride_q_d,
            stride_kh_h=stride_kh_h,
            stride_kh_s=stride_kh_s,
            stride_kh_d=stride_kh_d,
            stride_ksu_h=stride_ksu_h,
            stride_ksu_s=stride_ksu_s,
            stride_ksu_g=stride_ksu_g,
            stride_vs_h=stride_vs_h,
            stride_vs_tg=stride_vs_tg,
            stride_vs_d=stride_vs_d,
            stride_kr_h=stride_kr_h,
            stride_kr_s=stride_kr_s,
            stride_kr_d=stride_kr_d,
            stride_kb_h=stride_kb_h,
            stride_kb_t=stride_kb_t,
            stride_kb_d=stride_kb_d,
            stride_out_h=stride_out_h,
            stride_out_t=stride_out_t,
            stride_out_d=stride_out_d,
            BLOCK_D=bd,
            num_warps=nw,
        )
        return out.to(q.dtype)

else:

    def fused_verifier_block_attention_triton(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("Triton not available")


def fused_verifier_block_attention(
    q: torch.Tensor,
    k_uq_hist: torch.Tensor,
    k_lq_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    k_scale_l: torch.Tensor,
    k_zp_l: torch.Tensor,
    v_uq_hist: torch.Tensor,
    v_lq_hist: torch.Tensor,
    v_scale_u: torch.Tensor,
    v_zp_u: torch.Tensor,
    v_scale_l: torch.Tensor,
    v_zp_l: torch.Tensor,
    k_recent: torch.Tensor,
    v_recent: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    *,
    group_size_k: int,
    group_size_v: int,
    backend: str = "ref",
) -> torch.Tensor:
    """Dispatch ``ref`` (PyTorch) vs ``triton``."""
    if backend == "ref":
        return fused_verifier_block_attention_reference(
            q,
            k_uq_hist,
            k_lq_hist,
            k_scale_u,
            k_zp_u,
            k_scale_l,
            k_zp_l,
            v_uq_hist,
            v_lq_hist,
            v_scale_u,
            v_zp_u,
            v_scale_l,
            v_zp_l,
            k_recent,
            v_recent,
            k_block,
            v_block,
            group_size_k=group_size_k,
            group_size_v=group_size_v,
        )
    if backend == "triton":
        return fused_verifier_block_attention_triton(
            q,
            k_uq_hist,
            k_lq_hist,
            k_scale_u,
            k_zp_u,
            k_scale_l,
            k_zp_l,
            v_uq_hist,
            v_lq_hist,
            v_scale_u,
            v_zp_u,
            v_scale_l,
            v_zp_l,
            k_recent,
            v_recent,
            k_block,
            v_block,
            group_size_k=group_size_k,
            group_size_v=group_size_v,
        )
    raise ValueError(f"unknown backend {backend!r}")
