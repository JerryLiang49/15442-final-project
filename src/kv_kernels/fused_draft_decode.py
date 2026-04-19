"""Fused draft decode attention: single query (``q_len=1``) over INT4 history + FP16 recent tail.

**Layout (must match tests + hierarchical store)**

* **Packed bytes** ``uint8`` shape ``[S, D]`` per head: two 4-bit codes per byte when ``D`` is even:
  ``byte = low_nibble | (high_nibble << 4)`` (see :func:`cache.quant_spec_kv.pack_int4_pair`).
* **Draft K** uses **only the high nibble** for dequantized key: ``hi = (byte >> 4) & 15``.
  Per-channel groups along ``D``: ``g = d // group_size_k``; scales ``k_scale_u[s, g]``, ``k_zp_u[s, g]``.
* **Draft V** uses **only the high nibble** (upper INT4 path). **Value** uses **token-wise** groups along
  ``S``: ``tg = s // group_size_v``; scales ``v_scale_u[tg, d]``, ``v_zp_u[tg, d]`` (shape ``[n_gv, D]``).
* **Recent** tail: FP16 ``[S_r, D]`` for K and V (same head).

**Attention**

* Logits ``l_i = scale_attn * dot(q, k_i)`` with ``scale_attn = head_dim ** -0.5`` (Llama-style).
* **Online softmax** over **concat**(hist, recent) positions (two-segment safe max / sum-exp).
* Output ``o = sum_i softmax_i * v_i`` (two-pass: stats, then weighted V; logits/V recomputed — no full
  dequant KV buffer in device memory for Triton path).

**Scope**

* ``B = 1``, arbitrary ``H``, ``q_len = 1`` (primary supported mode).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .triton_runtime import require_triton, triton_available
from .tuning import active_kernel_tuning

# ---------------------------------------------------------------------------
# Reference (high-precision path; may materialize dequant for parity testing)
# ---------------------------------------------------------------------------


def _dequant_k_draft_row(
    packed_row: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    *,
    group_size_k: int,
) -> torch.Tensor:
    """One row ``[D]`` uint8 packed -> FP32 key row (draft upper nibble only)."""
    d = int(packed_row.shape[0])
    hi = ((packed_row.to(torch.int32) >> 4) & 15).to(torch.float32)
    n_g = d // group_size_k
    g = torch.arange(d, device=packed_row.device, dtype=torch.int64) // group_size_k
    su = k_scale_u[g].to(torch.float32)
    zu = k_zp_u[g].to(torch.float32)
    return hi * su + zu


def _dequant_v_draft_row(
    packed_row: torch.Tensor,
    s: int,
    v_scale_u: torch.Tensor,
    v_zp_u: torch.Tensor,
    *,
    group_size_v: int,
) -> torch.Tensor:
    """Position index ``s`` along history; ``v_scale_u`` is ``[n_gv, D]``."""
    d = int(packed_row.shape[0])
    hi = ((packed_row.to(torch.int32) >> 4) & 15).to(torch.float32)
    tg = s // group_size_v
    su = v_scale_u[tg].to(torch.float32)
    zu = v_zp_u[tg].to(torch.float32)
    return hi * su + zu


def fused_draft_decode_attention_reference(
    q: torch.Tensor,
    packed_k_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    packed_v_hist: torch.Tensor,
    v_scale_u: torch.Tensor,
    v_zp_u: torch.Tensor,
    k_recent: torch.Tensor,
    v_recent: torch.Tensor,
    *,
    group_size_k: int,
    group_size_v: int,
    scale_attn: float | None = None,
) -> torch.Tensor:
    """Reference attention output ``[1, H, 1, D]`` (materializes dequant rows in Python).

    Args:
        q: ``[1, H, 1, D]`` float32/float16.
        packed_k_hist / packed_v_hist: ``[H, S_h, D]`` uint8.
        k_scale_u / k_zp_u: ``[H, S_h, n_gk]``.
        v_scale_u / v_zp_u: ``[H, n_gv, D]`` (token groups for values).
        k_recent / v_recent: ``[H, S_r, D]`` float (FP16/FP32).
    """
    if q.dim() != 4 or q.shape[0] != 1 or q.shape[2] != 1:
        raise ValueError("q must be [1, H, 1, D]")
    b, h, one, d = q.shape
    _ = (b, one)
    if scale_attn is None:
        scale_attn = float(d**-0.5)
    qh = q[0, :, 0, :].to(torch.float32)  # [H, D]

    out = torch.zeros(h, d, device=q.device, dtype=torch.float32)

    for head in range(h):
        qv = qh[head]
        s_hist = int(packed_k_hist.shape[1])
        s_rec = int(k_recent.shape[1])

        logits: list[torch.Tensor] = []
        v_rows: list[torch.Tensor] = []

        for s in range(s_hist):
            kr = packed_k_hist[head, s]
            k_hat = _dequant_k_draft_row(kr, k_scale_u[head, s], k_zp_u[head, s], group_size_k=group_size_k)
            logits.append((qv * k_hat).sum() * scale_attn)
            vr = packed_v_hist[head, s]
            v_hat = _dequant_v_draft_row(vr, s, v_scale_u[head], v_zp_u[head], group_size_v=group_size_v)
            v_rows.append(v_hat)

        for s in range(s_rec):
            k_hat = k_recent[head, s].to(torch.float32)
            logits.append((qv * k_hat).sum() * scale_attn)
            v_rows.append(v_recent[head, s].to(torch.float32))

        logits_t = torch.stack(logits, dim=0)
        p = torch.softmax(logits_t, dim=0)
        acc = torch.zeros(d, device=q.device, dtype=torch.float32)
        for i in range(len(v_rows)):
            acc = acc + p[i] * v_rows[i]
        out[head] = acc

    return out.view(1, h, 1, d).to(q.dtype)


# ---------------------------------------------------------------------------
# Online softmax utilities (match two-pass fused kernel)
# ---------------------------------------------------------------------------


def _online_softmax_stats_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global ``m``, ``l`` for softmax over 1D logits."""
    m = logits.max()
    l = torch.exp(logits - m).sum()
    return m, l


def fused_draft_decode_attention_reference_two_pass(
    q: torch.Tensor,
    packed_k_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    packed_v_hist: torch.Tensor,
    v_scale_u: torch.Tensor,
    v_zp_u: torch.Tensor,
    k_recent: torch.Tensor,
    v_recent: torch.Tensor,
    *,
    group_size_k: int,
    group_size_v: int,
    scale_attn: float | None = None,
) -> torch.Tensor:
    """Same math as :func:`fused_draft_decode_attention_reference` but uses explicit two-pass structure."""
    if scale_attn is None:
        scale_attn = float(q.shape[-1] ** -0.5)
    b, h, one, d = q.shape
    _ = (b, one)
    qh = q[0, :, 0, :].to(torch.float32)
    out = torch.zeros(h, d, device=q.device, dtype=torch.float32)

    for head in range(h):
        qv = qh[head]
        s_hist = int(packed_k_hist.shape[1])
        s_rec = int(k_recent.shape[1])
        log_chunks: list[torch.Tensor] = []
        for s in range(s_hist):
            kr = packed_k_hist[head, s]
            k_hat = _dequant_k_draft_row(kr, k_scale_u[head, s], k_zp_u[head, s], group_size_k=group_size_k)
            log_chunks.append((qv * k_hat).sum() * scale_attn)
        for s in range(s_rec):
            k_hat = k_recent[head, s].to(torch.float32)
            log_chunks.append((qv * k_hat).sum() * scale_attn)
        logits_t = torch.stack(log_chunks)
        m, l = _online_softmax_stats_from_logits(logits_t)
        acc = torch.zeros(d, device=q.device, dtype=torch.float32)
        idx = 0
        for s in range(s_hist):
            p = torch.exp(logits_t[idx] - m) / l
            vr = packed_v_hist[head, s]
            v_hat = _dequant_v_draft_row(vr, s, v_scale_u[head], v_zp_u[head], group_size_v=group_size_v)
            acc = acc + p * v_hat
            idx += 1
        for s in range(s_rec):
            p = torch.exp(logits_t[idx] - m) / l
            acc = acc + p * v_recent[head, s].to(torch.float32)
            idx += 1
        out[head] = acc

    return out.view(1, h, 1, d).to(q.dtype)


# ---------------------------------------------------------------------------
# Triton fused kernel (B=1, q_len=1)
# ---------------------------------------------------------------------------

if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_draft_decode_kernel(
        q_ptr,
        out_ptr,
        packed_k_ptr,
        k_su_ptr,
        k_zu_ptr,
        packed_v_ptr,
        v_su_ptr,
        v_zu_ptr,
        k_rec_ptr,
        v_rec_ptr,
        S_hist: tl.constexpr,
        S_rec: tl.constexpr,
        D: tl.constexpr,
        GS_K: tl.constexpr,
        GS_V: tl.constexpr,
        N_GV: tl.constexpr,
        scale_attn: tl.constexpr,
        stride_q_h: tl.constexpr,
        stride_q_d: tl.constexpr,
        stride_pk_s: tl.constexpr,
        stride_pk_d: tl.constexpr,
        stride_ksu_s: tl.constexpr,
        stride_ksu_g: tl.constexpr,
        stride_vs_tg: tl.constexpr,
        stride_vs_d: tl.constexpr,
        stride_kr_s: tl.constexpr,
        stride_kr_d: tl.constexpr,
        stride_out_h: tl.constexpr,
        stride_out_d: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per head; two passes over hist+recent (recompute for output)."""
        h = tl.program_id(0)

        # --- Pass 1: online softmax stats (m, l) ---
        m_acc = -1.0e30
        l_acc = 0.0

        # Historical segment
        for s in range(S_hist):
            dot_k = 0.0
            for d0 in range(0, D, BLOCK_D):
                offs = d0 + tl.arange(0, BLOCK_D)
                mask = offs < D
                qv = tl.load(q_ptr + h * stride_q_h + offs * stride_q_d, mask=mask, other=0.0).to(tl.float32)
                b = tl.load(
                    packed_k_ptr + h * (S_hist * stride_pk_s) + s * stride_pk_s + offs * stride_pk_d,
                    mask=mask,
                    other=0,
                )
                hi = ((b.to(tl.int32) >> 4) & 15).to(tl.float32)
                g = offs // GS_K
                su = tl.load(
                    k_su_ptr + h * (S_hist * stride_ksu_s) + s * stride_ksu_s + g * stride_ksu_g,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                zu = tl.load(
                    k_zu_ptr + h * (S_hist * stride_ksu_s) + s * stride_ksu_s + g * stride_ksu_g,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                kv = hi * su + zu
                dot_k += tl.sum(tl.where(mask, qv * kv, 0.0))
            logit = dot_k * scale_attn
            m_new = tl.maximum(m_acc, logit)
            l_acc = l_acc * tl.exp(m_acc - m_new) + tl.exp(logit - m_new)
            m_acc = m_new

        # Recent FP16 segment
        for s in range(S_rec):
            dot_k = 0.0
            for d0 in range(0, D, BLOCK_D):
                offs = d0 + tl.arange(0, BLOCK_D)
                mask = offs < D
                qv = tl.load(q_ptr + h * stride_q_h + offs * stride_q_d, mask=mask, other=0.0).to(tl.float32)
                kv = tl.load(
                    k_rec_ptr + h * (S_rec * stride_kr_s) + s * stride_kr_s + offs * stride_kr_d,
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

        # --- Pass 2: weighted V (recompute logits; dequant V from packed_v at this d-block only) ---
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)

            for s in range(S_hist):
                logit = 0.0
                for dd in range(0, D, BLOCK_D):
                    o2 = dd + tl.arange(0, BLOCK_D)
                    m2 = o2 < D
                    q2 = tl.load(q_ptr + h * stride_q_h + o2 * stride_q_d, mask=m2, other=0.0).to(tl.float32)
                    b2 = tl.load(
                        packed_k_ptr + h * (S_hist * stride_pk_s) + s * stride_pk_s + o2 * stride_pk_d,
                        mask=m2,
                        other=0,
                    )
                    hi2 = ((b2.to(tl.int32) >> 4) & 15).to(tl.float32)
                    g2 = o2 // GS_K
                    su2 = tl.load(
                        k_su_ptr + h * (S_hist * stride_ksu_s) + s * stride_ksu_s + g2 * stride_ksu_g,
                        mask=m2,
                        other=0.0,
                    ).to(tl.float32)
                    zu2 = tl.load(
                        k_zu_ptr + h * (S_hist * stride_ksu_s) + s * stride_ksu_s + g2 * stride_ksu_g,
                        mask=m2,
                        other=0.0,
                    ).to(tl.float32)
                    kv2 = hi2 * su2 + zu2
                    logit += tl.sum(tl.where(m2, q2 * kv2, 0.0))
                logit = logit * scale_attn
                w = tl.exp(logit - m_final) / l_final

                bv = tl.load(
                    packed_v_ptr + h * (S_hist * stride_pk_s) + s * stride_pk_s + offs * stride_pk_d,
                    mask=mask,
                    other=0,
                )
                hi_v = ((bv.to(tl.int32) >> 4) & 15).to(tl.float32)
                tg = s // GS_V
                sv = tl.load(
                    v_su_ptr + h * (N_GV * stride_vs_tg) + tg * stride_vs_tg + offs * stride_vs_d,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                zv = tl.load(
                    v_zu_ptr + h * (N_GV * stride_vs_tg) + tg * stride_vs_tg + offs * stride_vs_d,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                vv = hi_v * sv + zv
                acc += w * vv

            for s in range(S_rec):
                logit = 0.0
                for dd in range(0, D, BLOCK_D):
                    o2 = dd + tl.arange(0, BLOCK_D)
                    m2 = o2 < D
                    q2 = tl.load(q_ptr + h * stride_q_h + o2 * stride_q_d, mask=m2, other=0.0).to(tl.float32)
                    kv2 = tl.load(
                        k_rec_ptr + h * (S_rec * stride_kr_s) + s * stride_kr_s + o2 * stride_kr_d,
                        mask=m2,
                        other=0.0,
                    ).to(tl.float32)
                    logit += tl.sum(tl.where(m2, q2 * kv2, 0.0))
                logit = logit * scale_attn
                w = tl.exp(logit - m_final) / l_final
                vv = tl.load(
                    v_rec_ptr + h * (S_rec * stride_kr_s) + s * stride_kr_s + offs * stride_kr_d,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                acc += w * vv

            tl.store(out_ptr + h * stride_out_h + offs * stride_out_d, acc, mask=mask)

    def fused_draft_decode_attention_triton(
        q: torch.Tensor,
        packed_k_hist: torch.Tensor,
        k_scale_u: torch.Tensor,
        k_zp_u: torch.Tensor,
        packed_v_hist: torch.Tensor,
        v_scale_u: torch.Tensor,
        v_zp_u: torch.Tensor,
        k_recent: torch.Tensor,
        v_recent: torch.Tensor,
        *,
        group_size_k: int,
        group_size_v: int,
        block_d: int | None = None,
        num_warps: int | None = None,
    ) -> torch.Tensor:
        """CUDA Triton fused draft decode; ``q`` is ``[1, H, 1, D]``.

        ``block_d`` / ``num_warps`` default from :func:`kv_kernels.tuning.active_kernel_tuning`
        (Phase O — draft ``q_len=1`` profile).
        """
        require_triton()
        if q.device.type != "cuda":
            raise ValueError("CUDA required for fused_draft_decode_attention_triton")
        if q.shape[0] != 1 or q.shape[2] != 1:
            raise ValueError("q must be [1, H, 1, D]")
        tun = active_kernel_tuning()
        bd = int(block_d) if block_d is not None else int(tun.draft_block_d)
        nw = int(num_warps) if num_warps is not None else int(tun.draft_num_warps)
        b, h, _, d = q.shape
        _ = b
        s_hist = int(packed_k_hist.shape[1])
        s_rec = int(k_recent.shape[1])
        n_gk = d // group_size_k
        n_gv = int(v_scale_u.shape[1])  # [H, n_gv, D]
        scale_attn = float(d**-0.5)

        q32 = q.contiguous().to(torch.float32)
        pk = packed_k_hist.contiguous()
        ks = k_scale_u.contiguous().to(torch.float32)
        kz = k_zp_u.contiguous().to(torch.float32)
        pv = packed_v_hist.contiguous()
        vs = v_scale_u.contiguous().to(torch.float32)
        vz = v_zp_u.contiguous().to(torch.float32)
        kr = k_recent.contiguous().to(torch.float32)
        vr = v_recent.contiguous().to(torch.float32)

        out = torch.empty(1, h, 1, d, device=q.device, dtype=torch.float32)

        # strides (elements) for Triton pointers
        stride_q_h = q32.stride(1)
        stride_q_d = q32.stride(3)
        stride_out_h = out.stride(1)
        stride_out_d = out.stride(3)
        stride_pk_s = pk.stride(1)
        stride_pk_d = pk.stride(2)
        stride_ksu_s = ks.stride(1)
        stride_ksu_g = ks.stride(2)
        stride_vs_tg = vs.stride(1)
        stride_vs_d = vs.stride(2)
        stride_kr_s = kr.stride(1)
        stride_kr_d = kr.stride(2)

        _fused_draft_decode_kernel[(h,)](
            q32,
            out,
            pk,
            ks,
            kz,
            pv,
            vs,
            vz,
            kr,
            vr,
            S_hist=s_hist,
            S_rec=s_rec,
            D=d,
            GS_K=group_size_k,
            GS_V=group_size_v,
            N_GV=n_gv,
            scale_attn=scale_attn,
            stride_q_h=stride_q_h,
            stride_q_d=stride_q_d,
            stride_pk_s=stride_pk_s,
            stride_pk_d=stride_pk_d,
            stride_ksu_s=stride_ksu_s,
            stride_ksu_g=stride_ksu_g,
            stride_vs_tg=stride_vs_tg,
            stride_vs_d=stride_vs_d,
            stride_kr_s=stride_kr_s,
            stride_kr_d=stride_kr_d,
            stride_out_h=stride_out_h,
            stride_out_d=stride_out_d,
            BLOCK_D=bd,
            num_warps=nw,
        )
        return out.to(q.dtype)

else:

    def fused_draft_decode_attention_triton(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("Triton not available")


def fused_draft_decode_attention(
    q: torch.Tensor,
    packed_k_hist: torch.Tensor,
    k_scale_u: torch.Tensor,
    k_zp_u: torch.Tensor,
    packed_v_hist: torch.Tensor,
    v_scale_u: torch.Tensor,
    v_zp_u: torch.Tensor,
    k_recent: torch.Tensor,
    v_recent: torch.Tensor,
    *,
    group_size_k: int,
    group_size_v: int,
    backend: str = "ref",
) -> torch.Tensor:
    """Dispatch ``ref`` (PyTorch) vs ``triton``."""
    if backend == "ref":
        return fused_draft_decode_attention_reference(
            q,
            packed_k_hist,
            k_scale_u,
            k_zp_u,
            packed_v_hist,
            v_scale_u,
            v_zp_u,
            k_recent,
            v_recent,
            group_size_k=group_size_k,
            group_size_v=group_size_v,
        )
    if backend == "triton":
        return fused_draft_decode_attention_triton(
            q,
            packed_k_hist,
            k_scale_u,
            k_zp_u,
            packed_v_hist,
            v_scale_u,
            v_zp_u,
            k_recent,
            v_recent,
            group_size_k=group_size_k,
            group_size_v=group_size_v,
        )
    raise ValueError(f"unknown backend {backend!r}")
