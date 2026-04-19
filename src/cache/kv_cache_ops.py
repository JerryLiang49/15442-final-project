"""Shared KV cache operations: quantize FP16 CF1 and append to INT4 history (reference math).

Used by :meth:`cache.hierarchical_kv_store.HierarchicalKVStore.rollover` for a single code path and tests.
**Pack** here means asymmetric INT4 quantization (upper + lower) matching :mod:`cache.quant_spec_kv`, not
necessarily ``uint8`` nibble packing — the store keeps int8 codes per position.
"""

from __future__ import annotations

import torch

from .quant_spec_kv import quantize_fp16_kv_to_upper_lower

from .hierarchical_kv_store import _pad_seq_to_multiple, append_hist_key, append_hist_value


def quantize_cf1_fp16_to_int4(
    cf1_k: torch.Tensor,
    cf1_v: torch.Tensor,
    cf1_n: int,
    gs: int,
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
    """Quantize ``cf1_n`` tokens of FP16 CF1 K/V to upper/lower INT4 + metadata (same as rollover)."""
    assert cf1_k is not None and cf1_v is not None
    ck, cv, _ = _pad_seq_to_multiple(cf1_k, cf1_v, gs)
    (
        kuq,
        kus,
        kuzp,
        klq,
        kls,
        klzp,
        vuq,
        vus,
        vuzp,
        vlq,
        vls,
        vlzp,
    ) = quantize_fp16_kv_to_upper_lower(ck, cv, group_size=gs)
    if kuq.shape[2] > cf1_n:
        n_tg = (cf1_n + gs - 1) // gs
        v_len = n_tg * gs
        kuq = kuq[:, :, :cf1_n, :]
        klq = klq[:, :, :cf1_n, :]
        vuq = vuq[:, :, :v_len, :]
        vlq = vlq[:, :, :v_len, :]
        kus = kus[:, :, :cf1_n, :]
        kuzp = kuzp[:, :, :cf1_n, :]
        kls = kls[:, :, :cf1_n, :]
        klzp = klzp[:, :, :cf1_n, :]
        vus = vus[:, :, :n_tg, :]
        vuzp = vuzp[:, :, :n_tg, :]
        vls = vls[:, :, :n_tg, :]
        vlzp = vlzp[:, :, :n_tg, :]
    return kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp


def append_quantized_cf1_to_hist(
    hist_len: int,
    upper_k: torch.Tensor,
    upper_k_scale: torch.Tensor,
    upper_k_zp: torch.Tensor,
    lower_k: torch.Tensor,
    lower_k_scale: torch.Tensor,
    lower_k_zp: torch.Tensor,
    upper_v: torch.Tensor,
    upper_v_scale: torch.Tensor,
    upper_v_zp: torch.Tensor,
    lower_v: torch.Tensor,
    lower_v_scale: torch.Tensor,
    lower_v_zp: torch.Tensor,
    kuq: torch.Tensor,
    kus: torch.Tensor,
    kuzp: torch.Tensor,
    klq: torch.Tensor,
    kls: torch.Tensor,
    klzp: torch.Tensor,
    vuq: torch.Tensor,
    vus: torch.Tensor,
    vuzp: torch.Tensor,
    vlq: torch.Tensor,
    vls: torch.Tensor,
    vlzp: torch.Tensor,
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
    int,
]:
    """Concatenate quantized CF1 chunk onto history tensors; return new tensors and ``hist_len + T``."""
    t = int(kuq.shape[2])
    if hist_len == 0:
        return (
            kuq,
            kus,
            kuzp,
            klq,
            kls,
            klzp,
            vuq,
            vus,
            vuzp,
            vlq,
            vls,
            vlzp,
            t,
        )
    uk, us, uz, lk, ls, lz = append_hist_key(
        upper_k,
        upper_k_scale,
        upper_k_zp,
        lower_k,
        lower_k_scale,
        lower_k_zp,
        kuq,
        kus,
        kuzp,
        klq,
        kls,
        klzp,
    )
    uv, vs, vz, lv, lss, lzz = append_hist_value(
        upper_v,
        upper_v_scale,
        upper_v_zp,
        lower_v,
        lower_v_scale,
        lower_v_zp,
        vuq,
        vus,
        vuzp,
        vlq,
        vls,
        vlzp,
    )
    return uk, us, uz, lk, ls, lz, uv, vs, vz, lv, lss, lzz, hist_len + t
