"""Tests for :class:`cache.hierarchical_kv_store.HierarchicalKVStore`."""

from __future__ import annotations

import torch

from cache.hierarchical_kv_store import HierarchicalKVStore
from cache.quant_spec_kv import (
    quantize_fp16_kv_to_upper_lower,
    reconstruct_key_target,
    reconstruct_value_target,
)


def _rand_kv(b: int, h: int, s: int, d: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(b, h, s, d, device=device, dtype=dtype)
    v = torch.randn(b, h, s, d, device=device, dtype=dtype)
    return k, v


def test_prefill_splits_history_and_cf1() -> None:
    B, H, D, L = 1, 2, 8, 3
    G = 4
    S = 20
    cap = 2 * G
    assert S > cap
    old = S - min(S, cap)
    assert old == 12

    store = HierarchicalKVStore(num_layers=L, num_heads=H, head_dim=D, batch_size=B, G=G, device=torch.device("cpu"))
    layers_k = []
    layers_v = []
    for _ in range(L):
        k, v = _rand_kv(B, H, S, D, torch.device("cpu"), torch.float16)
        layers_k.append(k)
        layers_v.append(v)

    store.prefill_from_fp16(layers_k, layers_v, recent_tokens_cap=cap)
    assert store.hist_len == old
    assert store.cf1_len == min(S, cap)
    assert store.cf2_len == 0
    assert store.logical_committed_seq_len() == S


def test_draft_view_omits_lower_residual() -> None:
    B, H, D, L = 1, 1, 4, 1
    store = HierarchicalKVStore(num_layers=L, num_heads=H, head_dim=D, batch_size=B, G=8, device=torch.device("cpu"))
    S = 16
    k, v = _rand_kv(B, H, S, D, torch.device("cpu"), torch.float16)
    store.prefill_from_fp16([k], [v], recent_tokens_cap=8)

    dv = store.draft_view()
    tv = store.target_view()
    assert dv.upper_only is True
    assert tv.upper_only is False
    # Same total seq length along time axis
    assert dv.layers_k[0].shape[2] == tv.layers_k[0].shape[2]


def test_cf2_append_and_clear_does_not_touch_hist() -> None:
    B, H, D, L = 1, 1, 4, 1
    store = HierarchicalKVStore(num_layers=L, num_heads=H, head_dim=D, batch_size=B, G=8, device=torch.device("cpu"))
    S = 24
    k, v = _rand_kv(B, H, S, D, torch.device("cpu"), torch.float16)
    store.prefill_from_fp16([k], [v], recent_tokens_cap=8)
    h_before = store.hist_len
    c1_before = store.cf1_len

    nk, nv = _rand_kv(B, H, 2, D, torch.device("cpu"), torch.float16)
    store.append_cf2_fp16([nk], [nv])
    assert store.cf2_len == 2
    assert store.hist_len == h_before
    assert store.cf1_len == c1_before

    store.clear_cf2()
    assert store.cf2_len == 0
    assert store.hist_len == h_before


def test_commit_cf2_prefix_moves_to_cf1() -> None:
    B, H, D, L = 1, 1, 4, 1
    store = HierarchicalKVStore(num_layers=L, num_heads=H, head_dim=D, batch_size=B, G=8, device=torch.device("cpu"))
    S = 10
    k, v = _rand_kv(B, H, S, D, torch.device("cpu"), torch.float16)
    store.prefill_from_fp16([k], [v], recent_tokens_cap=4)
    nk, nv = _rand_kv(B, H, 5, D, torch.device("cpu"), torch.float16)
    store.append_cf2_fp16([nk], [nv])
    assert store.cf2_len == 5
    c1_before = store.cf1_len
    store.commit_cf2_prefix_to_cf1(3)
    assert store.cf2_len == 2
    assert store.cf1_len == c1_before + 3


def test_rollover_moves_cf1_to_hist_and_cf2_to_cf1() -> None:
    B, H, D, L = 1, 1, 4, 1
    store = HierarchicalKVStore(num_layers=L, num_heads=H, head_dim=D, batch_size=B, G=2, device=torch.device("cpu"))
    S = 8
    k, v = _rand_kv(B, H, S, D, torch.device("cpu"), torch.float16)
    store.prefill_from_fp16([k], [v], recent_tokens_cap=4)
    assert store.cf1_len == 4
    nk, nv = _rand_kv(B, H, 2, D, torch.device("cpu"), torch.float16)
    store.append_cf2_fp16([nk], [nv])
    assert store.cf2_len == 2

    hist0 = store.hist_len
    store.rollover()
    assert store.hist_len == hist0 + 4
    assert store.cf1_len == 2
    assert store.cf2_len == 0


def test_quantize_roundtrip_small_error() -> None:
    gs = 8
    k = torch.randn(1, 1, 8, 8, dtype=torch.float16)
    v = torch.randn(1, 1, 8, 8, dtype=torch.float16)
    kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp = quantize_fp16_kv_to_upper_lower(
        k, v, group_size=gs
    )
    rk = reconstruct_key_target(kuq, kus, kuzp, klq, kls, klzp)
    rv = reconstruct_value_target(vuq, vus, vuzp, vlq, vls, vlzp, group_size=gs)
    assert (k.float() - rk.float()).abs().max().item() < 0.35
    assert (v.float() - rv.float()).abs().max().item() < 0.35
