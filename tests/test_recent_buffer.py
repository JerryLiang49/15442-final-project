"""Tests for :mod:`cache.recent_buffer` (CF1/CF2 manager + instrumentation)."""

from __future__ import annotations

import torch

from cache.hierarchical_kv_store import HierarchicalKVStore
from cache.recent_buffer import RecentBufferManager


def _rand_kv(b: int, h: int, s: int, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(b, h, s, d, dtype=torch.float16)
    v = torch.randn(b, h, s, d, dtype=torch.float16)
    return k, v


def _make_mgr(
    *,
    layers: int = 1,
    heads: int = 1,
    head_dim: int = 8,
    g: int = 4,
) -> RecentBufferManager:
    store = HierarchicalKVStore(
        num_layers=layers,
        num_heads=heads,
        head_dim=head_dim,
        batch_size=1,
        G=g,
        device=torch.device("cpu"),
    )
    return RecentBufferManager(store)


def test_prefill_fills_cf1_up_to_cap() -> None:
    """CF1 should hold min(S, cf1_max) after prefill_initialize (default cap)."""
    mgr = _make_mgr(g=4)
    cap = mgr.store.cf1_max_tokens  # 8
    S = 20
    k, v = _rand_kv(1, 1, S, 8)
    mgr.prefill_initialize([k], [v])
    assert mgr.store.hist_len == S - min(S, cap)
    assert mgr.store.cf1_len == min(S, cap)
    assert mgr.store.cf2_len == 0
    occ = mgr.occupancy()
    assert occ.cf1_len == min(S, cap)
    assert occ.logical_draft_seq_len == S


def test_rejection_at_different_positions_only_trims_cf2() -> None:
    mgr = _make_mgr(g=8)
    S = 12
    k, v = _rand_kv(1, 1, S, 8)
    mgr.prefill_initialize([k], [v], recent_tokens_cap=8)
    h0 = mgr.store.hist_len
    c1_0 = mgr.store.cf1_len

    nk, nv = _rand_kv(1, 1, 6, 8)
    mgr.append_draft([nk], [nv])
    assert mgr.store.cf2_len == 6

    # Full keep — no trim, no stats
    mgr.reject_speculative_suffix(6)
    assert mgr.stats.reject_trim_events == 0
    assert mgr.stats.rejected_tokens_total == 0

    # Trim to 1: drop 5 tokens
    mgr.reject_speculative_suffix(1)
    assert mgr.store.cf2_len == 1
    assert mgr.stats.reject_trim_events == 1
    assert mgr.stats.rejected_tokens_total == 5
    assert mgr.store.hist_len == h0
    assert mgr.store.cf1_len == c1_0

    # Clear rest (1 token)
    mgr.clear_speculative()
    assert mgr.store.cf2_len == 0
    assert mgr.stats.reject_trim_events == 2
    assert mgr.stats.rejected_tokens_total == 6


def test_rejection_at_zero_prefix() -> None:
    mgr = _make_mgr(g=4)
    k, v = _rand_kv(1, 1, 10, 8)
    mgr.prefill_initialize([k], [v], recent_tokens_cap=8)
    nk, nv = _rand_kv(1, 1, 4, 8)
    mgr.append_draft([nk], [nv])
    h, c1 = mgr.store.hist_len, mgr.store.cf1_len
    mgr.reject_speculative_suffix(0)
    assert mgr.store.cf2_len == 0
    assert mgr.store.hist_len == h
    assert mgr.store.cf1_len == c1
    assert mgr.stats.rejected_tokens_total == 4


def test_rollover_after_verification_round() -> None:
    """Accept part of CF2, then rollover: quantize expanded CF1, move remainder to CF1."""
    mgr = _make_mgr(g=2)  # cf1_max = 4
    S = 8
    k, v = _rand_kv(1, 1, S, 8)
    mgr.prefill_initialize([k], [v], recent_tokens_cap=4)
    assert mgr.store.cf1_len == 4
    assert mgr.store.hist_len == 4

    nk, nv = _rand_kv(1, 1, 5, 8)
    mgr.append_draft([nk], [nv])
    mgr.accept_verified_prefix(3)
    assert mgr.store.cf1_len == 7
    assert mgr.store.cf2_len == 2

    hist_before = mgr.store.hist_len
    mgr.rollover()
    assert mgr.stats.rollover_count == 1
    assert mgr.store.hist_len == hist_before + 7
    assert mgr.store.cf1_len == 2
    assert mgr.store.cf2_len == 0


def test_long_decode_multiple_rollovers() -> None:
    mgr = _make_mgr(g=2)
    k, v = _rand_kv(1, 1, 4, 8)
    mgr.prefill_initialize([k], [v], recent_tokens_cap=4)
    assert mgr.store.hist_len == 0
    assert mgr.store.cf1_len == 4

    for _ in range(5):
        t = _rand_kv(1, 1, 2, 8)
        mgr.append_draft([t[0]], [t[1]])
        assert mgr.store.cf2_len == 2
        mgr.rollover()

    # Rollover 0: hist += 4 (initial cf1), cf1 <- cf2 (2 tok). Then hist 4, cf1=2.
    # Next 4 rollovers: each hist += 2. Final hist = 4 + 4*2 = 12, cf1=2.
    assert mgr.stats.rollover_count == 5
    assert mgr.store.cf1_len == 2
    assert mgr.store.cf2_len == 0
    assert mgr.store.hist_len == 12


def test_instrumentation_dict() -> None:
    mgr = _make_mgr(g=4)
    k, v = _rand_kv(1, 1, 16, 8)
    mgr.prefill_initialize([k], [v])
    d = mgr.instrumentation_dict()
    assert "cf1_len" in d and "rollover_count" in d
    assert d["rollover_count"] == 0


def test_rollover_noop_does_not_increment_counter() -> None:
    mgr = _make_mgr(g=4)
    k, v = _rand_kv(1, 1, 8, 8)
    mgr.prefill_initialize([k], [v])
    mgr.rollover()
    assert mgr.stats.rollover_count == 1
    mgr.rollover()
    assert mgr.stats.rollover_count == 1
