"""Phase L: fast CF2 path vs legacy store parity (same logical KV state)."""

from __future__ import annotations

import torch

from cache.hierarchical_kv_store import HierarchicalKVStore


def _make_store(*, fast: bool, device: torch.device) -> HierarchicalKVStore:
    return HierarchicalKVStore(
        num_layers=1,
        num_heads=2,
        head_dim=8,
        batch_size=1,
        G=4,
        quant_group_size=8,
        device=device,
        dtype=torch.float16,
        use_fast_cf2=fast,
    )


def _run_scenario(store: HierarchicalKVStore) -> dict[str, torch.Tensor | int]:
    """Append, trim, commit, rollover; return fingerprint tensors + lengths."""
    d = store.head_dim
    h = store.num_heads
    b = store.batch_size

    def kv(toks: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.randn(b, h, toks, d, device=store.device, dtype=store.dtype)
        v = torch.randn(b, h, toks, d, device=store.device, dtype=store.dtype)
        return k, v

    k0, v0 = kv(6)
    store.prefill_from_fp16([k0], [v0], recent_tokens_cap=4)
    k1, v1 = kv(2)
    store.append_cf2_fp16([k1], [v1])
    store.trim_cf2(1)
    store.commit_cf2_prefix_to_cf1(1)
    store.rollover()

    cl = store.cf1_len
    c1 = store._cf1_k[0]
    return {
        "hist_len": store.hist_len,
        "cf1_len": cl,
        "cf2_len": store.cf2_len,
        "uk0": store._upper_k[0].clone(),
        "cf1k0": c1[:, :, :cl, :].clone() if c1 is not None and cl > 0 else None,
    }


def test_fast_matches_legacy_cpu() -> None:
    dev = torch.device("cpu")
    torch.manual_seed(0)
    a = _make_store(fast=True, device=dev)
    torch.manual_seed(0)
    b = _make_store(fast=False, device=dev)
    fa = _run_scenario(a)
    torch.manual_seed(0)
    fb = _run_scenario(b)
    assert fa["hist_len"] == fb["hist_len"]
    assert fa["cf1_len"] == fb["cf1_len"]
    assert fa["cf2_len"] == fb["cf2_len"]
    assert torch.equal(fa["uk0"], fb["uk0"])
    if fa["cf1k0"] is None:
        assert fb["cf1k0"] is None
    else:
        assert fb["cf1k0"] is not None
        assert torch.allclose(fa["cf1k0"], fb["cf1k0"])


def test_mutation_profile_records() -> None:
    dev = torch.device("cpu")
    st = HierarchicalKVStore(
        num_layers=1,
        num_heads=1,
        head_dim=8,
        G=4,
        quant_group_size=8,
        device=dev,
        enable_mutation_profiling=True,
    )
    assert st.mutation_profile is not None
    d = st.head_dim
    k = torch.randn(1, 1, 4, d, device=dev, dtype=torch.float16)
    v = torch.randn(1, 1, 4, d, device=dev, dtype=torch.float16)
    st.prefill_from_fp16([k], [v], recent_tokens_cap=4)
    st.append_cf2_fp16([k[:, :, :1, :]], [v[:, :, :1, :]])
    assert st.mutation_profile.n_append >= 1
    st.rollover()
    assert st.mutation_profile.n_rollover >= 1
    assert st.mutation_profile.n_pack >= 1
