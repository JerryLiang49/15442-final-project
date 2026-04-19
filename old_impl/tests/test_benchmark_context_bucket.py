"""Phase 8: context bucketing and MT-Bench JSON loader (no model)."""

from __future__ import annotations

from pathlib import Path

from mlsys_kv.benchmarks.context_buckets import classify_context_bucket
from mlsys_kv.benchmarks.experiment_runner import (
    build_sparse_config_for_prompt,
    resolve_speculative_mode,
)
from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.datasets.mt_bench import load_mt_bench_subset


def test_load_mt_bench_subset_minimal() -> None:
    root = Path(__file__).resolve().parents[1]
    subset = root / "data" / "mt_bench_subset.json"
    prompts = load_mt_bench_subset(subset)
    assert len(prompts) >= 1
    assert all(p.text for p in prompts)


def test_classify_buckets() -> None:
    assert classify_context_bucket(10, short_max=64, medium_max=256).value == "short"
    assert classify_context_bucket(100, short_max=64, medium_max=256).value == "medium"
    assert classify_context_bucket(300, short_max=64, medium_max=256).value == "long"


def test_build_sparse_config_fraction() -> None:
    cfg = build_sparse_config_for_prompt(
        100,
        sparsity_budget=0.2,
        recent_window_fraction=0.1,
        refresh_interval=2,
        scoring="key_norm",
    )
    assert cfg.recent_window == 10
    assert cfg.heavy_hitter_budget >= 1


def test_resolve_speculative_mode_mapping() -> None:
    assert resolve_speculative_mode("quant_only", 8)[0] is DraftCacheMode.QUANT_ONLY
    assert resolve_speculative_mode("quant_only", 16)[0] is DraftCacheMode.FP16
    assert resolve_speculative_mode("sparse_quant", 8)[0] is DraftCacheMode.SPARSE_QUANT


def test_resolve_int4_maps_for_quant_paths() -> None:
    m, eff, lab, _ = resolve_speculative_mode("quant_only", 4)
    assert m is DraftCacheMode.QUANT_ONLY
    assert eff == 4
    assert lab == "quant_only"
    m2, eff2, lab2, _ = resolve_speculative_mode("sparse_quant", 4)
    assert m2 is DraftCacheMode.SPARSE_QUANT
    assert eff2 == 4
    assert lab2 == "sparse_quant"


def test_resolve_fp16_ignores_quant_bits_in_matrix() -> None:
    m, eff, _, note = resolve_speculative_mode("speculative_fp16", 4)
    assert m is DraftCacheMode.FP16
    assert eff == 16
    assert "not_used" in note
