"""Phase 16 analysis: semantics checks and report smoke (no heavy plots in CI)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlsys_kv.benchmarks.analysis.extended_tables import (
    table_effective_compression_ratio,
    table_joint_sparse_quant_vs_sparse_only,
    theoretical_spec_speedup_factor,
)
from mlsys_kv.benchmarks.analysis.failure_tables import table_where_acceptance_dropped
from mlsys_kv.benchmarks.analysis.loader import load_benchmark_csv
from mlsys_kv.benchmarks.analysis.report import generate_phase16_report
from mlsys_kv.benchmarks.analysis.semantics import assert_semantics_consistent, quantization_scope_summary


def test_semantics_consistent_on_sample_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "mode": "autoregressive",
                "benchmark_label": "ar",
                "quantization_type": "none",
                "quant_bits_effective": -1,
                "is_parallel_verification": False,
            },
            {
                "mode": "quant_only",
                "benchmark_label": "spec_quant_memonly",
                "quantization_type": "memory_only",
                "quant_bits_effective": 4,
                "is_parallel_verification": True,
            },
        ]
    )
    assert assert_semantics_consistent(df) == []


def test_scope_summary_memory_only_note() -> None:
    df = pd.DataFrame(
        [
            {"benchmark_label": "spec_quant_memonly", "quantization_type": "memory_only"},
        ]
    )
    s = quantization_scope_summary(df)
    assert s["has_runtime_accelerated_quant"] is False
    assert "memory-only" in s["caption_note"].lower() or "Memory-only" in s["caption_note"]


def test_acceptance_drop_table_empty_when_no_fp16() -> None:
    df = pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "spec_k": 1,
                "context_bucket": "short",
                "benchmark_label": "spec_quant_memonly",
                "acceptance_rate": 0.9,
            }
        ]
    )
    t = table_where_acceptance_dropped(df)
    assert t.empty


_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPTIONAL_CSV = _REPO_ROOT / "results" / "sweep_full_schema_v2.csv"


@pytest.mark.skipif(
    not _OPTIONAL_CSV.is_file(),
    reason="optional project results CSV",
)
def test_report_smoke_on_repo_results(tmp_path: Path) -> None:
    p = generate_phase16_report(
        _OPTIONAL_CSV,
        tmp_path,
        title="Test report",
    )
    assert p.name == "INDEX.md"
    assert (tmp_path / "HOW_TO_VIEW.md").is_file()
    text = p.read_text(encoding="utf-8")
    assert "Quantization scope" in text
    assert (tmp_path / "figures").exists()
    assert (tmp_path / "tables" / "summary_by_mode.csv").is_file()


def test_theoretical_spec_speedup_factor_mid_alpha() -> None:
    # K=1, alpha=0.5 -> (1 - 0.25) / (2 * 0.5) = 0.75
    assert abs(theoretical_spec_speedup_factor(0.5, 1) - 0.75) < 1e-9


def test_joint_sparse_quant_vs_sparse_only_synthetic() -> None:
    keys = dict(prompt_id="p1", context_bucket="short", spec_k=3, sparsity_budget=0.4, trial_index=0)
    df = pd.DataFrame(
        [
            {
                **keys,
                "mode": "sparse_only",
                "benchmark_label": "spec_sparse",
                "quantization_type": "none",
                "quant_bits_effective": 16,
                "acceptance_rate": 0.8,
            },
            {
                **keys,
                "mode": "sparse_quant",
                "benchmark_label": "spec_sparse_quant_memonly",
                "quantization_type": "memory_only",
                "quant_bits_effective": 4,
                "acceptance_rate": 0.5,
            },
        ]
    )
    t = table_joint_sparse_quant_vs_sparse_only(df)
    assert not t.empty
    row = t[t["quant_bits_effective"] == 4].iloc[0]
    assert float(row["mean_delta_sparse_only_minus_joint"]) > 0


def test_effective_compression_ratio_synthetic() -> None:
    keys = dict(prompt_id="p1", context_bucket="short", spec_k=3, trial_index=0)
    df = pd.DataFrame(
        [
            {
                **keys,
                "mode": "speculative_fp16",
                "benchmark_label": "spec_fp16",
                "quantization_type": "none",
                "quant_bits_effective": 16,
                "logical_draft_kv_bytes": 1_000_000,
                "tokens_per_sec": 100.0,
            },
            {
                **keys,
                "mode": "sparse_only",
                "benchmark_label": "spec_sparse",
                "quantization_type": "none",
                "quant_bits_effective": 16,
                "logical_draft_kv_bytes": 500_000,
                "tokens_per_sec": 90.0,
            },
        ]
    )
    t = table_effective_compression_ratio(df)
    assert not t.empty
    sparse = t[t["benchmark_label"] == "spec_sparse"].iloc[0]
    assert abs(float(sparse["effective_compression_ratio"]) - 2.0) < 1e-6


def test_load_benchmark_csv_requires_v2_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_label"):
        load_benchmark_csv(bad)
