"""Phase 16: analysis, plots, and report-ready artifacts from benchmark CSV v2."""

from benchmarks.analysis.aggregates import build_summary_table
from benchmarks.analysis.loader import load_benchmark_csv
from benchmarks.analysis.report import generate_phase16_report
from benchmarks.analysis.semantics import (
    assert_semantics_consistent,
    quantization_scope_summary,
)

__all__ = [
    "assert_semantics_consistent",
    "build_summary_table",
    "generate_phase16_report",
    "load_benchmark_csv",
    "quantization_scope_summary",
]
