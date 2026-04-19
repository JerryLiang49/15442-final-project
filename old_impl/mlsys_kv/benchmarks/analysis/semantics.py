"""Verify CSV metadata matches implementation semantics; summarize quantization scope."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlsys_kv.benchmarks.experiment_schema import (
    QUANTIZATION_TYPE_MEMORY_ONLY,
    QUANTIZATION_TYPE_RUNTIME_ACCELERATED,
    benchmark_label_for_canonical_mode,
    quantization_type_for_row,
)


def expected_quantization_series(df: pd.DataFrame) -> pd.Series:
    """Recompute ``quantization_type`` from ``mode`` + ``quant_bits_effective`` (source of truth)."""

    def _one(row: pd.Series) -> str:
        try:
            qe = int(row["quant_bits_effective"])
        except (TypeError, ValueError):
            qe = -1
        return quantization_type_for_row(canonical_mode=str(row["mode"]), quant_bits_effective=qe)

    return df.apply(_one, axis=1)


def expected_benchmark_label_series(df: pd.DataFrame) -> pd.Series:
    return df["mode"].map(lambda m: benchmark_label_for_canonical_mode(str(m)))


def assert_semantics_consistent(df: pd.DataFrame) -> list[str]:
    """Return warning strings if rows disagree with recomputed semantics; empty if consistent."""

    issues: list[str] = []
    if df.empty:
        return issues

    eq = expected_quantization_series(df)
    bad_q = df["quantization_type"].astype(str) != eq.astype(str)
    n_bad_q = int(bad_q.sum())
    if n_bad_q:
        issues.append(
            f"{n_bad_q} rows: CSV quantization_type != recomputed from mode+quant_bits_effective "
            "(see experiment_schema.quantization_type_for_row)"
        )

    el = expected_benchmark_label_series(df)
    bad_l = df["benchmark_label"].astype(str) != el.astype(str)
    n_bad_l = int(bad_l.sum())
    if n_bad_l:
        issues.append(f"{n_bad_l} rows: benchmark_label != expected for canonical mode")

    if "is_parallel_verification" in df.columns:
        mode = df["mode"].astype(str)
        exp_b = mode != "autoregressive"

        def _cell_bool(x: object) -> bool:
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            return s in ("true", "1", "yes")

        par_b = df["is_parallel_verification"].map(_cell_bool)
        bad_p = par_b != exp_b
        if bad_p.any():
            issues.append(f"{int(bad_p.sum())} rows: is_parallel_verification inconsistent with mode")

    return issues


def quantization_scope_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Counts and flags for captions: memory-only vs none; whether runtime-accelerated appears."""

    if df.empty:
        return {
            "n_rows": 0,
            "has_runtime_accelerated_quant": False,
            "quantization_type_counts": {},
            "caption_note": "No rows.",
        }

    qt = df["quantization_type"].astype(str).value_counts().to_dict()
    has_runtime = (df["quantization_type"].astype(str) == QUANTIZATION_TYPE_RUNTIME_ACCELERATED).any()

    if has_runtime:
        note = (
            "This sweep includes rows labeled runtime_accelerated quantization; "
            "interpret speedups as potentially including compute effects, not only memory traffic."
        )
    elif qt.get(QUANTIZATION_TYPE_MEMORY_ONLY, 0) > 0:
        note = (
            "Quantization in this sweep is **memory-only** (packed KV; attention still runs in "
            "higher precision after dequant). Do not claim INT KV alone reduced attention **runtime** "
            "unless paired with profiling that separates dequant vs matmul."
        )
    else:
        note = (
            "No memory_only quantization rows in this slice; FP16-equivalent paths use quantization_type=none."
        )

    return {
        "n_rows": len(df),
        "has_runtime_accelerated_quant": bool(has_runtime),
        "quantization_type_counts": qt,
        "benchmark_label_counts": df["benchmark_label"].astype(str).value_counts().to_dict(),
        "caption_note": note,
        "semantics_reference": "mlsys_kv.benchmarks.experiment_schema.quantization_type_for_row",
    }
