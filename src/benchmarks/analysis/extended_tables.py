"""Phase 16 extended tables: joint sparse×quant effects, compression, verification model, context."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    scipy_stats = None  # type: ignore[assignment]
    HAS_SCIPY = False

from benchmarks.experiment_schema import (
    BENCHMARK_LABEL_SPEC_FP16,
    BENCHMARK_LABEL_SPEC_SPARSE,
    BENCHMARK_LABEL_SPEC_SPARSE_QUANT_MEMONLY,
)


def _cell_keys_with_genlen(df: pd.DataFrame, base: list[str]) -> list[str]:
    """Include ``max_new_tokens`` in grouping/merge keys when the CSV has multiple generation lengths."""

    out = list(base)
    if "max_new_tokens" in df.columns:
        out.append("max_new_tokens")
    return out


def _fp16_dense_draft_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Per cell: draft KV bytes from ``spec_fp16`` rows (includes ``max_new_tokens`` when present)."""

    need = {"benchmark_label", "prompt_id", "context_bucket", "spec_k", "trial_index", "logical_draft_kv_bytes"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    fp = df[df["benchmark_label"].astype(str) == BENCHMARK_LABEL_SPEC_FP16].copy()
    if fp.empty:
        return pd.DataFrame()

    gcols = _cell_keys_with_genlen(fp, ["prompt_id", "context_bucket", "spec_k", "trial_index"])
    g = fp.groupby(gcols, dropna=False)["logical_draft_kv_bytes"].mean()
    return g.rename("fp16_dense_draft_bytes_baseline").reset_index()


def table_effective_compression_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Effective storage compression vs dense FP16 draft (same prompt × bucket × K × trial).

    ``effective_compression_ratio`` = ``fp16_dense_draft_bytes_baseline / logical_draft_kv_bytes``
    (higher ⇒ smaller packed draft vs dense FP16 baseline).

    For INT packing, low-bit rows also get ``quantization_factor`` = ``quant_bits_effective / 16`` (informative).
    """

    base = _fp16_dense_draft_baseline(df)
    if base.empty:
        return pd.DataFrame()

    sub = df[df["benchmark_label"].astype(str) != "ar"].copy()
    mcols = _cell_keys_with_genlen(sub, ["prompt_id", "context_bucket", "spec_k", "trial_index"])
    need = sub.merge(base, on=mcols, how="inner")
    if need.empty:
        return pd.DataFrame()

    d = need["logical_draft_kv_bytes"].astype(float).replace(0, np.nan)
    need = need.assign(
        fp16_dense_draft_bytes_baseline=need["fp16_dense_draft_bytes_baseline"].astype(float),
        effective_compression_ratio=need["fp16_dense_draft_bytes_baseline"].astype(float) / d,
    )
    qb = pd.to_numeric(need.get("quant_bits_effective"), errors="coerce")
    need = need.assign(
        quantization_factor_bits_over_16=np.where(
            (qb > 0) & (qb < 16), qb / 16.0, np.nan
        ),
    )
    cols = [
        "benchmark_label",
        "mode",
        "quantization_type",
        "prompt_id",
        "context_bucket",
        "spec_k",
        "sparsity_budget",
        "quant_bits_effective",
        "trial_index",
        "fp16_dense_draft_bytes_baseline",
        "logical_draft_kv_bytes",
        "effective_compression_ratio",
        "tokens_per_sec",
        "quantization_factor_bits_over_16",
    ]
    cols = [c for c in cols if c in need.columns]
    return need[cols].sort_values(
        ["benchmark_label", "context_bucket", "spec_k", "prompt_id", "trial_index"]
    )


def table_joint_sparse_quant_vs_sparse_only(df: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison: ``sparse_only`` vs ``sparse_quant`` acceptance (same grid cell).

    Positive ``acceptance_delta_sparse_only_minus_joint`` means sparse-only **higher** acceptance
    (quant + sparse may corrupt draft–verifier agreement). Summary rows per ``quant_bits_effective``.
    """

    keys = _cell_keys_with_genlen(
        df,
        ["prompt_id", "context_bucket", "spec_k", "sparsity_budget", "trial_index"],
    )
    need = {"benchmark_label", "acceptance_rate", *keys}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    so = df[df["benchmark_label"].astype(str) == BENCHMARK_LABEL_SPEC_SPARSE].copy()
    sq = df[df["benchmark_label"].astype(str) == BENCHMARK_LABEL_SPEC_SPARSE_QUANT_MEMONLY].copy()
    if so.empty or sq.empty:
        return pd.DataFrame()

    so_m = so.groupby(keys, dropna=False)["acceptance_rate"].mean().rename("acceptance_sparse_only")
    merged = sq.merge(so_m.reset_index(), on=keys, how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["acceptance_delta_sparse_only_minus_joint"] = (
        merged["acceptance_sparse_only"].astype(float) - merged["acceptance_rate"].astype(float)
    )

    merged["quant_bits_effective"] = pd.to_numeric(merged["quant_bits_effective"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for qb, g in merged.groupby("quant_bits_effective", dropna=False):
        deltas = g["acceptance_delta_sparse_only_minus_joint"].astype(float).values
        deltas = deltas[np.isfinite(deltas)]
        n = len(deltas)
        mean_d = float(np.mean(deltas)) if n else float("nan")
        std_d = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
        p_val: float | str = float("nan")
        if HAS_SCIPY and scipy_stats is not None and n >= 2:
            t_res = scipy_stats.ttest_1samp(deltas, 0.0)
            p_val = float(t_res.pvalue)
        rows.append(
            {
                "quant_bits_effective": qb,
                "n_paired_rows": n,
                "mean_acceptance_sparse_only": float(g["acceptance_sparse_only"].mean()),
                "mean_acceptance_sparse_quant": float(g["acceptance_rate"].mean()),
                "mean_delta_sparse_only_minus_joint": mean_d,
                "std_delta": std_d,
                "p_value_delta_eq_0": p_val,
                "notes": "H0: delta=0; paired cells (prompt×bucket×K×sparsity×trial)",
            }
        )
    return pd.DataFrame(rows).sort_values("quant_bits_effective")


def theoretical_spec_speedup_factor(alpha: float, k: int) -> float:
    r"""Literature-style proxy: \((1 - \alpha^{K+1}) / ((K+1)(1-\alpha))\).

    Used as a **qualitative** comparison to measured ``tokens_per_sec`` ratios — not a proof of optimality.
    """

    a = float(alpha)
    kk = int(k)
    if kk < 1:
        return float("nan")
    if not (0.0 <= a <= 1.0):
        return float("nan")
    if abs(1.0 - a) < 1e-12:
        return 1.0
    num = 1.0 - (a ** (kk + 1))
    den = float(kk + 1) * (1.0 - a)
    if den <= 0:
        return float("nan")
    return num / den


def table_verification_bottleneck_spec_fp16(
    df: pd.DataFrame,
    *,
    baseline_label: str = "ar",
    spec_label: str = BENCHMARK_LABEL_SPEC_FP16,
) -> pd.DataFrame:
    """``spec_fp16`` vs AR: actual throughput ratio vs theoretical acceptance model (overhead proxy)."""

    need = {
        "benchmark_label",
        "prompt_id",
        "context_bucket",
        "trial_index",
        "tokens_per_sec",
        "acceptance_rate",
        "spec_k",
    }
    if not need.issubset(df.columns):
        return pd.DataFrame()

    mkeys = _cell_keys_with_genlen(df, ["prompt_id", "context_bucket", "trial_index"])
    ar_cols = [c for c in mkeys if c in df.columns] + ["tokens_per_sec"]
    ar = df[df["benchmark_label"].astype(str) == baseline_label][ar_cols].rename(
        columns={"tokens_per_sec": "tokens_per_sec_ar"}
    )

    sp = df[df["benchmark_label"].astype(str) == spec_label].copy()
    if ar.empty or sp.empty:
        return pd.DataFrame()

    m = sp.merge(ar, on=mkeys, how="inner")
    if m.empty:
        return pd.DataFrame()

    m["actual_speedup_tps_ratio"] = m["tokens_per_sec"].astype(float) / m["tokens_per_sec_ar"].astype(float).replace(
        0, np.nan
    )
    m["theoretical_speedup_factor"] = [
        theoretical_spec_speedup_factor(float(a), int(k))
        for a, k in zip(m["acceptance_rate"].astype(float), m["spec_k"].astype(int))
    ]
    m["system_overhead_ratio"] = m["actual_speedup_tps_ratio"] / m["theoretical_speedup_factor"].replace(0, np.nan)
    # Residual: 1 - min(actual/theory, theory/actual) style — report ratio instead
    m["interpretation"] = (
        "actual_tps_ratio_vs_theoretical_acceptance_model; "
        "overhead_ratio≈1 means measured speedup tracks model; "
        "<<1 suggests draft/verify/sync costs dominate."
    )
    keep = [
        "prompt_id",
        "context_bucket",
        "trial_index",
        "spec_k",
        "acceptance_rate",
        "tokens_per_sec",
        "tokens_per_sec_ar",
        "actual_speedup_tps_ratio",
        "theoretical_speedup_factor",
        "system_overhead_ratio",
    ]
    return m[keep].sort_values(["context_bucket", "spec_k", "prompt_id", "trial_index"])


def table_verification_bottleneck_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median actual vs theoretical speedup factor by ``spec_k`` (spec_fp16 vs AR)."""

    t = table_verification_bottleneck_spec_fp16(df)
    if t.empty:
        return pd.DataFrame()
    g = t.groupby("spec_k", dropna=False).agg(
        n=("prompt_id", "count"),
        mean_actual_ratio=("actual_speedup_tps_ratio", "mean"),
        mean_theoretical=("theoretical_speedup_factor", "mean"),
        mean_overhead_ratio=("system_overhead_ratio", "mean"),
        median_overhead_ratio=("system_overhead_ratio", "median"),
    )
    return g.reset_index().sort_values("spec_k")


def infer_context_bucket_token_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical ``prompt_len_tokens`` ranges per ``context_bucket`` in this CSV."""

    if "context_bucket" not in df.columns or "prompt_len_tokens" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("context_bucket", dropna=False)["prompt_len_tokens"].agg(["min", "max", "median", "count"])
    return g.reset_index().rename(
        columns={"min": "prompt_tokens_min", "max": "prompt_tokens_max", "median": "prompt_tokens_median"}
    )


def table_context_bucket_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Throughput and speedup vs AR by ``context_bucket``; sparsification lift for sparse modes."""

    need = {"benchmark_label", "context_bucket", "tokens_per_sec", "prompt_id"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    mkeys = _cell_keys_with_genlen(df, ["prompt_id", "context_bucket", "trial_index"])
    ar_cols = [c for c in mkeys if c in df.columns] + ["tokens_per_sec"]
    ar = df[df["benchmark_label"].astype(str) == "ar"][ar_cols].rename(columns={"tokens_per_sec": "ar_tps"})

    rows: list[dict[str, Any]] = []
    for lab, g in df[df["benchmark_label"].astype(str) != "ar"].groupby("benchmark_label"):
        m = g.merge(ar, on=mkeys, how="inner")
        if m.empty:
            continue
        m["speedup_vs_ar"] = m["tokens_per_sec"].astype(float) / m["ar_tps"].astype(float).replace(0, np.nan)
        for bucket, gb in m.groupby("context_bucket", dropna=False):
            vals = gb["speedup_vs_ar"].astype(float).values
            vals = vals[np.isfinite(vals)]
            rows.append(
                {
                    "benchmark_label": lab,
                    "context_bucket": bucket,
                    "mean_tokens_per_sec": float(gb["tokens_per_sec"].mean()),
                    "std_tokens_per_sec": float(gb["tokens_per_sec"].std(ddof=1))
                    if len(gb) > 1
                    else 0.0,
                    "mean_speedup_vs_ar": float(np.mean(vals)) if len(vals) else float("nan"),
                    "n_rows": int(len(gb)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Sparsification narrative: sparse modes vs AR in long bucket
    return out.sort_values(["context_bucket", "benchmark_label"])


def table_mean_compression_and_throughput_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean ``effective_compression_ratio`` and throughput per ``benchmark_label``."""

    t = table_effective_compression_ratio(df)
    if t.empty:
        return pd.DataFrame()
    g = t.groupby("benchmark_label", dropna=False).agg(
        mean_effective_compression=("effective_compression_ratio", "mean"),
        std_effective_compression=("effective_compression_ratio", "std"),
        mean_tokens_per_sec=("tokens_per_sec", "mean"),
        n=("prompt_id", "count"),
    )
    return g.reset_index().sort_values("mean_tokens_per_sec", ascending=False)
