"""Trial-level stats: mean/std, speedup vs AR, optional paired p-values."""

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


def enrich_sequence_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sequence_length_tokens`` ≈ prompt + generated tokens (per sweep config)."""

    out = df.copy()
    if "prompt_len_tokens" in out.columns and "max_new_tokens" in out.columns:
        out["sequence_length_tokens"] = (
            out["prompt_len_tokens"].astype(float) + out["max_new_tokens"].astype(float)
        )
    return out


def enrich_kv_mb(df: pd.DataFrame) -> pd.DataFrame:
    """Verifier KV size in MB (logical bytes)."""

    out = df.copy()
    if "logical_verifier_kv_bytes" in out.columns:
        out["kv_cache_verifier_mb"] = out["logical_verifier_kv_bytes"].astype(float) / 1e6
    return out


def residual_overhead_s(row: pd.Series) -> float:
    """Time not attributed to draft or verify timers (trimming, sync, other)."""

    lat = float(row.get("latency_e2e_s") or np.nan)
    d = row.get("draft_latency_total_s")
    v = row.get("verify_latency_total_s")
    if pd.isna(d) or pd.isna(v) or str(row.get("benchmark_label")) == "ar":
        return np.nan
    d, v = float(d), float(v)
    rem = lat - d - v
    return max(0.0, rem)


def mean_std_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Mean and std across ``trial_index`` (or single rows) within each group."""

    g = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"])
    g.columns = [f"{m}_{stat}" for m, stat in g.columns]
    return g.reset_index()


def paired_speedup_vs_ar(
    df: pd.DataFrame,
    *,
    baseline_label: str = "ar",
    metric: str = "tokens_per_sec",
) -> pd.DataFrame:
    """Per-row speedup vs AR on the same (prompt_id, context_bucket, trial_index)."""

    need = {"benchmark_label", "prompt_id", "context_bucket", "trial_index", metric}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    ar = df[df["benchmark_label"].astype(str) == baseline_label][
        ["prompt_id", "context_bucket", "trial_index", metric]
    ].rename(columns={metric: f"{metric}_ar"})

    m = df.merge(ar, on=["prompt_id", "context_bucket", "trial_index"], how="inner")
    m = m[m["benchmark_label"].astype(str) != baseline_label]
    if m.empty:
        return pd.DataFrame()

    m["speedup_vs_ar"] = m[metric].astype(float) / m[f"{metric}_ar"].astype(float).replace(0, np.nan)
    return m


def summarize_speedup_and_pvalues(
    df: pd.DataFrame,
    *,
    baseline_label: str = "ar",
    metric: str = "tokens_per_sec",
) -> pd.DataFrame:
    """Per ``benchmark_label``: mean speedup, std, paired t-test vs AR (if scipy)."""

    paired = paired_speedup_vs_ar(df, baseline_label=baseline_label, metric=metric)
    if paired.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for lab, g in paired.groupby("benchmark_label"):
        ratios = g["speedup_vs_ar"].astype(float).values
        ratios = ratios[np.isfinite(ratios)]
        n = len(ratios)
        if n == 0:
            continue
        mean_s = float(np.mean(ratios))
        std_s = float(np.std(ratios, ddof=1)) if n > 1 else 0.0

        # Paired differences on metric: need raw paired values
        ar_sub = df[df["benchmark_label"].astype(str) == baseline_label][
            ["prompt_id", "context_bucket", "trial_index", metric]
        ].rename(columns={metric: "v_ar"})
        merged = g.merge(ar_sub, on=["prompt_id", "context_bucket", "trial_index"])
        merged["diff"] = merged[metric].astype(float) - merged["v_ar"].astype(float)
        diffs = merged["diff"].values
        diffs = diffs[np.isfinite(diffs)]
        p_val: float | str = float("nan")
        if HAS_SCIPY and len(diffs) >= 2 and scipy_stats is not None:
            t_res = scipy_stats.ttest_1samp(diffs, 0.0)
            p_val = float(t_res.pvalue)

        rows.append(
            {
                "benchmark_label": lab,
                "n_pairs": n,
                "speedup_mean": mean_s,
                "speedup_std": std_s,
                "p_value_diff_vs_ar": p_val,
                "notes": "p_value: paired one-sample t on (mode - AR) for same prompt×bucket×trial"
                if HAS_SCIPY
                else "p_value: install scipy for paired t-test",
            }
        )
    return pd.DataFrame(rows)


def trial_stats_for_plots(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metric: str = "tokens_per_sec",
) -> pd.DataFrame:
    """Mean and std of metric across rows (trials) within each group — for error bars."""

    return df.groupby(group_cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
