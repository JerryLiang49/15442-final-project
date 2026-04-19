"""Aggregate tables for report-ready summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.analysis.labels import display_benchmark_label
from benchmarks.analysis.stats import (
    enrich_kv_mb,
    enrich_sequence_length,
    summarize_speedup_and_pvalues,
)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``benchmark_label`` (and ``max_new_tokens`` when that column varies)."""

    df = enrich_sequence_length(enrich_kv_mb(df))
    agg_map: dict[str, str] = {
        "tokens_per_sec": "mean",
        "latency_e2e_s": "mean",
        "acceptance_rate": "mean",
        "kv_cache_verifier_mb": "mean",
        "logical_draft_kv_bytes": "mean",
        "sequence_length_tokens": "mean",
    }
    use = {k: v for k, v in agg_map.items() if k in df.columns}
    group_cols = ["benchmark_label"]
    if "max_new_tokens" in df.columns and df["max_new_tokens"].nunique() > 1:
        group_cols.append("max_new_tokens")
    t = df.groupby(group_cols, dropna=False).agg(use)
    t = t.reset_index()
    qt_first = (
        df.groupby(group_cols)["quantization_type"].first().reset_index()
        if "quantization_type" in df.columns
        else None
    )

    def _disp(row: pd.Series) -> str:
        lab = str(row["benchmark_label"])
        if qt_first is not None and "quantization_type" in qt_first.columns:
            mask = qt_first["benchmark_label"] == lab
            if "max_new_tokens" in group_cols and "max_new_tokens" in row.index:
                mask = mask & (qt_first["max_new_tokens"] == row["max_new_tokens"])
            sub = qt_first.loc[mask]
            qt = str(sub["quantization_type"].iloc[0]) if len(sub) else None
        else:
            qt = None
        return display_benchmark_label(lab, quantization_type=qt)

    t["display_name"] = t.apply(_disp, axis=1)
    # Trial std for throughput
    if "trial_index" in df.columns:
        ts = df.groupby(group_cols, dropna=False)["tokens_per_sec"].std(ddof=1).reset_index()
        ts = ts.rename(columns={"tokens_per_sec": "tokens_per_sec_std_across_trials"})
        t = t.merge(ts, on=group_cols, how="left")
    speed = summarize_speedup_and_pvalues(df)
    if not speed.empty:
        merge_on = ["benchmark_label"]
        if "max_new_tokens" in speed.columns:
            merge_on.append("max_new_tokens")
        cols = [
            c
            for c in ["benchmark_label", "max_new_tokens", "speedup_mean", "speedup_std", "p_value_diff_vs_ar"]
            if c in speed.columns
        ]
        t = t.merge(speed[cols], on=merge_on, how="left")
    else:
        t["speedup_mean"] = np.nan
        t["speedup_std"] = np.nan
        t["p_value_diff_vs_ar"] = np.nan
    return t


def build_mode_comparison_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Flattened summary suitable for ``tables/summary_by_mode.csv``."""

    return build_summary_table(df)
