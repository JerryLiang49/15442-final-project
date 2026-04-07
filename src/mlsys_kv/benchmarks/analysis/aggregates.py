"""Aggregate tables for report-ready summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsys_kv.benchmarks.analysis.labels import display_benchmark_label
from mlsys_kv.benchmarks.analysis.stats import enrich_kv_mb, enrich_sequence_length, summarize_speedup_and_pvalues


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``benchmark_label``: mean throughput, latency, acceptance, KV MB, honest display name."""

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
    t = df.groupby("benchmark_label", dropna=False).agg(use)
    t = t.reset_index()
    qt_first = df.groupby("benchmark_label")["quantization_type"].first() if "quantization_type" in df.columns else None

    def _disp(row: pd.Series) -> str:
        lab = str(row["benchmark_label"])
        qt = str(qt_first[lab]) if qt_first is not None and lab in qt_first.index else None
        return display_benchmark_label(lab, quantization_type=qt)

    t["display_name"] = t.apply(_disp, axis=1)
    # Trial std for throughput
    if "trial_index" in df.columns:
        ts = df.groupby("benchmark_label", dropna=False)["tokens_per_sec"].std(ddof=1).reset_index()
        ts = ts.rename(columns={"tokens_per_sec": "tokens_per_sec_std_across_trials"})
        t = t.merge(ts, on="benchmark_label", how="left")
    speed = summarize_speedup_and_pvalues(df)
    if not speed.empty:
        t = t.merge(
            speed[["benchmark_label", "speedup_mean", "speedup_std", "p_value_diff_vs_ar"]],
            on="benchmark_label",
            how="left",
        )
    else:
        t["speedup_mean"] = np.nan
        t["speedup_std"] = np.nan
        t["p_value_diff_vs_ar"] = np.nan
    return t


def build_mode_comparison_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Flattened summary suitable for ``tables/summary_by_mode.csv``."""

    return build_summary_table(df)
