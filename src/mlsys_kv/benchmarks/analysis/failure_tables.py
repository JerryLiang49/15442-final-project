"""Failure-oriented tables: acceptance drops, sparse overhead, quant memory vs speed."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compression_ratio_draft_vs_verifier(df: pd.DataFrame) -> pd.Series:
    """``verifier_bytes / draft_bytes`` (≥1 when draft is smaller). NaN if invalid."""

    d = df["logical_draft_kv_bytes"].astype(float)
    v = df["logical_verifier_kv_bytes"].astype(float)
    out = v / d.replace(0, np.nan)
    return out


def table_acceptance_by_mode_spec_k(df: pd.DataFrame) -> pd.DataFrame:
    """Mean acceptance_rate by benchmark_label and spec_k (where defined)."""

    sub = df[df["benchmark_label"].astype(str) != "ar"].copy()
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["benchmark_label", "spec_k"], dropna=False)["acceptance_rate"].agg(["mean", "std", "count"])
    return g.reset_index().sort_values(["benchmark_label", "spec_k"])


def table_where_acceptance_dropped(
    df: pd.DataFrame,
    *,
    ref_label: str = "spec_fp16",
    min_delta: float = 0.05,
) -> pd.DataFrame:
    """Rows / groups where acceptance is materially below reference mode (same prompt, K, bucket)."""

    need = {"prompt_id", "spec_k", "context_bucket", "benchmark_label", "acceptance_rate"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    sub = df[df["benchmark_label"].astype(str) != "ar"].copy()
    if sub.empty:
        return pd.DataFrame()

    piv = sub.pivot_table(
        index=["prompt_id", "spec_k", "context_bucket"],
        columns="benchmark_label",
        values="acceptance_rate",
        aggfunc="mean",
    )
    if ref_label not in piv.columns:
        return pd.DataFrame()

    ref = piv[ref_label]
    cols = [c for c in piv.columns if c != ref_label]
    out_rows: list[dict[str, Any]] = []
    for c in cols:
        delta = ref - piv[c]
        bad = delta[delta > min_delta]
        for idx in bad.index:
            rv = ref.loc[idx]
            cv = piv[c].loc[idx]
            if pd.isna(rv) or pd.isna(cv):
                continue
            out_rows.append(
                {
                    "prompt_id": idx[0],
                    "spec_k": idx[1],
                    "context_bucket": idx[2],
                    "reference_label": ref_label,
                    "ref_acceptance": float(rv),
                    "compare_label": c,
                    "compare_acceptance": float(cv),
                    "acceptance_drop": float(delta.loc[idx]),
                }
            )
    return pd.DataFrame(out_rows)


def table_sparse_overhead_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """Draft latency share of draft+verify for sparse modes (proxy for sparse overhead)."""

    m = df["benchmark_label"].astype(str) == "spec_sparse"
    sub = df[m].copy()
    if sub.empty:
        return pd.DataFrame()

    d = sub["draft_latency_total_s"].astype(float)
    v = sub["verify_latency_total_s"].astype(float)
    tot = d + v
    sub = sub.assign(
        draft_latency_fraction=np.where(tot > 0, d / tot, np.nan),
        compression_ratio_verifier_over_draft=compression_ratio_draft_vs_verifier(sub),
    )
    g = sub.groupby(["spec_k", "sparsity_budget"], dropna=False).agg(
        draft_latency_fraction=("draft_latency_fraction", "mean"),
        compression_ratio_verifier_over_draft=("compression_ratio_verifier_over_draft", "mean"),
        acceptance_rate=("acceptance_rate", "mean"),
        n=("prompt_id", "count"),
    )
    return g.reset_index()


def table_quant_memory_vs_speed(
    df: pd.DataFrame,
    *,
    fp16_label: str = "spec_fp16",
    quant_label: str = "spec_quant_memonly",
) -> pd.DataFrame:
    """Compare throughput and memory metrics at same (spec_k, context_bucket, prompt) — memory-only quant path."""

    need = {"benchmark_label", "spec_k", "context_bucket", "prompt_id", "tokens_per_sec", "logical_draft_kv_bytes"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    sub = df[df["benchmark_label"].isin([fp16_label, quant_label])].copy()
    if sub.empty:
        return pd.DataFrame()

    piv = sub.pivot_table(
        index=["prompt_id", "spec_k", "context_bucket"],
        columns="benchmark_label",
        values=["tokens_per_sec", "logical_draft_kv_bytes", "memory_throughput_gb_s", "latency_per_new_token_s"],
        aggfunc="mean",
    )
    # flatten columns for readability
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    return piv.reset_index()


def table_best_throughput_under_memory_cap(
    df: pd.DataFrame,
    *,
    memory_col: str = "gpu_peak_memory_bytes_after_run",
    throughput_col: str = "tokens_per_sec",
    q: int = 3,
) -> pd.DataFrame:
    """Within each **equal-count** peak-memory bucket, best mean throughput by ``benchmark_label``."""

    if memory_col not in df.columns or throughput_col not in df.columns:
        return pd.DataFrame()

    w = df[[memory_col, throughput_col, "benchmark_label", "prompt_id"]].dropna(subset=[memory_col, throughput_col])
    if w.empty or len(w) < q:
        return pd.DataFrame()

    try:
        w = w.assign(mem_bucket=pd.qcut(w[memory_col], q=q, duplicates="drop"))
    except ValueError:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for bucket, grp in w.groupby("mem_bucket", observed=True):
        best = grp.groupby("benchmark_label")[throughput_col].mean().sort_values(ascending=False)
        if best.empty:
            continue
        rows.append(
            {
                "mem_bucket": str(bucket),
                "best_benchmark_label": best.index[0],
                "mean_tokens_per_sec": float(best.iloc[0]),
                "runner_up_label": best.index[1] if len(best) > 1 else "",
                "runner_up_tps": float(best.iloc[1]) if len(best) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def table_joint_mode_vs_components(df: pd.DataFrame) -> pd.DataFrame:
    """Mean tokens/s and draft bytes for sparse+quant vs sparse-only and quant-only (same K grid overlap)."""

    labels = {"spec_sparse", "spec_quant_memonly", "spec_sparse_quant_memonly"}
    present = set(df["benchmark_label"].astype(str).unique()) & labels
    if len(present) < 2:
        return pd.DataFrame()

    sub = df[df["benchmark_label"].astype(str).isin(labels)].copy()
    g = sub.groupby("benchmark_label").agg(
        mean_tokens_per_sec=("tokens_per_sec", "mean"),
        mean_draft_kv_bytes=("logical_draft_kv_bytes", "mean"),
        mean_acceptance=("acceptance_rate", "mean"),
        n=("prompt_id", "count"),
    )
    return g.reset_index().sort_values("mean_tokens_per_sec", ascending=False)
