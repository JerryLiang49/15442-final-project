"""Short markdown-friendly failure-analysis blurbs from dataframe stats."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlsys_kv.benchmarks.analysis.failure_tables import (
    table_sparse_overhead_fraction,
    table_where_acceptance_dropped,
)
def acceptance_loss_summary(df: pd.DataFrame, *, top_n: int = 5) -> str:
    t = table_where_acceptance_dropped(df)
    if t.empty:
        return "_No material acceptance drops vs `spec_fp16` at default threshold._\n"
    t = t.sort_values("acceptance_drop", ascending=False).head(top_n)
    lines = [
        f"- **{row['compare_label']}** vs FP16 at prompt `{row['prompt_id']}`, K={row['spec_k']}, "
        f"bucket={row['context_bucket']}: drop **{row['acceptance_drop']:.3f}** "
        f"(FP16 {row['ref_acceptance']:.3f} → {row['compare_acceptance']:.3f})"
        for _, row in t.iterrows()
    ]
    return "\n".join(lines) + "\n"


def sparse_overhead_summary(df: pd.DataFrame) -> str:
    t = table_sparse_overhead_fraction(df)
    if t.empty:
        return "_No sparse-mode rows._\n"
    hi = t.sort_values("draft_latency_fraction", ascending=False).head(3)
    lines = [
        f"- K={row['spec_k']}, sparsity_budget={row['sparsity_budget']}: draft share of draft+verify = "
        f"**{row['draft_latency_fraction']:.2f}** (mean over prompts)"
        for _, row in hi.iterrows()
    ]
    return (
        "**Interpretation:** high draft fraction suggests **selector/refresh** cost dominates that regime.\n\n"
        + "\n".join(lines)
        + "\n"
    )


def quantization_overhead_summary(df: pd.DataFrame) -> str:
    sub = df[df["benchmark_label"].astype(str) == "spec_quant_memonly"].copy()
    if sub.empty:
        return "_No memory-only quant rows (`spec_quant_memonly`)._\n"
    fp = df[df["benchmark_label"].astype(str) == "spec_fp16"].copy()
    if fp.empty:
        return "_No FP16 spec baseline to compare._\n"

    # Same (prompt, bucket, K, trial [, max_new_tokens]): ratio of latency
    keys = ["prompt_id", "context_bucket", "spec_k", "trial_index"]
    if "max_new_tokens" in df.columns:
        keys.append("max_new_tokens")
    q = sub.merge(
        fp[keys + ["latency_e2e_s"]].rename(columns={"latency_e2e_s": "lat_fp16"}),
        on=keys,
        how="inner",
    )
    if q.empty:
        return "_No aligned FP16 vs quant rows for overhead comparison._\n"
    q["latency_ratio_q_over_fp16"] = q["latency_e2e_s"].astype(float) / q["lat_fp16"].astype(float)
    mean_r = float(np.nanmean(q["latency_ratio_q_over_fp16"]))
    return (
        f"- Mean **latency ratio** (memory-only quant / FP16 spec), aligned runs: **{mean_r:.3f}**.\n"
        "- Values **> 1** are expected when dequant + standard attention dominates; this is **not** a "
        "contradiction of memory-only KV benefits on **bytes moved**.\n"
    )


def joint_tradeoff_summary(df: pd.DataFrame) -> str:
    j = df[df["benchmark_label"].astype(str) == "spec_sparse_quant_memonly"]
    if j.empty:
        return "_No `spec_sparse_quant_memonly` rows in this CSV — joint tradeoff table may be empty._\n"
    return (
        "- Joint mode combines **sparse retention** (runtime: scoring/refresh) with **memory-only INT KV**.\n"
        "- Attribute **speed** changes primarily to **sparsity**; attribute **KV bytes** to **both**.\n"
    )


def build_failure_section(df: pd.DataFrame) -> dict[str, str]:
    return {
        "acceptance_loss": acceptance_loss_summary(df),
        "sparse_overhead": sparse_overhead_summary(df),
        "quantization_overhead": quantization_overhead_summary(df),
        "joint_tradeoffs": joint_tradeoff_summary(df),
    }
