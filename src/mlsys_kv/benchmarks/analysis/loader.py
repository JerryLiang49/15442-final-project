"""Load and normalize Phase 15 benchmark sweep CSV (schema v2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_benchmark_csv(path: str | Path, *, ok_only: bool = True) -> pd.DataFrame:
    """Read CSV; coerce numerics; optional filter ``status == ok``.

    Raises ``ValueError`` if required v2 columns are missing.
    """

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    missing = [c for c in ("benchmark_label", "quantization_type", "mode") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing Phase 15 columns {missing}: {p}")

    num_cols = [
        "latency_e2e_s",
        "latency_per_new_token_s",
        "acceptance_rate",
        "tokens_per_sec",
        "logical_draft_kv_bytes",
        "logical_verifier_kv_bytes",
        "gpu_peak_memory_bytes_after_run",
        "model_weights_gb",
        "kv_cache_size_gb",
        "effective_memory_bandwidth_gb_s",
        "memory_throughput_gb_s",
        "draft_latency_total_s",
        "verify_latency_total_s",
        "spec_k",
        "sparsity_budget",
        "quant_bits_requested",
        "quant_bits_effective",
        "prompt_len_tokens",
        "max_new_tokens",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if ok_only and "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"].copy()

    return df


def aggregate_by_mode_bucket(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Mean across trials for key metrics (default: benchmark_label × context_bucket)."""

    gc = group_cols or ["benchmark_label", "context_bucket"]
    for c in gc:
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r} for aggregation")

    metrics = [
        "tokens_per_sec",
        "memory_throughput_gb_s",
        "acceptance_rate",
        "latency_e2e_s",
        "latency_per_new_token_s",
        "logical_draft_kv_bytes",
        "logical_verifier_kv_bytes",
        "gpu_peak_memory_bytes_after_run",
        "draft_latency_total_s",
        "verify_latency_total_s",
    ]
    use = [m for m in metrics if m in df.columns]
    g = df.groupby(gc, dropna=False)[use].mean(numeric_only=True)
    return g.reset_index()
