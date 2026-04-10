"""Phase 16 core plots (matplotlib, non-interactive)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlsys_kv.benchmarks.analysis.extended_tables import (
    table_context_bucket_performance,
    table_effective_compression_ratio,
    table_verification_bottleneck_spec_fp16,
)
from mlsys_kv.benchmarks.analysis.stats import (
    enrich_kv_mb,
    enrich_sequence_length,
    residual_overhead_s,
)

# Consistent colors by benchmark label (extend as needed)
LABEL_COLORS: dict[str, str] = {
    "ar": "#444444",
    "spec_fp16": "#1f77b4",
    "spec_quant_memonly": "#ff7f0e",
    "spec_sparse": "#2ca02c",
    "spec_sparse_quant_memonly": "#9467bd",
}

# Preferred order for paper-style figures
MODE_ORDER: list[str] = [
    "ar",
    "spec_fp16",
    "spec_quant_memonly",
    "spec_sparse",
    "spec_sparse_quant_memonly",
]


def _color(label: str) -> str:
    return LABEL_COLORS.get(str(label), "#7f7f7f")


def plot_throughput_vs_memory_bytes(
    df: pd.DataFrame,
    out: Path,
    *,
    throughput_col: str = "tokens_per_sec",
    memory_col: str = "gpu_peak_memory_bytes_after_run",
) -> None:
    """Scatter: peak VRAM vs throughput; color by ``benchmark_label``."""

    sub = df.dropna(subset=[throughput_col, memory_col, "benchmark_label"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, g in sub.groupby("benchmark_label"):
        ax.scatter(
            g[memory_col] / 1e9,
            g[throughput_col],
            label=str(lab),
            alpha=0.5,
            s=22,
            c=_color(str(lab)),
            edgecolors="none",
        )
    ax.set_xlabel("Peak GPU memory (GB, torch max allocated proxy)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Throughput vs peak memory (color = benchmark_label)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_acceptance_vs_compression(
    df: pd.DataFrame,
    out: Path,
) -> None:
    """Scatter: compression proxy vs acceptance (speculative rows only)."""

    sub = df[df["benchmark_label"].astype(str) != "ar"].copy()
    sub = sub.dropna(subset=["acceptance_rate", "logical_draft_kv_bytes", "logical_verifier_kv_bytes"])
    if sub.empty:
        return

    ratio = sub["logical_verifier_kv_bytes"].astype(float) / sub["logical_draft_kv_bytes"].astype(float).replace(
        0, np.nan
    )
    sub = sub.assign(compression_ratio_ver_over_draft=ratio)

    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, g in sub.groupby("benchmark_label"):
        gg = g.dropna(subset=["compression_ratio_ver_over_draft"])
        if gg.empty:
            continue
        ax.scatter(
            gg["compression_ratio_ver_over_draft"],
            gg["acceptance_rate"],
            label=str(lab),
            alpha=0.55,
            s=24,
            c=_color(str(lab)),
            edgecolors="none",
        )
    ax.set_xlabel("Verifier/draft KV size ratio (larger ⇒ smaller draft vs full verifier cache)")
    ax.set_ylabel("Acceptance rate")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Acceptance vs compression strength (verifier/draft KV ratio; not attention runtime)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_throughput_by_context_bucket(
    df: pd.DataFrame,
    out: Path,
    *,
    throughput_col: str = "tokens_per_sec",
) -> None:
    """Bar chart: mean throughput by context_bucket × benchmark_label (error bars = std across trials)."""

    need = {"context_bucket", "benchmark_label", throughput_col}
    if not need.issubset(df.columns):
        return

    means = df.pivot_table(
        index="context_bucket",
        columns="benchmark_label",
        values=throughput_col,
        aggfunc="mean",
    )
    stds = df.pivot_table(
        index="context_bucket",
        columns="benchmark_label",
        values=throughput_col,
        aggfunc="std",
    ).fillna(0)
    if means.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(means.index))
    w = 0.8 / max(len(means.columns), 1)
    for i, col in enumerate(means.columns):
        offs = (i - len(means.columns) / 2) * w + w / 2
        ax.bar(
            x + offs,
            means[col].values,
            width=w * 0.95,
            yerr=stds[col].values,
            label=str(col),
            color=_color(str(col)),
            ecolor="#333333",
            capsize=2,
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in means.index])
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xlabel("Context bucket")
    ax.set_title("Throughput by context bucket (mean ± std across trials)")
    ax.legend(title="benchmark_label", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ablation_modes(
    df: pd.DataFrame,
    out: Path,
    *,
    throughput_col: str = "tokens_per_sec",
) -> None:
    """Mean ± std throughput by benchmark_label (across all prompts/trials)."""

    if "benchmark_label" not in df.columns or throughput_col not in df.columns:
        return

    st = df.groupby("benchmark_label", dropna=False)[throughput_col].agg(["mean", "std", "count"])
    st = st.reindex([m for m in MODE_ORDER if m in st.index] + [x for x in st.index if x not in MODE_ORDER])
    st = st.dropna(how="all")
    if st.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(st))
    ax.bar(
        x,
        st["mean"].values,
        yerr=st["std"].fillna(0).values,
        color=[_color(str(i)) for i in st.index],
        ecolor="#222222",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(st.index), rotation=15, ha="right")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Ablation: mean ± std tokens/s by mode (all prompts×trials)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pareto_throughput_vs_kv_mb(
    df: pd.DataFrame,
    out: Path,
    *,
    throughput_col: str = "tokens_per_sec",
) -> None:
    """Pareto-style scatter: throughput vs logical verifier KV (MB); color=mode, size≈K."""

    d = enrich_kv_mb(enrich_sequence_length(df))
    need = {"kv_cache_verifier_mb", throughput_col, "benchmark_label", "spec_k"}
    if not need.issubset(d.columns):
        return
    sub = d.dropna(subset=["kv_cache_verifier_mb", throughput_col])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for lab, g in sub.groupby("benchmark_label"):
        sk = g["spec_k"].astype(float).fillna(1)
        ax.scatter(
            g["kv_cache_verifier_mb"],
            g[throughput_col],
            label=str(lab),
            c=_color(str(lab)),
            s=np.clip(sk * 14 + 18, 24, 220),
            alpha=0.55,
            edgecolors="white",
            linewidths=0.4,
        )
        idx = np.argsort(g["kv_cache_verifier_mb"].values)
        xs = g["kv_cache_verifier_mb"].values[idx]
        ys = g[throughput_col].values[idx]
        if len(xs) >= 2:
            ax.plot(xs, ys, color=_color(str(lab)), alpha=0.25, linewidth=1.2, linestyle="--")

    ax.set_xlabel("Verifier KV cache footprint (MB, logical)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Pareto-style frontier: throughput vs KV footprint (point size ∝ K)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_stacked_latency_single_prompt(
    df: pd.DataFrame,
    out: Path,
    *,
    prompt_id: str | None = None,
    spec_k: int | None = None,
) -> None:
    """Stacked bars: Draft / Verify / Overhead for one long prompt; AR as single segment."""

    need = {"benchmark_label", "latency_e2e_s", "prompt_id", "prompt_len_tokens"}
    if not need.issubset(df.columns):
        return

    d = df.copy()
    d["overhead_s"] = d.apply(residual_overhead_s, axis=1)

    if prompt_id is None:
        pid = str(d.loc[d["prompt_len_tokens"].idxmax(), "prompt_id"])
    else:
        pid = str(prompt_id)

    block = d[d["prompt_id"].astype(str) == pid]
    if block.empty:
        return

    if spec_k is None:
        sk = block[block["benchmark_label"].astype(str) != "ar"]["spec_k"]
        spec_k = int(sk.median()) if not sk.empty else 1

    rows: list[dict[str, float | str]] = []
    for lab in MODE_ORDER:
        sub = block[block["benchmark_label"].astype(str) == lab]
        if lab != "ar":
            sub = sub[sub["spec_k"].astype(int) == int(spec_k)]
        if sub.empty:
            continue
        draft_m = sub["draft_latency_total_s"].mean() if "draft_latency_total_s" in sub.columns else np.nan
        ver_m = sub["verify_latency_total_s"].mean() if "verify_latency_total_s" in sub.columns else np.nan
        oh_m = sub["overhead_s"].mean()
        lat_m = float(sub["latency_e2e_s"].mean())
        if lab == "ar":
            rows.append(
                {
                    "mode": lab,
                    "draft": 0.0,
                    "verify": 0.0,
                    "overhead": 0.0,
                    "ar_only": lat_m,
                }
            )
        else:
            if pd.isna(draft_m) or pd.isna(ver_m):
                continue
            dft = float(draft_m)
            vrf = float(ver_m)
            oh = float(oh_m) if not pd.isna(oh_m) else max(0.0, lat_m - dft - vrf)
            rows.append({"mode": lab, "draft": dft, "verify": vrf, "overhead": oh, "ar_only": 0.0})

    if not rows:
        return

    mat = pd.DataFrame(rows).set_index("mode")
    fig, ax = plt.subplots(figsize=(10, 5))
    modes = mat.index.tolist()
    x = np.arange(len(modes))
    dcol = mat["draft"].values
    vcol = mat["verify"].values
    ocol = mat["overhead"].values
    arcol = mat["ar_only"].values if "ar_only" in mat.columns else np.zeros(len(modes))

    ax.bar(x, arcol, label="Greedy forward (AR only)", color="#7f7f7f")
    ax.bar(x, dcol, bottom=arcol, label="Drafting (draft forwards + cache)", color="#aec7e8")
    ax.bar(x, vcol, bottom=arcol + dcol, label="Verification (parallel block)", color="#ffbb78")
    ax.bar(x, ocol, bottom=arcol + dcol + vcol, label="Overhead (trim / dequant / refresh / other)", color="#98df8a")

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Time (s, mean across trials)")
    ax.set_title(
        f'Latency breakdown — prompt `{pid}`, speculative K={spec_k}\n'
        "(Overhead = end-to-end − draft − verify timers)"
    )
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_acceptance_vs_sequence_length(
    df: pd.DataFrame,
    out: Path,
) -> None:
    """Mean acceptance vs sequence length (prompt + generated tokens), by mode."""

    sub = df[df["benchmark_label"].astype(str) != "ar"].copy()
    d = enrich_sequence_length(sub)
    need = {"sequence_length_tokens", "acceptance_rate", "benchmark_label"}
    if not need.issubset(d.columns):
        return
    d = d.dropna(subset=["sequence_length_tokens", "acceptance_rate"])
    if d.empty:
        return

    nuniq = int(d["sequence_length_tokens"].nunique())
    nbin = min(8, max(3, nuniq))
    try:
        d["seq_bin"] = pd.qcut(d["sequence_length_tokens"], q=nbin, duplicates="drop")
    except ValueError:
        d["seq_bin"] = pd.cut(d["sequence_length_tokens"], bins=min(nbin, max(nuniq, 2)))

    agg = d.groupby(["seq_bin", "benchmark_label"], observed=True)["acceptance_rate"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in agg.columns:
        ax.plot(
            range(len(agg.index)),
            agg[col].values,
            marker="o",
            label=str(col),
            color=_color(str(col)),
        )
    ax.set_xticks(range(len(agg.index)))
    ax.set_xticklabels([str(i) for i in agg.index], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Mean acceptance rate")
    ax.set_xlabel("Sequence length bin (prompt + max_new_tokens quantiles)")
    ax.set_title("Acceptance vs context length (SnapKV-style: quality vs growing distraction)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_best_throughput_under_memory_budget(
    df: pd.DataFrame,
    out: Path,
    *,
    throughput_col: str = "tokens_per_sec",
    memory_col: str = "gpu_peak_memory_bytes_after_run",
    q: int = 3,
) -> None:
    """Bar chart: best mean throughput by benchmark_label within each peak-memory tertile."""

    if memory_col not in df.columns or throughput_col not in df.columns:
        return
    w = df.dropna(subset=[memory_col, throughput_col, "benchmark_label"])
    if len(w) < q * 2:
        return
    try:
        w = w.assign(mem_bucket=pd.qcut(w[memory_col], q=q, duplicates="drop"))
    except ValueError:
        return

    rows: list[dict[str, object]] = []
    for bucket, grp in w.groupby("mem_bucket", observed=True):
        best = grp.groupby("benchmark_label")[throughput_col].mean().sort_values(ascending=False)
        if best.empty:
            continue
        rows.append({"bucket": str(bucket), "label": best.index[0], "tps": float(best.iloc[0])})

    if not rows:
        return
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tab))
    ax.bar(x, tab["tps"].values, color=[_color(str(l)) for l in tab["label"]])
    ax.set_xticks(x)
    ax.set_xticklabels(tab["bucket"], rotation=15, ha="right", fontsize=7)
    ax.set_ylabel("Mean throughput (tokens/s)")
    ax.set_title("Best mode by peak-memory bucket (fixed budget proxy)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_draft_vs_verify_latency(
    df: pd.DataFrame,
    out: Path,
) -> None:
    """Stacked perception: mean draft vs verify time for speculative modes."""

    sub = df[df["benchmark_label"].astype(str) != "ar"].copy()
    for c in ("draft_latency_total_s", "verify_latency_total_s"):
        if c not in sub.columns:
            return
    sub = sub.dropna(subset=["draft_latency_total_s", "verify_latency_total_s"])
    if sub.empty:
        return

    agg = sub.groupby("benchmark_label")[["draft_latency_total_s", "verify_latency_total_s"]].mean()
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    agg.plot(kind="bar", stacked=True, ax=ax, color=["#aec7e8", "#ffbb78"])
    ax.set_ylabel("Seconds (mean)")
    ax.set_title("Draft vs verify latency (where sparse overhead may show up in draft phase)")
    ax.legend(loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_compression_frontier_throughput(df: pd.DataFrame, out: Path) -> None:
    """Scatter: effective compression (dense FP16 draft / stored draft bytes) vs throughput."""

    t = table_effective_compression_ratio(df)
    if t.empty:
        return
    sub = t.dropna(subset=["effective_compression_ratio", "tokens_per_sec"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for lab, g in sub.groupby("benchmark_label"):
        ax.scatter(
            g["effective_compression_ratio"],
            g["tokens_per_sec"],
            label=str(lab),
            alpha=0.45,
            s=28,
            c=_color(str(lab)),
            edgecolors="none",
        )
    ax.set_xlabel("Effective compression ratio (FP16 dense draft bytes / logical draft bytes)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Efficiency frontier: storage compression vs throughput (memory semantics)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_spec_fp16_theoretical_vs_actual_speedup(df: pd.DataFrame, out: Path) -> None:
    """Scatter: theoretical acceptance-model factor vs measured spec_fp16/AR throughput ratio."""

    tbl = table_verification_bottleneck_spec_fp16(df)
    if tbl.empty:
        return
    sub = tbl.dropna(subset=["theoretical_speedup_factor", "actual_speedup_tps_ratio"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lo = float(
        min(sub["theoretical_speedup_factor"].min(), sub["actual_speedup_tps_ratio"].min()) * 0.85
    )
    hi = float(
        max(sub["theoretical_speedup_factor"].max(), sub["actual_speedup_tps_ratio"].max()) * 1.05
    )
    ax.plot([lo, hi], [lo, hi], color="#999999", linestyle="--", linewidth=1.0, label="y=x (ideal match)")
    ax.scatter(
        sub["theoretical_speedup_factor"],
        sub["actual_speedup_tps_ratio"],
        alpha=0.35,
        s=22,
        c="#1f77b4",
        edgecolors="none",
    )
    ax.set_xlabel("Theoretical factor (acceptance model)")
    ax.set_ylabel("Actual throughput ratio (spec_fp16 / AR)")
    ax.set_title("Verification bottleneck proxy: model vs measured speedup")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_context_bucket_sparsification_lift(df: pd.DataFrame, out: Path) -> None:
    """Mean speedup vs AR for sparse modes, by context bucket (line plot)."""

    t = table_context_bucket_performance(df)
    if t.empty:
        return
    sparse_labels = {"spec_sparse", "spec_sparse_quant_memonly"}
    sub = t[t["benchmark_label"].astype(str).isin(sparse_labels)]
    if sub.empty:
        return

    order = ["short", "medium", "long"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lab, g in sub.groupby("benchmark_label"):
        g2 = g.set_index("context_bucket").reindex(order).reset_index()
        ax.plot(
            g2["context_bucket"],
            g2["mean_speedup_vs_ar"],
            marker="o",
            label=str(lab),
            color=_color(str(lab)),
        )
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlabel("Context bucket (see table for token ranges in this sweep)")
    ax.set_ylabel("Mean throughput speedup vs AR (paired trials)")
    ax.set_title("Sparsification lift vs context bucket (paired prompt×bucket×trial)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render_all_core_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write PNGs; return paths created."""

    out_dir = Path(out_dir)
    paths: list[Path] = []
    for name, fn in [
        ("throughput_vs_memory.png", lambda: plot_throughput_vs_memory_bytes(df, out_dir / "throughput_vs_memory.png")),
        ("pareto_throughput_vs_kv_mb.png", lambda: plot_pareto_throughput_vs_kv_mb(df, out_dir / "pareto_throughput_vs_kv_mb.png")),
        ("acceptance_vs_compression.png", lambda: plot_acceptance_vs_compression(df, out_dir / "acceptance_vs_compression.png")),
        ("acceptance_vs_sequence_length.png", lambda: plot_acceptance_vs_sequence_length(df, out_dir / "acceptance_vs_sequence_length.png")),
        ("throughput_by_context_bucket.png", lambda: plot_throughput_by_context_bucket(df, out_dir / "throughput_by_context_bucket.png")),
        ("best_throughput_under_memory_budget.png", lambda: plot_best_throughput_under_memory_budget(df, out_dir / "best_throughput_under_memory_budget.png")),
        ("ablation_modes.png", lambda: plot_ablation_modes(df, out_dir / "ablation_modes.png")),
        ("stacked_latency_single_prompt.png", lambda: plot_stacked_latency_single_prompt(df, out_dir / "stacked_latency_single_prompt.png")),
        ("draft_vs_verify_latency.png", lambda: plot_draft_vs_verify_latency(df, out_dir / "draft_vs_verify_latency.png")),
        ("compression_frontier_throughput.png", lambda: plot_compression_frontier_throughput(df, out_dir / "compression_frontier_throughput.png")),
        ("spec_fp16_theoretical_vs_actual_speedup.png", lambda: plot_spec_fp16_theoretical_vs_actual_speedup(df, out_dir / "spec_fp16_theoretical_vs_actual_speedup.png")),
        ("context_bucket_sparsification_lift.png", lambda: plot_context_bucket_sparsification_lift(df, out_dir / "context_bucket_sparsification_lift.png")),
    ]:
        try:
            fn()
            p = out_dir / name
            if p.is_file():
                paths.append(p)
        except Exception:
            continue
    return paths
