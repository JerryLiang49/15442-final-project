"""Generate Phase 16 report: tables/, figures/, markdown index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.analysis.aggregates import build_summary_table
from benchmarks.analysis.failure_summaries import build_failure_section
from benchmarks.analysis.extended_tables import (
    infer_context_bucket_token_ranges,
    table_context_bucket_performance,
    table_effective_compression_ratio,
    table_joint_sparse_quant_vs_sparse_only,
    table_mean_compression_and_throughput_by_mode,
    table_verification_bottleneck_spec_fp16,
    table_verification_bottleneck_summary,
)
from benchmarks.analysis.failure_tables import (
    table_acceptance_by_mode_spec_k,
    table_best_throughput_under_memory_cap,
    table_joint_mode_vs_components,
    table_quant_memory_vs_speed,
    table_sparse_overhead_fraction,
    table_where_acceptance_dropped,
)
from benchmarks.analysis.labels import QUANT_HONESTY_FOOTNOTE
from benchmarks.analysis.loader import load_benchmark_csv
from benchmarks.analysis.plots import (
    plot_stacked_latency_single_prompt,
    render_all_core_plots,
)
from benchmarks.analysis.semantics import (
    assert_semantics_consistent,
    quantization_scope_summary,
)
from benchmarks.analysis.stats import summarize_speedup_and_pvalues


def _df_to_md(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "*No rows.*\n\n"
    d = df.head(max_rows).copy()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in d.columns) + " |")
    out = "\n".join(lines) + "\n\n"
    if len(df) > max_rows:
        out += f"*(Showing first {max_rows} of {len(df)} rows.)*\n\n"
    return out


def _caption_block(scope: dict[str, Any], issues: list[str]) -> str:
    lines = [
        "## Quantization scope (read before interpreting plots)",
        "",
        scope.get("caption_note", ""),
        "",
        "- **Counts by `quantization_type`:** "
        + str(scope.get("quantization_type_counts", {})),
        "",
        "- **Counts by `benchmark_label`:** " + str(scope.get("benchmark_label_counts", {})),
        "",
        "### Honest labeling (tables + captions)",
        "",
        QUANT_HONESTY_FOOTNOTE,
        "",
    ]
    if scope.get("has_runtime_accelerated_quant"):
        lines.extend(
            [
                "> **This sweep includes `runtime_accelerated` rows** — separate from memory-only plots.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> **No `runtime_accelerated` quantization in CSV.** INT KV is **Memory-Only** in this prototype.",
                "",
            ]
        )
    if issues:
        lines.append("### Semantics warnings")
        lines.append("")
        for i in issues:
            lines.append(f"- ⚠ {i}")
        lines.append("")
    else:
        lines.append("### Semantics check")
        lines.append("")
        lines.append("Recomputed `quantization_type` and `benchmark_label` match CSV for all rows.")
        lines.append("")
    return "\n".join(lines)


def generate_phase16_report(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    title: str = "Phase 16 benchmark analysis",
    stacked_prompt_id: str | None = None,
    stacked_spec_k: int | None = None,
) -> Path:
    """Write ``INDEX.md``, ``tables/*``, ``figures/*``; return path to ``INDEX.md``."""

    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_benchmark_csv(csv_path, ok_only=True)
    issues = assert_semantics_consistent(df)
    scope = quantization_scope_summary(df)

    fig_paths = render_all_core_plots(df, fig_dir)

    if stacked_prompt_id is not None or stacked_spec_k is not None:
        plot_stacked_latency_single_prompt(
            df,
            fig_dir / "stacked_latency_single_prompt.png",
            prompt_id=stacked_prompt_id,
            spec_k=stacked_spec_k,
        )
        if (fig_dir / "stacked_latency_single_prompt.png").is_file():
            fig_paths = [p for p in fig_paths if p.name != "stacked_latency_single_prompt.png"]
            fig_paths.append(fig_dir / "stacked_latency_single_prompt.png")

    summary = build_summary_table(df)
    summary.to_csv(tables_dir / "summary_by_mode.csv", index=False)

    speed_tbl = summarize_speedup_and_pvalues(df)
    if not speed_tbl.empty:
        speed_tbl.to_csv(tables_dir / "speedup_vs_ar_paired.csv", index=False)

    acc_tbl = table_acceptance_by_mode_spec_k(df)
    drop_tbl = table_where_acceptance_dropped(df)
    sparse_tbl = table_sparse_overhead_fraction(df)
    qms_tbl = table_quant_memory_vs_speed(df)
    memcap_tbl = table_best_throughput_under_memory_cap(df)
    joint_tbl = table_joint_mode_vs_components(df)

    joint_sparse_tbl = table_joint_sparse_quant_vs_sparse_only(df)
    eff_comp_long = table_effective_compression_ratio(df)
    eff_comp_summary = table_mean_compression_and_throughput_by_mode(df)
    verif_tbl = table_verification_bottleneck_spec_fp16(df)
    verif_sum = table_verification_bottleneck_summary(df)
    ctx_perf = table_context_bucket_performance(df)
    bucket_ranges = infer_context_bucket_token_ranges(df)

    if not joint_sparse_tbl.empty:
        joint_sparse_tbl.to_csv(tables_dir / "joint_sparse_quant_vs_sparse_only.csv", index=False)
    if not eff_comp_long.empty:
        eff_comp_long.to_csv(tables_dir / "effective_compression_ratio_rows.csv", index=False)
    if not eff_comp_summary.empty:
        eff_comp_summary.to_csv(tables_dir / "effective_compression_by_mode.csv", index=False)
    if not verif_tbl.empty:
        verif_tbl.to_csv(tables_dir / "verification_bottleneck_spec_fp16_rows.csv", index=False)
    if not verif_sum.empty:
        verif_sum.to_csv(tables_dir / "verification_bottleneck_by_spec_k.csv", index=False)
    if not ctx_perf.empty:
        ctx_perf.to_csv(tables_dir / "context_bucket_performance.csv", index=False)
    if not bucket_ranges.empty:
        bucket_ranges.to_csv(tables_dir / "context_bucket_prompt_token_ranges.csv", index=False)

    failures = build_failure_section(df)

    rel_figs = [str(p.relative_to(out_dir)) for p in sorted(fig_paths)]

    md: list[str] = [
        f"# {title}",
        "",
        f"**Source CSV:** `{csv_path.resolve()}`  ",
        f"**Rows (status=ok):** {len(df)}",
        "",
        "See **`HOW_TO_VIEW.md`** in this folder for paths and viewing order.",
        "",
        _caption_block(scope, issues),
        "",
        "## Report-ready layout",
        "",
        "```text",
        f"{out_dir.name}/",
        "  INDEX.md           # this file",
        "  HOW_TO_VIEW.md     # where plots live + how to open them",
        "  tables/            # CSV summaries (Excel, pandas, paper tables)",
        "  figures/           # PNG figures for slides/paper",
        "```",
        "",
        "## Summary table (memory-only labels in `display_name`)",
        "",
        _df_to_md(summary, max_rows=30),
        "",
        "### Speedup vs AR (paired prompts × buckets × trials)",
        "",
        _df_to_md(speed_tbl if not speed_tbl.empty else pd.DataFrame()),
        "",
        "## Figures",
        "",
    ]
    for rp in rel_figs:
        md.append(f"![{rp}]({rp})")
        md.append("")

    md.extend(
        [
            "### Figure guide",
            "",
            "| File | What it shows |",
            "|------|----------------|",
            "| `figures/pareto_throughput_vs_kv_mb.png` | **Paper core:** throughput vs **verifier KV (MB)**; color=mode; point size ∝ K; dashed lines connect sweep points per mode. |",
            "| `figures/stacked_latency_single_prompt.png` | **Where time goes:** AR vs draft / verify / residual overhead for one long prompt. |",
            "| `figures/acceptance_vs_sequence_length.png` | Acceptance vs binned **prompt+gen** length. |",
            "| `figures/acceptance_vs_compression.png` | Acceptance vs **compression strength** (verifier/draft KV ratio). |",
            "| `figures/throughput_by_context_bucket.png` | Throughput by **short/medium/long** with **±std** across trials. |",
            "| `figures/best_throughput_under_memory_budget.png` | Best mode per **peak-VRAM tertile** (fixed budget proxy). |",
            "| `figures/ablation_modes.png` | Global ablation with **error bars** (trial std). |",
            "| `figures/throughput_vs_memory.png` | Tokens/s vs **peak torch memory** (hardware footprint). |",
            "| `figures/compression_frontier_throughput.png` | **Compression frontier:** dense FP16 draft baseline / stored draft bytes vs throughput. |",
            "| `figures/spec_fp16_theoretical_vs_actual_speedup.png` | **Verification model:** acceptance-based factor vs measured spec_fp16/AR ratio. |",
            "| `figures/context_bucket_sparsification_lift.png` | Sparse modes: mean speedup vs AR by **context bucket**. |",
            "",
            "## Interpretation (quantization + sparsity)",
            "",
            "- **Joint effect (sparse + quant):** If `sparse_quant` acceptance is **materially lower** than `sparse_only` "
            "at the same (prompt, bucket, K, sparsity, trial), low-bit **memory-only** packing may be disturbing "
            "draft–verifier agreement — see `tables/joint_sparse_quant_vs_sparse_only.csv`.",
            "",
            "- **Effective compression:** `effective_compression_ratio` = FP16 **dense draft** bytes (from `spec_fp16` at the "
            "same cell) divided by **logical** draft bytes. This maps **storage** compression to throughput; it does **not** "
            "claim faster attention unless `quantization_type` indicates runtime acceleration.",
            "",
            "- **Verification bottleneck:** The scatter compares a simple **acceptance-based factor** to the measured "
            "**spec_fp16/AR** throughput ratio. Points **below** the y=x line suggest **system overhead** (draft/verify/sync, "
            "data movement) vs the idealized model.",
            "",
            "- **Context buckets:** This sweep labels **short / medium / long** using YAML thresholds (see "
            "`tables/context_bucket_prompt_token_ranges.csv` for empirical token ranges in **this** CSV). "
            "Compare to fixed 512/1024 token cuts only if you re-bucket in post-processing.",
            "",
            "## Separation of effects (honest)",
            "",
            "- **Sparse-driven runtime:** draft latency fraction, acceptance vs length, sparse rows in tables.",
            "- **Quantization-driven memory:** lower `logical_draft_kv_bytes` for Memory-Only INT KV; speed may not follow (dequant overhead).",
            "- **True runtime gains:** only claim if `quantization_type` / metrics JSON indicate accelerated kernels (not default here).",
            "",
            "## Failure-analysis summaries",
            "",
            "### Acceptance loss",
            "",
            failures["acceptance_loss"],
            "",
            "### Sparse overhead",
            "",
            failures["sparse_overhead"],
            "",
            "### Quantization overhead (memory-only vs FP16 spec)",
            "",
            failures["quantization_overhead"],
            "",
            "### Joint method tradeoffs",
            "",
            failures["joint_tradeoffs"],
            "",
            "## Detailed tables",
            "",
            "### Mean acceptance by mode and K",
            "",
            _df_to_md(acc_tbl),
            "",
            "### Acceptance drops vs `spec_fp16`",
            "",
            _df_to_md(drop_tbl),
            "",
            "### Sparse draft-latency share",
            "",
            _df_to_md(sparse_tbl),
            "",
            "### FP16 vs memory-only quant (pivot)",
            "",
            _df_to_md(qms_tbl),
            "",
            "### Best throughput under memory tertiles",
            "",
            _df_to_md(memcap_tbl),
            "",
            "### Joint vs components",
            "",
            _df_to_md(joint_tbl),
            "",
            "### Joint sparse×quant vs sparse-only (acceptance)",
            "",
            _df_to_md(joint_sparse_tbl if not joint_sparse_tbl.empty else pd.DataFrame()),
            "",
            "### Effective compression ratio (rows + by-mode means)",
            "",
            _df_to_md(eff_comp_summary if not eff_comp_summary.empty else pd.DataFrame(), max_rows=40),
            "",
            "*Full row-level table:* `tables/effective_compression_ratio_rows.csv`.",
            "",
            "### Verification bottleneck (spec_fp16 vs AR)",
            "",
            _df_to_md(verif_sum if not verif_sum.empty else pd.DataFrame()),
            "",
            "*Per-row ratios:* `tables/verification_bottleneck_spec_fp16_rows.csv`.",
            "",
            "### Context bucket: empirical prompt lengths + performance",
            "",
            _df_to_md(bucket_ranges if not bucket_ranges.empty else pd.DataFrame()),
            "",
            _df_to_md(ctx_perf if not ctx_perf.empty else pd.DataFrame(), max_rows=60),
            "",
            "---",
            "",
            "Interpretation guide: `docs/PHASE16_ANALYSIS.md`.",
            "",
        ]
    )

    index_path = out_dir / "INDEX.md"
    index_path.write_text("\n".join(md), encoding="utf-8")

    how = f"""# How to view Phase 16 outputs

All paths below are relative to the report directory you passed to `benchmark-report` (e.g. `results/phase16_run1/`).

## Quick open

- **Start here:** `INDEX.md` — full narrative, embedded figures, and tables.
- **Paper figures:** `figures/` — PNGs at 150 DPI; import into Google Slides, Keynote, or LaTeX (`\\includegraphics`).
- **Numbers for tables:** `tables/summary_by_mode.csv` and `tables/speedup_vs_ar_paired.csv` — open in Excel, pandas, or Overleaf `csvsimple`.

## File map

| Path | Purpose |
|------|---------|
| `INDEX.md` | Main report (semantics, captions, failure summaries, embedded images). |
| `tables/summary_by_mode.csv` | Per-mode means + **display_name** (Memory-Only where applicable) + speedup vs AR + optional **p-values** (needs `scipy`). |
| `tables/speedup_vs_ar_paired.csv` | Paired speedup and **p_value_diff_vs_ar** per non-AR mode. |
| `tables/joint_sparse_quant_vs_sparse_only.csv` | Paired acceptance: sparse-only minus sparse+quant by **quant_bits**. |
| `tables/effective_compression_*.csv` | Storage compression vs dense FP16 draft baseline; frontier numbers. |
| `tables/verification_bottleneck_*.csv` | spec_fp16 vs AR: actual / theoretical acceptance-model factor. |
| `tables/context_bucket_*.csv` | Empirical token ranges per bucket + speedup vs AR by bucket. |
| `figures/pareto_throughput_vs_kv_mb.png` | **Pareto / frontier** plot (throughput vs KV MB). |
| `figures/compression_frontier_throughput.png` | Compression ratio vs throughput. |
| `figures/spec_fp16_theoretical_vs_actual_speedup.png` | Verification bottleneck scatter. |
| `figures/context_bucket_sparsification_lift.png` | Sparse modes: speedup vs AR by context bucket. |
| `figures/stacked_latency_single_prompt.png` | Latency breakdown for longest prompt (override with CLI flags). |
| `figures/acceptance_vs_sequence_length.png` | Acceptance vs context length bins. |
| Other `figures/*.png` | Context buckets, ablations, memory budget, draft vs verify, etc. |

## Regenerate

```bash
PYTHONPATH=src python -m mlsys_kv.cli benchmark-report --csv path/to/sweep.csv --out results/my_phase16
```

Optional: `--stacked-prompt-id mt-010 --stacked-spec-k 5` to fix the stacked-latency prompt/K.

## Viewing on GitHub / VS Code

- Preview `INDEX.md` in VS Code (Markdown preview) or push to GitHub — **relative image links** resolve to `figures/`.
"""
    (out_dir / "HOW_TO_VIEW.md").write_text(how, encoding="utf-8")

    return index_path
