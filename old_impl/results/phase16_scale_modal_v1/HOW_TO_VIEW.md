# How to view Phase 16 outputs

All paths below are relative to the report directory you passed to `benchmark-report` (e.g. `results/phase16_run1/`).

## Quick open

- **Start here:** `INDEX.md` — full narrative, embedded figures, and tables.
- **Paper figures:** `figures/` — PNGs at 150 DPI; import into Google Slides, Keynote, or LaTeX (`\includegraphics`).
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
