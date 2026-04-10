# Phase 16: interpreting benchmark results

This document pairs with **`mlsys-kv benchmark-report`** (or
`mlsys_kv.benchmarks.analysis.generate_phase16_report`), which reads **schema v2** CSVs and writes:

- **`INDEX.md`** — main narrative (embedded PNGs, failure summaries, interpretation blocks)
- **`HOW_TO_VIEW.md`** — file map and how to open outputs
- **`tables/`** — core: `summary_by_mode.csv`, `speedup_vs_ar_paired.csv` (optional **`scipy`** for p-values); **extended:**
  `joint_sparse_quant_vs_sparse_only.csv`, `effective_compression_ratio_rows.csv`, `effective_compression_by_mode.csv`,
  `verification_bottleneck_spec_fp16_rows.csv`, `verification_bottleneck_by_spec_k.csv`,
  `context_bucket_performance.csv`, `context_bucket_prompt_token_ranges.csv`
- **`figures/`** — Pareto, stacked latency, acceptance vs length, **compression frontier**, **spec_fp16 theoretical vs actual speedup**, **context-bucket sparsification lift**, etc.

## 1. Split analysis by quantization semantics

| `quantization_type` (CSV) | Meaning in this codebase |
|---------------------------|---------------------------|
| `none` | No INT packing on the draft KV path (FP16-equivalent or pure FP16 draft). |
| `memory_only` | Draft KV is stored in low-bit packed form; attention uses dequantized values — **reduces bytes moved / footprint**, not proven matmul cycle reduction. |
| `runtime_accelerated` | Reserved for future native low-bit attention (not in current prototype). |

**Plots and captions:** If your sweep has **only** `none` and `memory_only`, state explicitly that **quantization does not, by itself, prove faster attention** — only lower memory traffic or smaller KV unless you add microbenchmarks of the attention kernel.

## 2. What the core plots show

- **Throughput vs memory:** Explores the **memory–throughput frontier** using peak allocated memory. Good for comparing **sparse** and **sparse+quant** under similar peaks.
- **Acceptance vs compression:** Uses `verifier_kv / draft_kv` as a **size proxy**. Drops in acceptance point to verifier–draft disagreement or overly aggressive pruning — **not** “compression failed” in a codec sense.
- **Throughput by context bucket:** Isolates **short vs medium vs long** regime effects (KV growth).
- **Ablations across modes:** High-level **mean tokens/s** by `benchmark_label`.
- **Draft vs verify latency:** Locates **sparse scoring / refresh** overhead (often draft-heavy) vs **parallel verify** time.

## 3. Extended analyses (Phase 16 revision)

### 3.1 Joint effect (sparse × quantization)

`tables/joint_sparse_quant_vs_sparse_only.csv` pairs **`spec_sparse`** and **`spec_sparse_quant_memonly`** at the same
`(prompt_id, context_bucket, spec_k, sparsity_budget, trial_index)` and tests whether **acceptance** differs by
`quant_bits_effective`. A **positive** `mean_delta_sparse_only_minus_joint` means sparse-only achieves higher acceptance
than sparse+quant — consistent with low-bit packing disturbing draft–verifier agreement (memory-only semantics).

### 3.2 Effective compression ratio

Row-level ratios are in `tables/effective_compression_ratio_rows.csv`:

- **Numerator:** dense FP16 draft bytes from the matching **`spec_fp16`** cell (`fp16_dense_draft_bytes_baseline`).
- **Denominator:** `logical_draft_kv_bytes` for the row (already reflects sparsity + packing).
- **Ratio** = baseline / stored — maps **storage compression** to throughput in `figures/compression_frontier_throughput.png`.

This is **not** the same as “attention ran in INT4”; it is consistent with **memory-only** KV in the prototype.

### 3.3 Verification bottleneck (spec_fp16 vs AR)

`tables/verification_bottleneck_spec_fp16_rows.csv` compares:

- **Actual:** `tokens_per_sec(spec_fp16) / tokens_per_sec(AR)` on the same `(prompt, bucket, trial)`.
- **Theoretical proxy:** \((1-\alpha^{K+1}) / ((K+1)(1-\alpha))\) with \(\alpha=\) acceptance rate (qualitative).

`system_overhead_ratio` = actual / theoretical. Values **≪ 1** suggest **draft/verify/sync** costs dominate vs the idealized
acceptance model. See `figures/spec_fp16_theoretical_vs_actual_speedup.png`.

### 3.4 Context buckets

Buckets are **sweep-defined** (YAML `short_token_max`, `medium_token_max`), not fixed 512/1024 cuts.
`tables/context_bucket_prompt_token_ranges.csv` summarizes empirical `prompt_len_tokens` per label in **your** CSV.
`tables/context_bucket_performance.csv` and `figures/context_bucket_sparsification_lift.png` report paired speedup vs AR
for sparse modes across buckets.

## 4. Failure-analysis tables (in the generated report)

- **Acceptance drops vs `spec_fp16`:** Same prompt, `spec_k`, bucket — flags where speculative paths lose acceptance vs a strong FP16 baseline.
- **Sparse overhead:** Draft latency fraction of draft+verify for `spec_sparse`.
- **Quant memory vs speed:** Pivoted FP16 vs memory-only quant — compare **draft bytes** and **tokens/s**; interpret memory wins without overstating **speed** wins.
- **Best throughput under memory buckets:** Tertiles of peak VRAM — which **label** wins when memory is constrained.
- **Joint vs components:** When `spec_sparse_quant_memonly` rows exist, compare to sparse-only and quant-only.

## 5. What must be verified before claiming results

1. **`assert_semantics_consistent`** (used by the report): CSV `quantization_type` and `benchmark_label` match `experiment_schema` rules.
2. **Plots** distinguish **memory-only** from any future **runtime-accelerated** rows (legend / faceting).
3. **Conclusions** match **Phase 13** semantics: packed KV is not the same as faster attention.
4. **Theoretical speedup** column is a **diagnostic** — not a claim that the implementation achieves an information-theoretic bound.

## 6. Prototype limitations (be explicit in the paper)

- **Memory-only quantization:** Pack/unpack and dequant before matmul — benefits are **dominated by memory traffic and capacity**, not instruction count in attention.
- **Sparse retention:** SnapKV-style heuristic; acceptance and overhead depend on **prompt**, **K**, and **budget**.
- **Single-GPU, fixed model:** Throughput numbers are **hardware- and model-specific**.

## 7. Future work for true low-bit attention acceleration

- Fused INT4/INT8 **GEMM** or **FlashAttention**-style kernels on quantized K/V without full dequant to FP16.
- **Co-design** with sparsity (block-sparse attention on retained tokens).
- Calibrated **per-layer** bitwidth and **runtime** vs **memory** Pareto curves.

## Command

```bash
PYTHONPATH=src python -m mlsys_kv.cli benchmark-report \
  --csv results/sweep_full_schema_v2.csv \
  --out results/phase16_run1
# Optional: fix prompt/K for stacked latency figure
#   --stacked-prompt-id mt-010 --stacked-spec-k 5
```

Install **`pip install "scipy>=1.11"`** (or `pip install -e ".[analysis]"`) for paired **p-values** in `tables/speedup_vs_ar_paired.csv`.
