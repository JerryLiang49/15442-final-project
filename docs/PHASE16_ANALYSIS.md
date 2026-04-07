# Phase 16: interpreting benchmark results

This document pairs with **`mlsys-kv benchmark-report`** (or
`mlsys_kv.benchmarks.analysis.generate_phase16_report`), which reads **schema v2** CSVs and writes:

- **`INDEX.md`** — main narrative (embedded PNGs, failure summaries)
- **`HOW_TO_VIEW.md`** — file map and how to open outputs
- **`tables/`** — `summary_by_mode.csv`, `speedup_vs_ar_paired.csv` (optional `scipy` for p-values)
- **`figures/`** — all PNGs (Pareto, stacked latency, acceptance vs length, etc.)

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

## 3. Failure-analysis tables (in the generated report)

- **Acceptance drops vs `spec_fp16`:** Same prompt, `spec_k`, bucket — flags where speculative paths lose acceptance vs a strong FP16 baseline.
- **Sparse overhead:** Draft latency fraction of draft+verify for `spec_sparse`.
- **Quant memory vs speed:** Pivoted FP16 vs memory-only quant — compare **draft bytes** and **tokens/s**; interpret memory wins without overstating **speed** wins.
- **Best throughput under memory buckets:** Tertiles of peak VRAM — which **label** wins when memory is constrained.
- **Joint vs components:** When `spec_sparse_quant_memonly` rows exist, compare to sparse-only and quant-only.

## 4. What must be verified before claiming results

1. **`assert_semantics_consistent`** (used by the report): CSV `quantization_type` and `benchmark_label` match `experiment_schema` rules.
2. **Plots** distinguish **memory-only** from any future **runtime-accelerated** rows (legend / faceting).
3. **Conclusions** match **Phase 13** semantics: packed KV is not the same as faster attention.

## 5. Prototype limitations (be explicit in the paper)

- **Memory-only quantization:** Pack/unpack and dequant before matmul — benefits are **dominated by memory traffic and capacity**, not instruction count in attention.
- **Sparse retention:** SnapKV-style heuristic; acceptance and overhead depend on **prompt**, **K**, and **budget**.
- **Single-GPU, fixed model:** Throughput numbers are **hardware- and model-specific**.

## 6. Future work for true low-bit attention acceleration

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
