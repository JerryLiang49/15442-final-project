# Phase 15 benchmark harness

## Benchmark labels (`benchmark_label` column)

YAML `modes` may use **canonical** names (`autoregressive`, …) or **aliases** mapped in `experiment_schema.canonical_sweep_mode`:

| Label | Canonical mode |
|-------|----------------|
| `ar` | `autoregressive` |
| `spec_fp16` | `speculative_fp16` |
| `spec_quant_memonly` | `quant_only` |
| `spec_sparse` | `sparse_only` |
| `spec_sparse_quant_memonly` | `sparse_quant` |

Reserved for future native accelerated quant (not implemented): `spec_quant_runtime`, `spec_sparse_quant_runtime`.

## Strict labeled grid

Default config flag: **`strict_labeled_grid: true`**.

Each mode only varies hyperparameters that actually affect decoding for that mode. Ignored dimensions are fixed to sentinels so rows are comparable and resume keys are not duplicated (for example, autoregressive is **not** multiplied by every `quant_bits` value).

Sentinels in strict mode:

- **Autoregressive:** `spec_k=0` (draft K is not used), `sparsity_budget=0.0`, `quant_bits_requested=-1`.
- **Speculative FP16:** `sparsity_budget=0.0`, `quant_bits_requested=16` (metadata only; draft is FP16).
- **Quant-only:** `sparsity_budget=0.0`.
- **Sparse-only:** `quant_bits_requested=16` (FP16 retained KV).
- **Sparse + quant:** full product of `k_values × sparsity_budgets × quant_bits`.

Set **`strict_labeled_grid: false`** only for legacy full-factorial experiments (may include ambiguous duplicate runs).

## Metadata columns (CSV v2)

- **`is_parallel_verification`:** `true` for all speculative modes (Phase 9 parallel block verify); `false` for AR.
- **`quantization_type`:** `none` or `memory_only` per Phase 13 (`quantization_type_for_row`).
- **`memory_throughput_gb_s` / `effective_memory_bandwidth_gb_s`:** \((W + KV) / t_\text{per-token}\) with \(t_\text{per-token} = \texttt{latency_e2e_s} / \texttt{new\_tokens}\).
- **`gpu_peak_memory_bytes_after_run`:** from `torch.cuda.max_memory_allocated()` (via `max_memory_allocated_bytes`).
- **`draft_latency_total_s` / `verify_latency_total_s`:** from speculative metrics (empty for AR).

## Staged runs

1. **Pre-sweep gate:** `python scripts/benchmark_presweep_gate.py` (pytest `benchmark_gate` + tiny local sweep; optional Modal).
2. **Stage 1:** `configs/benchmark_stage1_local.yaml` then `configs/benchmark_stage1_modal.yaml`.
3. **Scale:** `benchmark_medium.yaml`, `benchmark_full.yaml` (Modal: `*_modal.yaml`).

## Resume and CSV versioning

- **`mode`** in the CSV stays **canonical** for stable resume keys.
- Existing CSVs without `benchmark_label` are rejected (schema v2).

## Processed outputs

After each sweep, **`processed_json`** (default `outputs/benchmarks/processed/{sweep_id}_rollup.json`) holds rollups by `benchmark_label`. Raw rows remain in **`raw_jsonl_path`**.
