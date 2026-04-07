# Benchmark readiness (Phase 14)

Gate command (run locally or in CI before large sweeps):

```bash
cd 15442-final-project
PYTHONPATH=src pytest -m benchmark_gate --slow -q
```

## Checklist

| Item | How it is enforced |
|------|---------------------|
| Autoregressive baseline | `test_phase14_autoregressive_baseline` |
| Speculative FP16, quant-only, sparse-only, sparse-quant match greedy AR | `test_phase14_speculative_matches_ar_metrics_verifier_and_schema` with `verify_match=True` |
| Verifier KV length equals final token count | Same test (`verifier_cache_seq_len_hf`) |
| Metrics JSON matches frozen schema | `assert_speculative_metrics_jsonable_shape` |
| Acceptance rate in `[0, 1]` | Same test |
| Quantization labeled **memory-only** (not runtime-accelerated) | `draft_kv_quantization_semantics`, `draft_runtime_accelerated_quant_attention`, cache `kv_quantization_semantics` |
| Sparse draft cache `reset()` between prompts when reusing an instance | `test_phase14_sparse_draft_cache_reset_between_prompts_on_reused_cache` |
| Second decode on same `SpeculativeDecoder` | `test_phase14_two_sequential_decodes_same_decoder_instance` |
| `SpeculativeMetrics` ↔ `SPECULATIVE_METRICS_JSON_KEYS` alignment | `test_phase14_frozen_speculative_metrics_key_set_covers_schema` |

## Frozen vocabulary (do not rename casually)

- **Sweep modes** (`SWEEP_LOGICAL_MODES` in `src/mlsys_kv/benchmarks/experiment_schema.py`): `autoregressive`, `speculative_fp16`, `quant_only`, `sparse_only`, `sparse_quant`.
- **Draft cache mode strings** (`DraftCacheMode.value` / `DRAFT_CACHE_MODE_VALUES`): `fp16`, `quant_only`, `sparse_only`, `sparse_quant`.
- **Speculative metrics JSON keys**: `SPECULATIVE_METRICS_JSON_KEYS` in `experiment_schema.py` — must match `SpeculativeMetrics.to_jsonable()`.
- **CSV columns**: `BENCHMARK_CSV_FIELDNAMES` in `experiment_schema.py`.
- **Schema version**: `EXPERIMENT_SCHEMA_VERSION` (bump if you add/remove metric keys).

## Known limitations

- **Runtime-accelerated quantization** (attention natively on INT8/INT4 KV) is **deferred**; draft path uses **memory-only** quantization with dequantization before HF attention (`kv_quantization_semantics: memory_only`).
- Speculative path is **batch size 1**; verifier/draft invariants are validated for that setting.
- Models with sliding-window caches use crop/repair fallbacks; GPT-2 smoke tests use standard `DynamicCache` layers.

## Phase 15

After this gate passes, sweeps should log `EXPERIMENT_SCHEMA_VERSION` and avoid renaming `SPECULATIVE_METRICS_JSON_KEYS` or CSV fields without a version bump and migration notes.
