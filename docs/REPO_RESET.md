# Repository reset plan (cleanup before next phases)

The **Phase 0** narrative (why rewrite, what stayed, verification) is in **`docs/RESET_PLAN.md`**. This file is a concise **inventory** and migration checklist.

## Snapshot (`old_impl/`)

- **`old_impl/mlsys_kv/`** — legacy full package (speculative decoding, cache, multi-mode benchmark runner). Removed from active `src/mlsys_kv/`.
- **`old_impl/configs/`** — factorial benchmark YAMLs and `speculative.yaml` (multi-mode grids). Active repo keeps small AR-only configs in `configs/` (e.g. `benchmark_smoke_ar.yaml`).
- **`old_impl/tests/`**, **`old_impl/scripts/`** — tests and scripts that depended on the legacy stack.

**Active installs** include **`src/mlsys_kv/`** (CLI + infra + models + datasets) **and** top-level **`src/cache/`**, **`src/decoding/`**, **`src/benchmarks/`** (AR-only sweep).

## What counts as “legacy” (moved or to retire from main later)

These are tied to the **previous** design; they assume:

| Area | Legacy behavior |
|------|------------------|
| **Decoding** | `decoding/speculative.py` — per-round draft cache reconstruction from verifier snapshots; K-step draft + block verify with same model. |
| **Cache** | Separate draft backends (`kv_cache_sparse*`, `kv_cache_quantized`, `draft_factory`, `sparse_hf_integration`, …) built around that speculative loop. |
| **Quantization** | Memory-only INT paths with dequant before standard attention; documented in Phase 13 semantics. |
| **Benchmarks** | `experiment_runner` + `experiment_schema` + strict grid for `sparse_only` / `sparse_quant` / `quant_only` labels. |
| **Configs** | `benchmark_full*.yaml`, `benchmark_scale*.yaml`, `speculative.yaml`, etc., referencing those modes. |

## What to keep in the main tree (reusable, mode-agnostic)

These are good candidates to **preserve or lightly refactor** rather than delete:

- **Infra:** `infra/config.py`, `infra/seed.py`, `infra/device.py`, `infra/env_meta.py`, logging helpers.
- **Models:** `models/hf_loader.py` — tokenizer/model load; adjust dtypes/devices as needed.
- **Benchmarks (skeleton):** `benchmarks/timer.py`, `benchmarks/memory.py`, `benchmarks/metrics.py`, `context_buckets.py`, `datasets/mt_bench.py` — trim coupling to legacy modes when you redefine schema.
- **Analysis:** `benchmarks/analysis/*` — plots/loaders can stay useful if CSV schema stays compatible or you add adapters.
- **Modal:** `modal_sweep.py` (root) — Volume + remote sweep pattern; wire to new runner when ready.
- **CLI:** `cli.py` — evolve subcommands to new entrypoints.
- **Autoregressive baseline:** `decoding/autoregressive.py` — often still the right AR reference.

## Target layout (suggested)

After migration, a cleaner package shape might look like:

```text
project_root/
  old_impl/                    # frozen snapshot (this reset)
  src/mlsys_kv/
    infra/                     # config, device, seed, logging
    models/                    # HF load, tokenizer
    cache/                     # new KV abstractions (or slim facades)
    decoding/                  # new speculative / AR paths
    kernels/                   # optional fused ops / quant kernels
    serving/                   # optional batching, server glue
    benchmarks/                # runner, schema, CSV, analysis
  configs/
  scripts/
  tests/
  docs/
```

Top-level packages under `src/`: **`cache/`** (FP16 + base), **`decoding/`** (autoregressive), **`benchmarks/`** (runner shell + analysis; **AR-only** sweep in active), **`kernels/`**, **`serving/`** (stubs). See `RESET_PLAN.md`.

## Checklist for the next phase

1. [ ] Implement new decoding + cache integration **without** snapshot reconstruction each round (or document why you keep it).
2. [ ] Define a new **benchmark schema** (or version bump) if mode names or columns change.
3. [ ] Port tests: start from `old_impl/tests/` and shrink to the new surface area.
4. [ ] Update `README.md`, `docs/BENCHMARK_*.md`, and remove or redirect legacy commands.
5. [ ] When stable, **delete** superseded modules from `src/mlsys_kv/` (not from `old_impl/` — keep the archive).

## Phase 16 / results

Existing **`results/`**, **`results_gpt2/`**, and generated Phase 16 folders are **not** moved into `old_impl/`; they remain project artifacts. Re-run `benchmark-report` after new CSV schemas exist.
