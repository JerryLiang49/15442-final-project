# Reset plan — QuantSpec-style rebuild

## What moved

- **Full legacy package** under `old_impl/mlsys_kv/` (self-speculative loop, draft/verifier caches, sparse/quant HF integration, legacy benchmark runner with speculative modes).
- **Legacy configs** (`benchmark_*.yaml`, `speculative.yaml`) → `old_impl/configs/` (active tree keeps `base.yaml`, `baseline.yaml`, and small **autoregressive-only** sweep YAMLs).
- **Legacy tests and scripts** that depended on speculative/cache internals → `old_impl/tests/`, `old_impl/scripts/`.

Nothing was deleted: the archive is the source for rerunning the old stack with `PYTHONPATH=old_impl`.

## What stayed (active)

- **`mlsys_kv`**: CLI (`mlsys-kv`), `infra/` (config, device, seed, logging), `models/` (HF load), `datasets/`.
- **Top-level packages** (under `src/`): `cache/` (FP16 KV base path for AR), `decoding/` (greedy autoregressive), `benchmarks/` (metrics, timers, schema, analysis, **AR-only** `run_benchmark_sweep`), `kernels/`, `serving/` (stubs).
- **Root** `modal_app.py`, `modal_sweep.py` (imports `benchmarks`).

## Why rebuild

The previous design coupled draft-side compression to a **specific** speculative pipeline (separate draft caches, per-round snapshot rebuild, memory-only quant + dequant attention). The next implementation targets a **QuantSpec-style** architecture without inheriting those coupling points by default.

## Verification

- No `import old_impl` in application code.
- `pip install -e .` succeeds; `pytest tests/` passes (autoregressive + Phase 16 analysis smoke).
- Legacy run: `PYTHONPATH=old_impl python -m mlsys_kv.cli speculative --config old_impl/configs/speculative.yaml` (separate environment from editable install if needed).

See also `docs/REPO_RESET.md` and `old_impl/README.md`.
