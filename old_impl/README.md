# Archived implementation (legacy self-speculative stack)

This tree holds the **previous end-to-end** package and assets so nothing was deleted when the active codebase moved to a QuantSpec-style rebuild layout.

## Contents

| Path | Description |
|------|-------------|
| `mlsys_kv/` | Legacy **`mlsys_kv`** package: speculative decoding, full `cache/` (sparse/quant), `decoding/speculative.py`, `benchmarks/` with multi-mode `experiment_runner`, etc. Synced from `src/mlsys_kv/` before those modules were removed from the active tree. |
| `configs/` | Benchmark sweep YAMLs and `speculative.yaml` (multi-mode grids). Active repo keeps only minimal AR-only configs at the root. |
| `tests/` | Tests for speculative/cache/sparse/quant and benchmark gate; plus copies of tests that were moved here from `tests/`. |
| `scripts/` | Legacy sweep helpers (`run_benchmark_full.sh`, `local_mini_sweep.py`, …). |
| `modal_sweep.py`, `modal_app.py` | Reference copies of Modal entrypoints. **Use the repo root** `modal_sweep.py` for current runs. |

## Running the legacy stack

From the repo root (prefer a clean shell or separate venv so the editable install does not shadow `mlsys_kv`):

```bash
PYTHONPATH=old_impl python -m mlsys_kv.cli speculative --config old_impl/configs/speculative.yaml
PYTHONPATH=old_impl python -m mlsys_kv.cli benchmark-sweep --config old_impl/configs/benchmark_smoke.yaml
```

Do **not** import `old_impl` from active code under `src/`.

## See also

- `docs/RESET_PLAN.md` — what moved, what stayed, why rebuild.
- `docs/REPO_RESET.md` — inventory.
