# Benchmark sweep: commands (Phase 15–16)

The **joint** mode is `sparse_quant` in YAML (label **`spec_sparse_quant_memonly`**). It must appear in `modes:` and, on Modal, the job needs enough **timeout** (see `modal_sweep.py`, default **4h**).

## 1. Gate (before any long sweep)

```bash
cd 15442-final-project
PYTHONPATH=src pytest -m benchmark_gate -q
```

Optional presweep (local + report checks):

```bash
python scripts/benchmark_presweep_gate.py --skip-pytest
```

## 2. Quick Modal check that **joint** rows are produced

Runs all five labels on a **tiny** grid (`configs/benchmark_joint_smoke_modal.yaml`):

```bash
modal run modal_sweep.py --sweep configs/benchmark_joint_smoke_modal.yaml --gpu A10G
```

Download or mount Volume and confirm CSV contains `benchmark_label` = `spec_sparse_quant_memonly`.

## 3. Stage 1 (narrow, all five modes — includes joint)

**Local:**

```bash
PYTHONPATH=src python -m mlsys_kv.cli benchmark-sweep --config configs/benchmark_stage1_local.yaml
```

**Modal:**

```bash
modal run modal_sweep.py --sweep configs/benchmark_stage1_modal.yaml --gpu A10G
```

## 4. Full sweep (paper-scale grid)

### v4 (recommended): smaller search space, full five modes

| Parameter | v3 (old) | v4 (reduced) |
|-----------|----------|--------------|
| `k_values` | `1,3,5,7` (4) | `1,3,7` (3) — drop middle **5** |
| `sparsity_budgets` | `0.1,0.2,0.4,1.0` (4) | `0.1,0.4,1.0` (3) — drop **0.2** |
| `quant_bits` | `4,8,16` | **unchanged** (all three kept) |
| Grid cells | **81** | **49** |
| Rows (17 prompts × 3 trials) | **4,131** | **2,499** |

**Modal** (`modal_sweep.py` uses **86400s = 24h** — Modal’s maximum; timeouts cannot be removed):

```bash
modal run modal_sweep.py --sweep configs/benchmark_full_modal_v4.yaml --gpu A10G
```

Outputs: `/results/sweep_full_modal_v4.csv` on the Volume.

**Local (GPU)** — same grid as v4:

```bash
PYTHONPATH=src python -m mlsys_kv.cli benchmark-sweep --config configs/benchmark_full_v4.yaml
```

**v3** (larger grid) still available as `configs/benchmark_full_modal.yaml` → `sweep_full_modal_v3.csv`.

If a previous run **stopped early**, the same command with `resume: true` continues unfinished cells.

## 5. Scale sweep (larger model + longer prompts, **separate** CSV from v4)

**Prompts:** `data/mt_bench_scale_prompts.json` — the original **17** `mt_bench_subset.json` entries plus **8** curated
medium/long synthetic passages. Regenerate (after editing the script) with:

```bash
PYTHONPATH=src python scripts/build_mt_bench_scale_prompts.py
```

**Model:** `gpt2-xl` (~1.5B), FP16. Stays within **1024-token** GPT-2 context (`max_new_tokens=32`).

**Buckets:** `short_token_max: 256`, `medium_token_max: 640` so prompts ~692–884 tok land in **long**.

**Grid (gpt2-xl):** K ∈ `{1,3,5,7}`, `max_new_tokens` ∈ `{32,64}` → **65** cells × **25** prompts × **2** trials × **2** gen lengths = **6,500** rows.

**Modal (full gpt2-xl grid):**

```bash
modal run modal_sweep.py --sweep configs/benchmark_scale_modal_v1.yaml --gpu A10G
```

**Modal (same grid, long-context model — TinyLlama 1.1B, separate CSV):**

```bash
modal run modal_sweep.py --sweep configs/benchmark_scale_longctx_modal_v1.yaml --gpu A10G
```

**Modal (smoke, ~44 rows):**

```bash
modal run modal_sweep.py --sweep configs/benchmark_scale_smoke_modal.yaml --gpu A10G
```

**Local GPU:**

```bash
PYTHONPATH=src python -m mlsys_kv.cli benchmark-sweep --config configs/benchmark_scale_local_v1.yaml
# optional: TinyLlama grid
PYTHONPATH=src python -m mlsys_kv.cli benchmark-sweep --config configs/benchmark_scale_longctx_local_v1.yaml
```

Volume outputs: `/results/sweep_scale_modal_v1.csv` and `/results/sweep_scale_longctx_modal_v1.csv` (do **not** overwrite v4).

**Phase 16 report** (keep your existing v4 report; write scale results to a **new** directory):

```bash
pip install "scipy>=1.11"
PYTHONPATH=src python -m mlsys_kv.cli benchmark-report \
  --csv results/sweep_scale_modal_v1.csv \
  --out results/phase16_scale_v1
```

## 6. Phase 16 report (original v4 CSV)

Copy the sweep CSV from Modal Volume to e.g. `results/sweep_full_modal_v4.csv`, then:

```bash
pip install "scipy>=1.11"
PYTHONPATH=src python -m mlsys_kv.cli benchmark-report \
  --csv results/sweep_full_modal_v4.csv \
  --out results/phase16_report
```

## 7. Why an older CSV had no joint rows

- **`sparse_quant` was last** in mode order and the Modal **1h** timeout was hit first, **or**
- The run used an older config without `sparse_quant`.

Current defaults: **joint scheduled third** (after AR + FP16), Modal **timeout = 4h**, full Modal outputs **`sweep_full_modal_v3.csv`**.
