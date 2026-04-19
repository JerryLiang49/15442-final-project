## Title: 
Joint KV Cache Sparsification and Quantization for Efficient Self-Speculative Decoding
## Team Members: 
Harry Hu (yuehanh), Jerry Liang (zhanminl), Soham Khatavkar (skhatavk)

## Introduction
Large language model serving is increasingly bottlenecked by the growth of the key-value (KV) cache during autoregressive decoding. Self-speculative decoding improves inference efficiency by using a cheaper draft version of the same model to propose tokens, which are then verified by the full model. Recent work such as QuantSpec shows that a quantized draft cache can preserve high acceptance rates while significantly improving throughput. Meanwhile, prior work such as H2O and SnapKV shows that attention is highly sparse, with most attention mass concentrated on a small set of heavy-hitter tokens, enabling substantial KV-cache reduction with limited quality degradation.

These two compression directions, however, have mostly been studied independently. We propose to study their interaction in the draft path of self-speculative decoding. Our hypothesis is that combining sparsification and quantization can reduce draft-side memory and compute cost more than either method alone, but may also lower proposal quality and thus reduce acceptance rate. This creates a systems tradeoff between draft efficiency and verification efficiency that has not been well characterized.

## Problem
Our project asks whether jointly applying KV-cache sparsification and quantization to the draft path of self-speculative decoding can improve end-to-end decoding throughput beyond using only one technique at a time. In our setup, the verifier always uses the full-precision KV cache, so the final outputs remain identical to standard decoding. The challenge is that a more aggressively compressed draft cache generates proposals faster, but may decrease acceptance rate and increase wasted verification work. We aim to characterize this tradeoff and identify operating points that maximize throughput under memory constraints.

## Status Quo
QuantSpec studies hierarchical quantized KV caches for self-speculative decoding and reports substantial speedups while maintaining high acceptance rates. H2O and SnapKV introduce attention-score-based token eviction methods for standard LLM inference by retaining heavy-hitter tokens together with a recent-token window. KIVI and MiniKV further show that aggressive KV-cache quantization can be effective in practice. However, prior work has not directly studied the interaction between sparsification and quantization within the draft path of self-speculative decoding. Our project focuses on this gap and evaluates whether these two compression mechanisms are complementary or conflicting in practice.

 
## High-Level Implementation Plan
We will build a self-speculative decoding prototype in PyTorch for Llama-2-7B using Hugging Face components and vLLM-style KV-cache abstractions where possible, running on Modal GPU instances. The verifier will use a full FP16 KV cache as the source of truth, while the draft path will operate on a compressed cache. For the draft cache, we will apply SnapKV-style heavy-hitter selection with a fixed recent window, then quantize the retained KV entries to INT8 or INT4. The draft will propose K candidate tokens, which the verifier will check against the full cache. We will compare four settings: standard autoregressive decoding, quantization-only draft compression, sparsification-only draft compression, and the combined method.

## Evaluation
We will evaluate each method on 50–100 prompts from MT-Bench while sweeping sparsification ratios, quantization levels, draft lengths, and context lengths. Our primary metrics are acceptance rate, tokens per second, peak KV-cache memory, and per-token latency. We will also analyze how the combined method changes the tradeoff frontier relative to quantization-only and sparsification-only baselines. If time permits, we will explore periodic heavy-hitter refresh and per-head token selection as extensions.
References
 Li, Yuhong, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM Knows What You are Looking for Before Generation. NeurIPS, 2024.
Liu, Zirui, Jiayi Yuan, Hong Jin, Shaofeng Zhong, Zizheng Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML, 2024.
Tiwari, Rishabh, Haocheng Xi, Aditya Tomar, Coleman Hooper, Sehoon Kim, Maxwell Horton, Mahyar Najibi, Michael W. Mahoney, Kurt Keutzer, and Amir Gholami. QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache. ICML, 2025.
Zhang, Zhenyu, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. NeurIPS, 2023.
Sharma, Akshat, Hangliang Ding, Jianping Li, Neel Dani, and Minjia Zhang. MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache. arXiv:2411.18077, 2024.


---

## Project status

Core decoding and evaluation are implemented end-to-end:

| Phase | Scope |
|-------|--------|
| 1–2 | Autoregressive smoke and instrumented greedy baseline (JSONL metrics). |
| 3 | Self-speculative decoding loop (draft proposals + greedy verifier). |
| 4 | Pluggable draft cache (`KVCacheBase`) with FP16 draft. |
| 5 | Draft-only symmetric quantization (`quant_only`). |
| 6 | Draft-only **SnapKV-style** sparsity: recent window + heavy hitters (`sparse_only`). |
| 7 | **Joint** draft cache: **sparsify retained tokens, then quantize** those tensors only (`sparse_quant`). |
| 8–15 | **Benchmark harness**: MT-Bench subset sweeps, YAML-driven grids (`mlsys-kv benchmark-sweep`), CSV/JSONL/processed rollup with **schema v2** (per-mode labels, throughput/VRAM/draft–verify split, optional Modal Volume commit via `modal_sweep.py`). |
| 16 | **Analysis / report**: `mlsys-kv benchmark-report` writes `INDEX.md`, `HOW_TO_VIEW.md`, `tables/`, `figures/` from CSV v2 (see `docs/PHASE16_ANALYSIS.md`). |
| N | **Modal roofline / fused throughput**: YAML-driven grid via `benchmarks.phase_h_runner` (schema v4), Llama-arch **QuantSpec attention patch** auto-enabled for `kv_kernel_backend=triton`, `context_length_tokens_values` for context sweeps (default config: **OpenLlama 7B**); `python -m benchmarks.phase_n_report` for roofline markdown; `modal run modal_phase_n.py --sweep configs/phase_n_smoke_modal.yaml`. |
| O | **Kernel tuning / production perf**: named Triton presets in `kv_kernels.tuning` (separate draft `q_len=1` vs verifier `q_len=γ` vs hist Q·K); `kernel_tuning_profile` + JSON in benchmark CSV (schema v5); `python -m benchmarks.phase_o_profile` for post-fusion hotspots; `python -m benchmarks.kernel_tune_microbench`; regression tests in `tests/test_kernel_tuning_regression.py`. |

The **verifier** always uses a **full FP16** KV cache; the **draft** path supports **four modes**: `fp16`, `quant_only`, `sparse_only`, and `sparse_quant`. Final greedy outputs match standard autoregressive decoding when speculative verification is enabled (`verify_match`).

**Config keys:** YAML uses **`dtype`** (e.g. `float16`) for model weights; **`--dtype`** on the CLI (legacy **`--torch-dtype`** is still accepted).

---

## Repository reset (`old_impl/`)

Before the next implementation phase, a **frozen snapshot** of the legacy package, configs, tests, scripts, and Modal entrypoints lives under **`old_impl/`** (see **`old_impl/README.md`**). It documents the self-speculative draft/verifier split, memory-only quant + dequant attention, and benchmark schema tied to those modes.

**Active code:** `mlsys_kv` holds the CLI plus `infra/`, `models/`, and `datasets/`. **Decoding** (autoregressive), **KV cache** (FP16 base for AR), and **benchmarks** (metrics, schema, analysis, AR-only sweep) live under **`src/cache/`**, **`src/decoding/`**, and **`src/benchmarks/`**. Legacy speculative/cache/sweep code is only under **`old_impl/`**. Details: **`docs/RESET_PLAN.md`**, inventory: **`docs/REPO_RESET.md`**.

---

## Repository layout

Install with `pip install -e .` from the repo root (packages `cache`, `decoding`, `benchmarks`, `kernels`, `serving`, `mlsys_kv` under `src/`).

```
15442-final-project/
├── old_impl/                # full legacy mlsys_kv + benchmark configs/sweep scripts/tests
├── configs/
│   ├── base.yaml, baseline.yaml, benchmark_smoke_ar.yaml, benchmark_presweep_*.yaml
│   └── (legacy factorial sweeps: old_impl/configs/)
├── data/
│   └── mt_bench_subset.json
├── docs/
│   ├── RESET_PLAN.md, REPO_RESET.md
│   ├── BENCHMARK_READINESS.md, BENCHMARK_PHASE15.md, PHASE16_ANALYSIS.md, …
├── scripts/
│   ├── benchmark_presweep_gate.py, run_benchmark_smoke.sh, run_local_*.sh, …
├── src/
│   ├── cache/             # KVCacheBase + FP16 (AR path)
│   ├── decoding/          # greedy autoregressive
│   ├── benchmarks/        # metrics, schema, analysis, AR-only experiment_runner
│   ├── kernels/, serving/ # stubs for future work
│   └── mlsys_kv/          # CLI, infra, models, datasets
├── tests/                 # AR + Phase 16 analysis (legacy tests moved under old_impl/tests/)
├── modal_app.py, modal_sweep.py
├── pyproject.toml
└── README.md
```

Optional committed artifacts (e.g. `results/`) may hold exported sweep CSVs/summaries for the writeup.

**Note:** Multi-mode speculative sweeps and `mlsys-kv speculative` use the **archive**; see `docs/RESET_PLAN.md`.

---

## System architecture

**Draft path (compressed).** On each draft step, the cache absorbs the full `past_key_values` returned by Hugging Face for that forward, then:

1. **Sparsify (when applicable):** compute token importance (key-norm and/or attention-based scoring where supported), apply a fixed **recent window** plus **heavy-hitter** budget, and **gather** K/V along sorted global indices so the cache length is `R ≤ L`.
2. **Quantize (when applicable):** apply **symmetric INT8 per tensor** to the **retained** keys and values only (composition order: **sparsify first, quantize second** for `sparse_quant`).

**Verifier path (exact).** The verifier replays accepted draft tokens (or corrects on mismatch) using a **fresh FP16** clone of its KV state. Draft compression does not change committed tokens under greedy verification; **correctness** is preserved because the verifier is the source of truth for what gets appended to the live sequence.

**Metrics.** Experiments log acceptance rate, wall-clock runtime, logical draft KV bytes (payload + metadata splits where implemented), and draft-side timing such as dequantization and selector refresh for sparse modes.

---

## Hardware and environment

| Requirement | Notes |
|-------------|--------|
| **Python** | ≥ 3.10 (`pyproject.toml`). |
| **PyTorch** | ≥ 2.2 (CUDA optional; `device: auto` in YAML). |
| **Transformers / HF** | Default Phase N Modal sweep uses **OpenLlama 7B** (`openlm-research/open_llama_7b_v2`, ungated). For **Meta Llama-2-7B** (`meta-llama/Llama-2-7b-hf`), accept the license, get repo access, and set **`HF_TOKEN`** (Modal `hf-token` secret). |
| **Modal** | Optional cloud GPU: `modal_app.py` (AR smoke) or `modal_sweep.py` (benchmark sweeps); Modal account + `modal` CLI. |
| **Development** | `pip install -e ".[dev]"` installs pytest/ruff per `pyproject.toml`. |

**Local setup:**

```bash
cd 15442-final-project
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e ".[dev]"
```

`requirements.txt` aligns with runtime dependencies; `pyproject.toml` is the authoritative package definition (including optional `dev` extras).

---

## Usage and configuration

**CLI entrypoint** (after editable install): `mlsys-kv`, or `python -m mlsys_kv.cli`.

| Command | Purpose |
|---------|---------|
| `mlsys-kv smoke --config configs/base.yaml` | Short autoregressive generation smoke test. |
| `mlsys-kv baseline --config configs/baseline.yaml` | Instrumented AR baseline; JSONL under `output_dir`. |
| `mlsys-kv speculative --config configs/speculative.yaml` | Self-speculative decode; see draft mode below. |
| `mlsys-kv benchmark-sweep --config configs/benchmark_smoke.yaml` | MT-Bench subset sweep; append CSV + JSONL + processed rollup (see `docs/BENCHMARK_PHASE15.md`). |
| `mlsys-kv benchmark-report --csv results/s.csv --out results/phase16` | Phase 16: Pareto/stacked-latency plots, CSV summaries, paired speedup + optional **p-values** (`pip install scipy` or `pip install -e ".[analysis]"`). |

**Draft cache mode** is selected by `draft_cache_mode` in the speculative config (or override with `--draft-mode`):

| `draft_cache_mode` | Draft KV |
|--------------------|----------|
| `fp16` | Full precision (baseline). |
| `quant_only` | Quantized full sequence (e.g. INT8/INT4 in sweep configs; memory-only semantics). |
| `sparse_only` | Heavy-hitters + recent window (FP16 retained entries). |
| `sparse_quant` | Same retention as sparse; quantization on **retained** tensors only. |

Sparse hyperparameters (shared by `sparse_only` and `sparse_quant`) are read from YAML when present: `sparse_recent_window`, `sparse_heavy_hitter_budget`, `sparse_refresh_interval`, `sparse_scoring` (`key_norm` or `attention`). See `src/mlsys_kv/infra/config.py` for defaults.

**Example: speculative smoke with joint draft cache**

```bash
mlsys-kv speculative \
  --config configs/speculative.yaml \
  --draft-mode sparse_quant \
  --prompt "Hello, I am" \
  --max-new-tokens 16
```

**Editing `configs/base.yaml`:** adjust `model_name` (e.g. keep `meta-llama/Llama-2-7b-hf` or switch to an ungated model such as `gpt2` for quick CPU trials), `dtype`, `max_new_tokens`, `device`, and `prompt`. Merge the same keys into other configs as needed for fair comparisons.

---

## Testing and validation

Run the full suite:

```bash
pytest tests/ -v
```

Skip slow tests (no HF model download):

```bash
pytest tests/ -m "not slow"
```

| Test module | What is validated |
|-------------|-------------------|
| `test_autoregressive.py` | Greedy `decode_greedy_autoregressive` matches `model.generate`-style reference; edge case `max_new_tokens=0`. |
| `test_speculative.py` | Speculative **full token sequence** equals greedy AR for FP16, sparse, and **sparse_quant** drafts (`verify_match`); factory smoke; high self-acceptance sanity case; metric reproducibility (deterministic fields). |
| `test_quantization.py` | INT8 round-trip error bound; `KVCacheQuantized` rebuild vs HF `DynamicCache`; logical memory accounting; speculative **quant_only** matches AR. |
| `test_sparse_cache.py` | Retention policy keeps recent window; `strip_last` sequence length; sparse cache stats/memory; key-norm score length. |
| `test_sparse_quantized.py` | Joint cache: sparsity reduces INT8 **payload** vs quant-only on a long prefix; metadata splits. |

Together, these tests enforce **output equality** between autoregressive and speculative paths under greedy verification, and exercise **KV cache shape / memory accounting** for quantized and sparse representations.

**Benchmark gate and factorial sweeps (Phase 14–15):** see [`docs/BENCHMARK_READINESS.md`](docs/BENCHMARK_READINESS.md) for the `benchmark_gate` pytest marker and checklist, and [`docs/BENCHMARK_PHASE15.md`](docs/BENCHMARK_PHASE15.md) for YAML mode aliases, strict labeled grids, and CSV schema v2. Presweep helper: `python scripts/benchmark_presweep_gate.py`.

---

## Modal integration

Modal entrypoints (see each file for GPU image and volumes):

| File | Role |
|------|------|
| **`modal_app.py`** | Phase-1 style **autoregressive smoke** on GPU (remote subprocess to the packaged CLI). Good for a quick “does the stack run on Modal?” check. |
| **`modal_sweep.py`** | Runs **`run_benchmark_sweep`** on GPU with YAML from `configs/benchmark_*_modal.yaml`, persisting **CSV / JSONL / rollup** to a Modal **Volume** (`/results`) with a commit after each row (resume-friendly). |
| **`modal_phase_m_parity.py`** | Phase M **fused-kernel parity** (`pytest -m parity_cuda`). |
| **`modal_phase_n.py`** | Phase N **roofline / throughput** sweep (`benchmarks.phase_h_runner` + volume). |
| **`modal_phase_o_tools.py`** | Phase O **profiler** + **kernel microbench** + optional **`scripts/profile_kernel.py`**; artifacts under `/results/phase_o/` on volume ``MODAL_PHASE_O_VOLUME``. |

**GPU without editing Python:** set **`MODAL_PHASE_N_GPU`** (Phase N) or **`MODAL_PHASE_O_GPU`** (Phase O tools) before `modal run`, e.g. `MODAL_PHASE_N_GPU=A100 modal run modal_phase_n.py --sweep configs/phase_n_sweep_modal.yaml`.

**Phase N configs**

| File | Role |
|------|------|
| **`configs/phase_n_sweep_modal.yaml`** | **Default** 7B-class sweep: **OpenLlama 7B v2**, contexts **1024 / 2048** (max context 2048; no gate). |
| **`configs/phase_n_sweep_modal_primary.yaml`** | Same intent as default sweep (duplicate for clarity). |
| **`configs/phase_n_sweep_modal_llama2_meta_gated.yaml`** | **Meta Llama-2-7B**, contexts **2048 / 4096** — requires Hugging Face **gated** access + `hf-token`. |
| **`configs/phase_n_sweep_modal_gpt2xl_short.yaml`** | Short-context **gpt2-xl** grid (smoke / regression; not the primary perf regime). |
| **`configs/phase_n_smoke_modal.yaml`** | Tiny Modal smoke. |

**Reporting:** `PYTHONPATH=src python -m benchmarks.phase_n_report --csv <path> --out report.md --extended` adds a **model × GPU × mode** table and notes on regime.

**Prerequisites:** [Modal](https://modal.com) account, `pip install modal`, `modal token new`, and for gated models a Modal **Secret** (e.g. `hf-token`) with `HF_TOKEN` (see `secrets=[...]` in each file).

**Examples:**

```bash
# AR smoke on Modal
modal run modal_app.py

# Benchmark sweep (default: configs/benchmark_smoke_modal.yaml)
modal run modal_sweep.py

# Full grid (long); optionally set --gpu to match `modal_resource_tag` in logs
modal run modal_sweep.py --sweep configs/benchmark_full_modal.yaml --gpu A10G

# Phase O — GPU profiler trace + fused-verifier microbench (CPU Mac cannot run these meaningfully)
modal run modal_phase_o_tools.py --tool both --gpu A10G
modal run modal_phase_o_tools.py --tool profile --trace-out phase_o/trace.json --gpu A10G
modal run modal_phase_o_tools.py --tool microbench --tuning-profile verifier_wide --microbench-iters 200 --gpu A10G

# Phase N — default 7B sweep (OpenLlama; optional HF_TOKEN if hub rate-limits)
MODAL_PHASE_N_GPU=A10G modal run modal_phase_n.py --sweep configs/phase_n_sweep_modal.yaml --gpu A10G

# QuantSpec kernel profiler (steady-state warmup; Chrome trace on volume)
modal run modal_phase_o_tools.py --tool profile_kernel --profile-kernel-extra "--profiler-warmup-iters 50" --gpu A10G
```

Adjust **`MODAL_PHASE_N_GPU` / `MODAL_PHASE_O_GPU`**, timeout, and secrets as needed. Helper scripts under `scripts/run_benchmark_*.sh` wrap local sweeps; Modal uses `modal_sweep.py` directly.


