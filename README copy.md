# MLSys KV project (Phase 1 scaffold)

CMU MLSys course project: joint KV-cache sparsification and quantization for efficient **self-speculative decoding** on **Llama-2-7B** (single GPU). This phase provides repo layout, config, logging, Hugging Face loading, autoregressive smoke decoding, and Modal GPU smoke.

## Environment setup

- **Python**: 3.10+ recommended.
- **GPU (local)**: NVIDIA GPU with CUDA capable of running Llama-2-7B in FP16 (≈14GB+ VRAM without aggressive offloading).
- **Hugging Face**: `meta-llama/Llama-2-7b-hf` is gated. Accept the license on Hugging Face, then authenticate:
  - `huggingface-cli login`
  - or set `HF_TOKEN` in the environment (required for Modal if the volume has no cached weights yet).

Install from the repository root:

```bash
cd mlsys-kv-project
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Model checkpoint

- **Name**: `meta-llama/Llama-2-7b-hf` (configurable via `configs/base.yaml` → `model_name`).
- Pin a revision in YAML if you need strict reproducibility across HF updates.

## Local smoke test

```bash
./scripts/run_local_smoke.sh
```

Or:

```bash
python -m mlsys_kv.cli smoke --config configs/base.yaml
```

This loads tokenizer + causal LM, runs one **greedy** autoregressive generation for `max_new_tokens`, prints text and timing, and writes JSONL under `output_dir`.

## Modal smoke test (GPU)

Prerequisites: [Modal](https://modal.com) account, `modal token new`, and (for gated Llama-2) a Modal secret with `HF_TOKEN` (see `modal_app.py`).

```bash
./scripts/run_modal_smoke.sh
```

Or:

```bash
modal run modal_app.py
```

The app uses a **PyTorch CUDA** base image, mounts this repo, sets `HF_HOME` on a persistent volume for downloads, installs the package, and runs the same CLI smoke command.

### GPU type

The default Modal GPU is **A10G** (edit `gpu=` in `modal_app.py` if you prefer A100, L40S, etc.). Document the GPU you use in experiment logs; Phase-1 JSONL includes host/device fields when available.

## Determinism

Smoke decoding uses **greedy** generation (`do_sample=False`). With a fixed `seed` (set on Python/NumPy/torch/CUDA if available), runs should match for a given PyTorch/CUDA/transformers stack. Minor differences can still appear across hardware or library builds; record versions in logs.

## Repo layout

- `src/mlsys_kv/` — package (`models`, `cache`, `decoding`, `benchmarks`, `datasets`, `infra`, `analysis`)
- `configs/` — YAML
- `scripts/` — local / Modal smoke wrappers
- `outputs/` — run artifacts (JSONL, etc.)
