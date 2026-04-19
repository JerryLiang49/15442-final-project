"""Modal GPU runner — Phase N roofline / fused-kernel throughput benchmarking.

Uses PyTorch **2.6** CUDA runtime so Hugging Face can load legacy **``.bin``** checkpoints with recent
Transformers (CVE-2025-32434 guard requires torch ≥ 2.6 unless weights are safetensors). Stack also
includes Transformers 5.x + Triton JIT + gcc (``build-essential``).

**Cheap smoke** (few rows)::

    modal run modal_phase_n.py --sweep configs/phase_n_smoke_modal.yaml --gpu A10G

**Full sweep** (long; tune ``context_length_tokens_values`` and ``max_prompts``)::

    modal run modal_phase_n.py --sweep configs/phase_n_sweep_modal.yaml --gpu A100

Results volume: ``MODAL_PHASE_N_VOLUME`` (default ``mlsys-kv-phase-n-results-v1``), separate from legacy sweeps.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent

# Override without editing code: ``MODAL_PHASE_N_GPU=A100 modal run modal_phase_n.py ...``
_MODAL_PHASE_N_GPU = os.environ.get("MODAL_PHASE_N_GPU", "A10G")

HF_VOLUME = modal.Volume.from_name("mlsys-kv-hf-cache-v1", create_if_missing=True)
_PHASE_N_VOL = os.environ.get("MODAL_PHASE_N_VOLUME", "mlsys-kv-phase-n-results-v1")
RESULTS_VOLUME = modal.Volume.from_name(_PHASE_N_VOL, create_if_missing=True)

IMAGE = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime",
        add_python=False,
    )
    .apt_install("build-essential")
    .pip_install(
        "transformers>=4.40.0,<6",
        "accelerate>=0.28.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.0",
    )
    .add_local_dir(str(PROJECT_ROOT), remote_path="/repo")
)

app = modal.App("mlsys-kv-phase-n-roofline")


@app.function(
    image=IMAGE,
    gpu=_MODAL_PHASE_N_GPU,
    timeout=86400,
    volumes={"/hf": HF_VOLUME, "/results": RESULTS_VOLUME},
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_phase_n_remote(sweep_config_rel: str, modal_resource_tag: str) -> None:
    # Relative paths in YAML (e.g. data/mt_bench_subset.json) assume repo root as cwd.
    os.chdir("/repo")
    repo_src = "/repo/src"
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)

    env = os.environ.copy()
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    env["HF_HOME"] = "/hf/huggingface"
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])
    env["PYTHONPATH"] = repo_src
    env["MODAL_RESOURCE_TAG"] = modal_resource_tag

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", "/repo"],
        env=env,
    )

    from benchmarks.phase_h_runner import run_phase_h_benchmark

    def _commit() -> None:
        RESULTS_VOLUME.commit()

    cfg_path = Path("/repo") / sweep_config_rel
    run_phase_h_benchmark(cfg_path, volume_commit_fn=_commit)
    RESULTS_VOLUME.commit()
    HF_VOLUME.commit()


@app.local_entrypoint()
def main(
    sweep: str = "configs/phase_n_smoke_modal.yaml",
    gpu: str = "A10G",
) -> None:
    print(
        f"[modal_phase_n] results volume: {_PHASE_N_VOL}  "
        f"decorator_gpu={_MODAL_PHASE_N_GPU}  modal_resource_tag={gpu}",
        flush=True,
    )
    run_phase_n_remote.remote(sweep, gpu)
