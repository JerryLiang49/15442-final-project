"""Modal GPU runner for Phase H (QuantSpec evaluation). Persists CSV + JSONL to a dedicated Modal Volume.

**Volume name**

Set ``MODAL_PHASE_H_VOLUME`` **before** ``modal run`` so the module picks it up at import time
(created on first use with ``create_if_missing=True``)::

    MODAL_PHASE_H_VOLUME=mlsys-kv-phase-h-20260417 modal run modal_phase_h.py --sweep configs/phase_h_full_modal.yaml

Default: ``mlsys-kv-phase-h-results-v1`` (separate from legacy ``mlsys-kv-benchmark-results-v1``).

**Prerequisites**

* ``modal token new`` (logged in).
* Secret ``hf-token`` with ``HF_TOKEN`` for Hugging Face.

**Examples**::

    modal run modal_phase_h.py --sweep configs/phase_h_full_modal.yaml --gpu A10G

See ``configs/phase_h_full_modal.yaml`` for the full ablation grid.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent

# HF cache shared with other project Modal apps
HF_VOLUME = modal.Volume.from_name("mlsys-kv-hf-cache-v1", create_if_missing=True)

# Phase H artifacts only — override with MODAL_PHASE_H_VOLUME=my-name modal run ...
_PHASE_H_VOL_NAME = os.environ.get("MODAL_PHASE_H_VOLUME", "mlsys-kv-phase-h-results-v1")
RESULTS_VOLUME = modal.Volume.from_name(_PHASE_H_VOL_NAME, create_if_missing=True)

IMAGE = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime",
        add_python=False,
    )
    .pip_install(
        "transformers>=4.40.0,<5",
        "accelerate>=0.28.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.0",
        "triton>=2.2,<3",
    )
    .add_local_dir(str(PROJECT_ROOT), remote_path="/repo")
)

app = modal.App("mlsys-kv-phase-h-benchmark")


@app.function(
    image=IMAGE,
    gpu="A10G",
    timeout=86400,
    volumes={"/hf": HF_VOLUME, "/results": RESULTS_VOLUME},
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_phase_h_on_gpu(sweep_config_rel: str, modal_resource_tag: str) -> None:
    """Run Phase H sweep; commit results volume after each CSV/JSONL row."""
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
    sweep: str = "configs/phase_h_full_modal.yaml",
    gpu: str = "A10G",
) -> None:
    """Dispatch remote Phase H run. ``gpu`` should match the decorator GPU for accurate tagging."""

    print(f"[modal_phase_h] results volume name: {_PHASE_H_VOL_NAME}", flush=True)
    run_phase_h_on_gpu.remote(sweep, gpu)
