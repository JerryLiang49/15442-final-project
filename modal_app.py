"""Modal GPU entrypoint for Phase-1 autoregressive smoke.

Run::

    modal run modal_app.py

**Hugging Face gated models:** create a Modal Secret (e.g. ``hf-token``) whose
environment mapping includes ``HF_TOKEN``, then uncomment ``secrets=...`` below.

**GPU choice:** adjust ``gpu=`` (for example ``\"A100\"``, ``\"A10G\"``, ``\"L40S\"``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent

# Persist Hugging Face downloads across invocations (mount at HF_HOME inside the container).
HF_VOLUME = modal.Volume.from_name("mlsys-kv-hf-cache-v1", create_if_missing=True)

# Official PyTorch CUDA runtime image (includes torch); pin for stability in writeups.
IMAGE = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime",
        add_python=False,
    )
    .pip_install(
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.0",
    )
    # Package the repo into the container image so `modal run` always sees current files.
    .add_local_dir(str(PROJECT_ROOT), remote_path="/repo")
)

app = modal.App("mlsys-kv-phase1")


@app.function(
    image=IMAGE,
    gpu="A10G",
    timeout=60 * 45,
    volumes={"/hf": HF_VOLUME},
    # Uncomment after creating a Modal Secret with HF_TOKEN for gated Llama-2 weights:
    # secrets=[modal.Secret.from_name("hf-token")],
)
def run_smoke_remote() -> None:
    """Install the local package and run the same CLI smoke test as local dev."""
    env = os.environ.copy()
    env["HF_HOME"] = "/hf/huggingface"
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])
    env["PYTHONPATH"] = "/repo/src"

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/repo"],
        env=env,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "mlsys_kv.cli",
            "smoke",
            "--config",
            "/repo/configs/base.yaml",
            "--output-dir",
            "/repo/outputs/raw",
            "--device",
            "cuda",
        ],
        cwd="/repo",
        env=env,
    )
    HF_VOLUME.commit()


@app.local_entrypoint()
def main() -> None:
    """``modal run modal_app.py`` dispatches to the remote GPU function."""
    run_smoke_remote.remote()
