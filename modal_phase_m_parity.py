"""Run Phase M fused-kernel parity tests (``-m parity_cuda``) on a Modal GPU.

Requires: ``modal`` CLI logged in (``modal token new``).

Usage::

    modal run modal_phase_m_parity.py
    modal run modal_phase_m_parity.py --gpu A10G

This installs the repo editable **without** re-resolving PyTorch (the CUDA image already has
conda PyTorch). It also installs ``build-essential`` so Triton can JIT-compile kernels (the
stock ``*-runtime`` CUDA image has no C compiler).

Inside the container::

    export PYTHONPATH=/repo/src
    bash scripts/run_phase_m_parity.sh -m parity_cuda -v

**Note:** GPU type is selected via ``--gpu`` on ``modal run`` (see Modal docs); the default in
``@app.function`` is ``A10G``.

**Stack:** This repo imports ``DynamicLayer`` / ``DynamicSlidingWindowLayer`` from
``transformers.cache_utils`` (Transformers **5.x** API) and needs ``is_torch_available()`` to pass
(Transformers **5.5+** requires **PyTorch ≥ 2.4**; loading **``.bin``** checkpoints with recent
Transformers also requires **PyTorch ≥ 2.6** per HF ``torch.load`` policy). The Modal image is
**PyTorch 2.6** CUDA runtime + **transformers 4.40+** from pip (typically 5.x). Editable
``pip install -e .`` uses ``--no-deps`` so conda PyTorch is not replaced.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent

HF_VOLUME = modal.Volume.from_name("mlsys-kv-hf-cache-v1", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry(
        # 2.6+ for Transformers 5.x load_state_dict on .bin weights (CVE-2025-32434 guard).
        "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime",
        add_python=False,
    )
    # Triton compiles host stubs with gcc on first kernel launch.
    .apt_install("build-essential")
    .pip_install(
        "transformers>=4.40.0,<6",
        "accelerate>=0.28.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.26.0",
        "pytest>=8.0.0",
        # Conda PyTorch bundles Triton; do not pin an old triton from a legacy image.
    )
    .add_local_dir(str(PROJECT_ROOT), remote_path="/repo")
)

app = modal.App("mlsys-kv-phase-m-parity")


@app.function(
    image=IMAGE,
    gpu="A10G",
    timeout=60 * 30,
    volumes={"/hf": HF_VOLUME},
)
def run_parity_remote() -> int:
    """Return pytest exit code (0 = all selected tests passed)."""
    env = os.environ.copy()
    env["HF_HOME"] = "/hf/huggingface"
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])
    env["PYTHONPATH"] = "/repo/src"

    # Do not let pip install torch/transformers from PyPI over the image's CUDA stack.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            "/repo",
            "--no-deps",
        ],
        env=env,
    )
    cmd = [
        "bash",
        "/repo/scripts/run_phase_m_parity.sh",
        "-m",
        "parity_cuda",
        "-v",
    ]
    print("[modal_phase_m_parity] running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd="/repo", env=env)
    HF_VOLUME.commit()
    return int(p.returncode)


@app.local_entrypoint()
def main() -> None:
    rc = run_parity_remote.remote()
    if rc != 0:
        sys.exit(rc)
