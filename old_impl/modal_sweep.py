"""Modal GPU sweep entrypoint (Phase 8). Persists CSV/JSONL to a Modal Volume after **each** row.

Run (smoke)::

    modal run modal_sweep.py --sweep configs/benchmark_smoke_modal.yaml

Full grid (use ``benchmark_full_modal_v4.yaml`` for reduced grid; Modal **max** execution time is **24h**)::

    modal run modal_sweep.py --sweep configs/benchmark_full_modal_v4.yaml

Override GPU::

    modal run modal_sweep.py --sweep configs/benchmark_smoke_modal.yaml --gpu A100

Mount your Hugging Face API token by creating a Modal secret named ``hf-token`` whose keys
include ``HF_TOKEN`` (same as ``modal secret create hf-token HF_TOKEN=hf_...``). The sweep
function uses ``Secret.from_name("hf-token")`` so ``transformers`` can download gated checkpoints.

Environment inside the container sets ``MODAL_RESOURCE_TAG`` to the same string for CSV logging
(e.g. ``A10G``) alongside ``torch.cuda.get_device_name`` (e.g. ``NVIDIA A10G``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent

HF_VOLUME = modal.Volume.from_name("mlsys-kv-hf-cache-v1", create_if_missing=True)
RESULTS_VOLUME = modal.Volume.from_name("mlsys-kv-benchmark-results-v1", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime",
        add_python=False,
    )
    .pip_install(
        # Transformers 5.x expects torch>=2.4; this image ships torch 2.2.1.
        "transformers>=4.40.0,<5",
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

app = modal.App("mlsys-kv-benchmark-sweep")


@app.function(
    image=IMAGE,
    gpu="A10G",
    # Modal allows at most 24h per function run — cannot omit timeout entirely.
    timeout=86400,
    volumes={"/hf": HF_VOLUME, "/results": RESULTS_VOLUME},
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_sweep_on_gpu(sweep_config_rel: str, modal_resource_tag: str) -> None:
    """Execute :func:`run_benchmark_sweep` with Volume commit after every CSV append."""
    repo_src = "/repo/src"
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)

    env = os.environ.copy()
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    env["HF_HOME"] = "/hf/huggingface"
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])
    env["PYTHONPATH"] = repo_src
    env["MODAL_RESOURCE_TAG"] = modal_resource_tag

    # ``mlsys-kv`` depends on ``modal`` and open ``transformers`` pins; a normal editable
    # install upgrades Transformers to 5.x, which then **disables** torch 2.2.1. Only install
    # our sources; the image layer already has the runtime stack above.
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", "/repo"],
        env=env,
    )

    from mlsys_kv.benchmarks.experiment_runner import run_benchmark_sweep

    def _commit() -> None:
        RESULTS_VOLUME.commit()

    cfg_path = Path("/repo") / sweep_config_rel
    run_benchmark_sweep(cfg_path, volume_commit_fn=_commit, modal_resource_tag=modal_resource_tag)
    RESULTS_VOLUME.commit()
    HF_VOLUME.commit()


@app.local_entrypoint()
def main(
    sweep: str = "configs/benchmark_smoke_modal.yaml",
    gpu: str = "A10G",
) -> None:
    """Dispatch remote sweep; ``gpu`` is written to CSV as ``modal_resource_tag`` (match Modal ``gpu=``)."""

    run_sweep_on_gpu.remote(sweep, gpu)
