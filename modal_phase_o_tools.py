"""Modal GPU — Phase O tools: ``phase_o_profile`` (profiler) and ``kernel_tune_microbench``.

Same CUDA stack as Phase N (PyTorch **2.6** runtime + Triton + gcc). Artifacts under ``/results/phase_o/`` on the volume.

**Profiler (Chrome trace)**

    modal run modal_phase_o_tools.py --tool profile --gpu A10G

**Microbench (fused verifier timing)**

    modal run modal_phase_o_tools.py --tool microbench --tuning-profile verifier_wide --iters 200 --gpu A10G

**Both**

    modal run modal_phase_o_tools.py --tool both --gpu A10G

**QuantSpec ``profile_kernel.py`` (Q·K hist + roofline + Chrome trace)**

    modal run modal_phase_o_tools.py --tool profile_kernel --gpu A10G

Download: Modal dashboard → Volume, or ``modal volume get`` / mount — or copy from function return logs.

Environment: ``modal token new``, optional ``hf-token`` secret (not required for these tools).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent

_MODAL_PHASE_O_GPU = os.environ.get("MODAL_PHASE_O_GPU", "A10G")

HF_VOLUME = modal.Volume.from_name("mlsys-kv-hf-cache-v1", create_if_missing=True)
_PHASE_O_VOL = os.environ.get("MODAL_PHASE_O_VOLUME", "mlsys-kv-phase-o-results-v1")
RESULTS_VOLUME = modal.Volume.from_name(_PHASE_O_VOL, create_if_missing=True)

IMAGE = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime",
        add_python=False,
    )
    .apt_install("build-essential")
    .pip_install(
        "transformers>=4.40.0,<6",
        "numpy>=1.26.0",
        "pytest>=8.0.0",
    )
    .add_local_dir(str(PROJECT_ROOT), remote_path="/repo")
)

app = modal.App("mlsys-kv-phase-o-tools")


@app.function(
    image=IMAGE,
    gpu=_MODAL_PHASE_O_GPU,
    timeout=60 * 60,
    volumes={"/hf": HF_VOLUME, "/results": RESULTS_VOLUME},
)
def run_tools_remote(
    tool: str,
    tuning_profile: str,
    microbench_iters: int,
    trace_out_rel: str,
    modal_resource_tag: str,
    profile_kernel_extra: str,
) -> None:
    """Run profiler and/or microbench; write under ``/results``."""
    os.chdir("/repo")
    repo_src = "/repo/src"
    sys.path.insert(0, repo_src)
    env = os.environ.copy()
    env["HF_HOME"] = "/hf/huggingface"
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])
    env["PYTHONPATH"] = repo_src
    env["MODAL_RESOURCE_TAG"] = modal_resource_tag
    env["KV_KERNEL_TUNING_PROFILE"] = tuning_profile

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", "/repo"],
        env=env,
    )

    out_dir = Path("/results/phase_o")
    out_dir.mkdir(parents=True, exist_ok=True)
    if tool not in ("profile", "microbench", "both", "profile_kernel"):
        raise ValueError(f"tool must be profile|microbench|both|profile_kernel, got {tool!r}")

    lines: list[str] = []

    if tool in ("profile", "both"):
        trace_path = out_dir / "trace.json"
        if trace_out_rel.strip():
            trace_path = Path("/results") / trace_out_rel.lstrip("/")
            trace_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "benchmarks.phase_o_profile",
                "--out",
                str(trace_path),
                "--repeat",
                "3",
            ],
            cwd="/repo",
            env=env,
        )
        lines.append(f"phase_o_profile trace: {trace_path}")
        print(lines[-1], flush=True)

    if tool in ("microbench", "both"):
        log = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.kernel_tune_microbench",
                "--profile",
                tuning_profile,
                "--iters",
                str(microbench_iters),
            ],
            cwd="/repo",
            env=env,
            capture_output=True,
            text=True,
        )
        lines.append("--- kernel_tune_microbench stdout ---")
        lines.append(log.stdout or "")
        if log.stderr:
            lines.append("--- stderr ---")
            lines.append(log.stderr)
        lines.append(f"exit_code={log.returncode}")
        print(log.stdout, flush=True)
        if log.returncode != 0:
            raise RuntimeError(f"microbench failed: {log.stderr}")

    if tool == "profile_kernel":
        trace_pk = out_dir / "profile_kernel_trace.json"
        extra = (profile_kernel_extra or "").strip()
        cmd = [
            sys.executable,
            "-u",
            "/repo/scripts/profile_kernel.py",
            "--device",
            "cuda",
            "--tuning-profile",
            tuning_profile,
            "--trace-out",
            str(trace_pk),
        ]
        if extra:
            import shlex

            cmd.extend(shlex.split(extra))
        log_pk = subprocess.run(cmd, cwd="/repo", env=env, capture_output=True, text=True)
        print(log_pk.stdout, flush=True)
        if log_pk.stderr:
            print(log_pk.stderr, file=sys.stderr, flush=True)
        lines.append("--- profile_kernel.py ---")
        lines.append(log_pk.stdout or "")
        if log_pk.stderr:
            lines.append(log_pk.stderr)
        lines.append(f"exit_code={log_pk.returncode}")
        lines.append(f"trace: {trace_pk}")
        if log_pk.returncode != 0:
            raise RuntimeError("profile_kernel.py failed")

    RESULTS_VOLUME.commit()
    HF_VOLUME.commit()
    summary = "\n".join(lines)
    (out_dir / "modal_phase_o_summary.txt").write_text(summary, encoding="utf-8")
    RESULTS_VOLUME.commit()
    print("[modal_phase_o_tools] done. Summary written to /results/phase_o/modal_phase_o_summary.txt", flush=True)


@app.local_entrypoint()
def main(
    tool: str = "both",
    gpu: str = "A10G",
    tuning_profile: str = "default",
    microbench_iters: int = 200,
    trace_out: str = "phase_o/trace.json",
    profile_kernel_extra: str = "",
) -> None:
    """``tool``: profile | microbench | both | profile_kernel. ``trace_out`` is relative to ``/results``.

    For ``profile_kernel``, optional ``profile_kernel_extra`` is shell-split extra args, e.g.
    ``--include-fused-verifier --seq-len 1024``.
    """
    print(f"[modal_phase_o_tools] volume={_PHASE_O_VOL} tag={gpu}", flush=True)
    print(
        f"[modal_phase_o_tools] decorator_gpu={_MODAL_PHASE_O_GPU} (set MODAL_PHASE_O_GPU=A100, etc.)",
        flush=True,
    )
    run_tools_remote.remote(
        tool=tool,
        tuning_profile=tuning_profile,
        microbench_iters=microbench_iters,
        trace_out_rel=trace_out,
        modal_resource_tag=gpu,
        profile_kernel_extra=profile_kernel_extra,
    )
