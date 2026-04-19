#!/usr/bin/env python3
"""Pre-sweep smoke gate (Phase 15): pytest benchmark_gate + tiny benchmark sweeps.

Runs:
  1. ``pytest -m benchmark_gate`` (includes ``@pytest.mark.slow`` gate tests)
  2. Tiny local sweep from ``configs/benchmark_presweep_local.yaml``
  3. Optional Modal sweep (``--modal``): ``modal run modal_sweep.py --sweep configs/benchmark_presweep_modal.yaml``

Exits non-zero on failure. Validates Phase 2 CSV headers include ``benchmark_label``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _run(cmd: list[str], *, cwd: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def _csv_has_v2_header(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as f:
        line = f.readline()
    return "benchmark_label" in line and "memory_throughput_gb_s" in line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modal",
        action="store_true",
        help="Also run tiny Modal sweep (requires modal CLI and auth).",
    )
    ap.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest benchmark_gate (for debugging harness only).",
    )
    args = ap.parse_args()

    if not args.skip_pytest:
        rc = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_autoregressive.py",
                "tests/test_phase16_analysis.py",
                "-q",
            ],
            cwd=REPO,
        )
        if rc != 0:
            return rc

    sweep_py = [
        sys.executable,
        "-m",
        "mlsys_kv.cli",
        "benchmark-sweep",
        "--config",
        "configs/benchmark_presweep_local.yaml",
    ]
    rc = _run(sweep_py, cwd=REPO)
    if rc != 0:
        return rc

    local_csv = REPO / "outputs" / "benchmarks" / "presweep_local.csv"
    if not _csv_has_v2_header(local_csv):
        print(f"ERROR: missing Phase 15 CSV columns in {local_csv}", file=sys.stderr)
        return 2

    rollup = REPO / "outputs" / "benchmarks" / "processed" / "bench_presweep_local_v1_rollup.json"
    if not rollup.is_file():
        print(f"ERROR: expected processed rollup {rollup}", file=sys.stderr)
        return 2

    if args.modal:
        modal_bin = shutil.which("modal")
        if modal_bin is None:
            print("ERROR: --modal requested but `modal` not on PATH", file=sys.stderr)
            return 2
        rc = subprocess.call(
            [modal_bin, "run", "modal_sweep.py", "--sweep", "configs/benchmark_presweep_modal.yaml"],
            cwd=str(REPO),
        )
        if rc != 0:
            return rc
        print(
            "[gate] Modal sweep dispatched; confirm CSV on results Volume "
            "(benchmark_label + memory_throughput_gb_s columns).",
            flush=True,
        )

    print("[gate] presweep OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
