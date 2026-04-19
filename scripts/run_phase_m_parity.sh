#!/usr/bin/env bash
# Phase M — numerical parity for fused Triton kernels (CI / smoke / pre-Modal).
#
# Usage (from repo root):
#   bash scripts/run_phase_m_parity.sh
#
# On a CUDA machine, run only GPU+Triton parity tests (Modal preflight):
#   bash scripts/run_phase_m_parity.sh -m parity_cuda
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
python -m pytest \
  tests/test_kernel_parity_harness.py \
  tests/test_e2e_fused_kernel_parity.py \
  tests/test_fused_verifier_block.py \
  -v --tb=short "$@"
