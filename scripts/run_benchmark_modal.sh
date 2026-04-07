#!/usr/bin/env bash
# Persistent results on Modal Volume; edit gpu= in modal_sweep.py to match --gpu tag for honest logging.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec modal run modal_sweep.py --sweep configs/benchmark_smoke_modal.yaml --gpu A10G
