#!/usr/bin/env bash
# Local autoregressive smoke (Phase 1). Requires GPU VRAM for Llama-2-7B FP16 unless
# you override --model-name / config.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python -m mlsys_kv.cli smoke --config configs/base.yaml
