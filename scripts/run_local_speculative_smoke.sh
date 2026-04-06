#!/usr/bin/env bash
# Phase 3: self-speculative with uncompressed FP16 draft/verifier (default gpt2).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python -m mlsys_kv.cli speculative --config configs/speculative.yaml "$@"
