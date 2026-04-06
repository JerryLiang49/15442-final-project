#!/usr/bin/env bash
# Phase 2: instrumented autoregressive baseline (defaults to configs/baseline.yaml → gpt2).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python -m mlsys_kv.cli baseline --config configs/baseline.yaml
