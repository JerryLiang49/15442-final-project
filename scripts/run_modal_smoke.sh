#!/usr/bin/env bash
# Run the same smoke test on Modal (GPU). Requires: modal token, optional HF secret for Llama-2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec modal run modal_app.py
