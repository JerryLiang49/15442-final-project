#!/usr/bin/env python3
"""Quick Phase 7 smoke run: self-speculative decode with ``sparse_quant`` draft cache.

Uses the same stack as ``mlsys-kv speculative`` (JSONL under ``output_dir``). Handy for
trying 1–2 prompts before a full benchmark sweep.

Examples::

    # Editable install from repo root: pip install -e .
    python scripts/phase7_smoke_prompts.py

    python scripts/phase7_smoke_prompts.py --prompt "The number one is" --prompt "Hello, I am"

    python scripts/phase7_smoke_prompts.py --config configs/speculative.yaml --max-new-tokens 20 --verbose

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Phase 7 sparse_quant draft smoke (1+ prompts).")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "speculative.yaml",
        help="Base YAML (default: configs/speculative.yaml).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="Prompt text (repeatable). Default: two short built-ins.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--spec-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sparse-scoring", type=str, default=None, choices=["key_norm", "attention"])
    parser.add_argument("--no-verify-match", action="store_true", help="Skip AR equality check.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    from mlsys_kv.infra.config import load_run_config
    from mlsys_kv.main import run_speculative

    overrides: dict[str, object] = {"draft_cache_mode": "sparse_quant"}
    if args.model_name is not None:
        overrides["model_name"] = args.model_name
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = args.max_new_tokens
    if args.spec_k is not None:
        overrides["spec_k"] = args.spec_k
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.sparse_scoring is not None:
        overrides["sparse_scoring"] = args.sparse_scoring

    cfg = load_run_config(args.config, overrides=overrides)
    prompts = list(args.prompts) if args.prompts else [
        "The number one is",
        "Hello, I am",
    ]

    return run_speculative(
        cfg,
        prompts,
        verbose=args.verbose,
        verify_match=not args.no_verify_match,
    )


if __name__ == "__main__":
    raise SystemExit(main())
