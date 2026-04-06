"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlsys_kv import __version__
from mlsys_kv.datasets.prompt_loader import load_prompts_file
from mlsys_kv.infra.config import load_run_config
from mlsys_kv.main import run_baseline, run_smoke, run_speculative


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlsys-kv")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Autoregressive smoke test (Phase 1).")
    smoke.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config (default: configs/base.yaml).",
    )
    smoke.add_argument("--model-name", type=str, default=None)
    smoke.add_argument("--seed", type=int, default=None)
    smoke.add_argument("--max-new-tokens", type=int, default=None)
    smoke.add_argument("--device", type=str, default=None)
    smoke.add_argument("--output-dir", type=str, default=None)
    smoke.add_argument("--prompt", type=str, default=None)
    smoke.add_argument("--torch-dtype", type=str, default=None)

    baseline = sub.add_parser(
        "baseline",
        help="Instrumented autoregressive baseline with JSONL metrics (Phase 2).",
    )
    baseline.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config (default: configs/baseline.yaml).",
    )
    baseline.add_argument("--model-name", type=str, default=None)
    baseline.add_argument("--seed", type=int, default=None)
    baseline.add_argument("--max-new-tokens", type=int, default=None)
    baseline.add_argument("--device", type=str, default=None)
    baseline.add_argument("--output-dir", type=str, default=None)
    baseline.add_argument("--torch-dtype", type=str, default=None)
    baseline.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="Prompt text (repeatable). Falls back to config.prompt if none given.",
    )
    baseline.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Text file with one prompt per line.",
    )
    baseline.add_argument("--warmup-runs", type=int, default=None)
    baseline.add_argument("--num-trials", type=int, default=None)

    spec = sub.add_parser(
        "speculative",
        help="Self-speculative decoding with uncompressed FP16 draft KV (Phase 3).",
    )
    spec.add_argument(
        "--config",
        type=str,
        default="configs/speculative.yaml",
        help="Path to YAML config (default: configs/speculative.yaml).",
    )
    spec.add_argument("--model-name", type=str, default=None)
    spec.add_argument("--seed", type=int, default=None)
    spec.add_argument("--max-new-tokens", type=int, default=None)
    spec.add_argument("--device", type=str, default=None)
    spec.add_argument("--output-dir", type=str, default=None)
    spec.add_argument("--torch-dtype", type=str, default=None)
    spec.add_argument("--spec-k", type=int, default=None, help="Draft proposals per round (K).")
    spec.add_argument(
        "--draft-mode",
        type=str,
        default=None,
        help="Draft KV backend: fp16 (default), quant_only, sparse_only, sparse_quant.",
    )
    spec.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="Prompt (repeatable); defaults to config.prompt.",
    )
    spec.add_argument("--prompts-file", type=str, default=None)
    spec.add_argument("--verbose", action="store_true", help="Per-round debug logging.")
    spec.add_argument(
        "--no-verify-match",
        action="store_true",
        help="Skip AR equality check (faster; not recommended for debugging).",
    )
    return p


def _run_with_exit(fn: object) -> None:
    try:
        code = fn()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    raise SystemExit(code)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint used by ``python -m mlsys_kv.cli`` and the ``mlsys-kv`` console script."""
    args = _build_parser().parse_args(argv)
    if args.command == "smoke":
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise SystemExit(f"Config not found: {cfg_path.resolve()}")
        overrides = {}
        if args.model_name is not None:
            overrides["model_name"] = args.model_name
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.max_new_tokens is not None:
            overrides["max_new_tokens"] = args.max_new_tokens
        if args.device is not None:
            overrides["device"] = args.device
        if args.output_dir is not None:
            overrides["output_dir"] = args.output_dir
        if args.prompt is not None:
            overrides["prompt"] = args.prompt
        if args.torch_dtype is not None:
            overrides["torch_dtype"] = args.torch_dtype
        cfg = load_run_config(cfg_path, overrides=overrides or None)
        _run_with_exit(lambda: run_smoke(cfg))

    elif args.command == "baseline":
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise SystemExit(f"Config not found: {cfg_path.resolve()}")
        overrides: dict[str, object] = {}
        if args.model_name is not None:
            overrides["model_name"] = args.model_name
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.max_new_tokens is not None:
            overrides["max_new_tokens"] = args.max_new_tokens
        if args.device is not None:
            overrides["device"] = args.device
        if args.output_dir is not None:
            overrides["output_dir"] = args.output_dir
        if args.torch_dtype is not None:
            overrides["torch_dtype"] = args.torch_dtype
        if args.warmup_runs is not None:
            overrides["warmup_runs"] = args.warmup_runs
        if args.num_trials is not None:
            overrides["num_trials"] = args.num_trials
        cfg = load_run_config(cfg_path, overrides=overrides or None)

        prompts: list[str] = []
        if getattr(args, "prompts", None):
            prompts.extend(args.prompts)
        if args.prompts_file is not None:
            pfile = Path(args.prompts_file)
            if not pfile.is_file():
                raise SystemExit(f"Prompts file not found: {pfile.resolve()}")
            prompts.extend(load_prompts_file(pfile))
        if not prompts:
            prompts = [cfg.prompt]

        _run_with_exit(lambda: run_baseline(cfg, prompts))

    elif args.command == "speculative":
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise SystemExit(f"Config not found: {cfg_path.resolve()}")
        overrides2: dict[str, object] = {}
        if args.model_name is not None:
            overrides2["model_name"] = args.model_name
        if args.seed is not None:
            overrides2["seed"] = args.seed
        if args.max_new_tokens is not None:
            overrides2["max_new_tokens"] = args.max_new_tokens
        if args.device is not None:
            overrides2["device"] = args.device
        if args.output_dir is not None:
            overrides2["output_dir"] = args.output_dir
        if args.torch_dtype is not None:
            overrides2["torch_dtype"] = args.torch_dtype
        if args.spec_k is not None:
            overrides2["spec_k"] = args.spec_k
        if args.draft_mode is not None:
            overrides2["draft_cache_mode"] = args.draft_mode
        cfg = load_run_config(cfg_path, overrides=overrides2 or None)

        prompts_s: list[str] = []
        if getattr(args, "prompts", None):
            prompts_s.extend(args.prompts)
        if args.prompts_file is not None:
            pf = Path(args.prompts_file)
            if not pf.is_file():
                raise SystemExit(f"Prompts file not found: {pf.resolve()}")
            prompts_s.extend(load_prompts_file(pf))
        if not prompts_s:
            prompts_s = [cfg.prompt]

        _run_with_exit(
            lambda: run_speculative(
                cfg,
                prompts_s,
                verbose=bool(args.verbose),
                verify_match=not bool(args.no_verify_match),
            )
        )

    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
