#!/usr/bin/env python3
"""List benchmark sweep keys that are not yet present in a CSV (for resume planning).

Uses the same key tuple as ``experiment_runner.load_completed_keys`` / resume skip logic.

Example::

    PYTHONPATH=src python scripts/sweep_missing_report.py \\
      --config configs/benchmark_full_modal_v4.yaml \\
      --csv results/sweep_full_modal_v4.csv

With ``--write missing_keys.csv``, writes one row per missing key (columns match key fields).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from mlsys_kv.benchmarks.context_buckets import classify_context_bucket, prompt_token_length
from mlsys_kv.benchmarks.experiment_runner import (
    expand_sweep_grid,
    load_completed_keys,
    max_new_tokens_from_sweep_cfg,
)
from mlsys_kv.benchmarks.experiment_schema import SWEEP_INPUT_MODES, canonical_sweep_mode
from mlsys_kv.datasets.mt_bench import load_mt_bench_subset


def expected_keys_from_config(cfg: dict) -> set[tuple[object, ...]]:
    modes_raw = [str(m) for m in (cfg.get("modes") or [])]
    unknown = set(modes_raw) - SWEEP_INPUT_MODES
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")
    modes_canon = [canonical_sweep_mode(m) for m in modes_raw]

    k_values = [int(x) for x in (cfg.get("k_values") or [1])]
    sparsity_budgets = [float(x) for x in (cfg.get("sparsity_budgets") or [0.0])]
    quant_bits_list = [int(x) for x in (cfg.get("quant_bits") or [8])]
    strict_labeled_grid = bool(cfg.get("strict_labeled_grid", True))

    grid = expand_sweep_grid(
        modes_canon,
        k_values,
        sparsity_budgets,
        quant_bits_list,
        strict_labeled_grid=strict_labeled_grid,
    )

    mt_path = Path(cfg.get("mt_bench_path", "data/mt_bench_subset.json"))
    max_prompts = cfg.get("max_prompts")
    short_max = int(cfg.get("short_token_max", 64))
    medium_max = int(cfg.get("medium_token_max", 256))
    buckets_filter = {str(b).lower() for b in (cfg.get("context_buckets") or ["short", "medium", "long"])}
    num_trials = int(cfg.get("num_trials", 1))
    max_nt_list = max_new_tokens_from_sweep_cfg(cfg)

    prompts_all = load_mt_bench_subset(mt_path)
    if max_prompts is not None:
        prompts_all = prompts_all[: int(max_prompts)]

    from transformers import AutoTokenizer

    model_name = str(cfg["model_name"])
    tok = AutoTokenizer.from_pretrained(model_name)
    if getattr(tok, "pad_token_id", None) is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    indexed: list[tuple] = []
    for i, pr in enumerate(prompts_all):
        ntok = prompt_token_length(tok, pr.text)
        bucket = classify_context_bucket(ntok, short_max=short_max, medium_max=medium_max)
        if bucket.value not in buckets_filter:
            continue
        indexed.append((i, pr, bucket, ntok))

    keys: set[tuple[object, ...]] = set()
    for max_nt in max_nt_list:
        for mode, spec_k, sparsity_budg, qb_req in grid:
            qb_req_i = int(qb_req)
            for _pi, pr, bucket, _ntok in indexed:
                for trial in range(num_trials):
                    key = (
                        pr.id,
                        mode,
                        str(spec_k),
                        str(sparsity_budg),
                        str(qb_req_i),
                        bucket.value,
                        str(max_nt),
                        str(trial),
                    )
                    keys.add(key)
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="Sweep YAML (same as benchmark run)")
    ap.add_argument("--csv", type=Path, required=True, help="Existing sweep CSV")
    ap.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Optional path to write missing keys as CSV",
    )
    ap.add_argument("--limit", type=int, default=30, help="How many missing keys to print (default 30)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    expected = expected_keys_from_config(cfg)
    completed = load_completed_keys(args.csv, retry_failures=bool(cfg.get("retry_failures", False)))
    missing = expected - completed

    print(f"Expected keys: {len(expected)}")
    print(f"Completed (ok in CSV): {len(completed)}")
    print(f"Missing: {len(missing)}")
    if not missing:
        print("Sweep is complete for this config + CSV.")
        return 0

    # Group missing by mode for a short summary
    by_mode: dict[str, int] = {}
    for k in missing:
        m = str(k[1])
        by_mode[m] = by_mode.get(m, 0) + 1
    print("\nMissing count by canonical mode:")
    for m in sorted(by_mode.keys()):
        print(f"  {m}: {by_mode[m]}")

    print(
        f"\nFirst {min(args.limit, len(missing))} missing keys "
        "(prompt_id, mode, spec_k, sparsity, quant, bucket, max_new_tokens, trial):"
    )
    for i, k in enumerate(sorted(missing, key=lambda x: (str(x[0]), str(x[1]), x[2], x[3], x[4], x[5], x[6]))):
        if i >= args.limit:
            break
        print(" ", k)

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with args.write.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "prompt_id",
                    "mode",
                    "spec_k",
                    "sparsity_budget",
                    "quant_bits_requested",
                    "context_bucket",
                    "max_new_tokens",
                    "trial_index",
                ]
            )
            for k in sorted(missing, key=lambda x: (str(x[0]), str(x[1]), x[2], x[3], x[4], x[5], x[6], x[7])):
                w.writerow(k)
        print(f"\nWrote {len(missing)} rows to {args.write.resolve()}")

    print(
        "\nTo run **only** missing rows: use the **same** sweep YAML with `resume: true` and the **same** "
        "`output_csv` path — the harness skips completed keys automatically."
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
