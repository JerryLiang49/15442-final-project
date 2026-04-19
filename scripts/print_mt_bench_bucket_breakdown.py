#!/usr/bin/env python3
"""Print short/medium/long counts for MT-Bench JSON prompts (GPT-2 tokenizer).

Usage (repo root)::

    PYTHONPATH=src python scripts/print_mt_bench_bucket_breakdown.py \\
        --json data/mt_bench_subset.json \\
        --short-max 64 --medium-max 256

Optional: write markdown::

    PYTHONPATH=src python scripts/print_mt_bench_bucket_breakdown.py \\
        --json data/mt_bench_subset.json --write-md data/mt_bench_subset_buckets.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmarks.context_buckets import ContextBucket, classify_context_bucket, prompt_token_length


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--short-max", type=int, default=64)
    ap.add_argument("--medium-max", type=int, default=256)
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--write-md", type=Path, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = json.loads(args.json.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("# MT-Bench subset: token length buckets\n")
    lines.append(
        f"Tokenizer: `{args.model}`. Buckets: **short** ≤ {args.short_max}, "
        f"**medium** ({args.short_max}, {args.medium_max}], **long** > {args.medium_max}.\n"
    )
    lines.append("| id | category | raw_tokens | bucket |\n")
    lines.append("|---|---|---:|---|\n")

    counts = {ContextBucket.SHORT: 0, ContextBucket.MEDIUM: 0, ContextBucket.LONG: 0}
    for row in rows:
        pid = row.get("id", "")
        cat = row.get("category", "")
        text = str(row.get("text", ""))
        n = prompt_token_length(tok, text)
        b = classify_context_bucket(n, short_max=args.short_max, medium_max=args.medium_max)
        counts[b] += 1
        lines.append(f"| {pid} | {cat} | {n} | {b.value} |\n")

    lines.append("\n## Summary\n\n")
    lines.append(f"- **short**: {counts[ContextBucket.SHORT]}\n")
    lines.append(f"- **medium**: {counts[ContextBucket.MEDIUM]}\n")
    lines.append(f"- **long**: {counts[ContextBucket.LONG]}\n")
    lines.append(f"- **total prompts**: {len(rows)}\n")

    text_out = "".join(lines)
    print(text_out, end="")
    if args.write_md is not None:
        args.write_md.parent.mkdir(parents=True, exist_ok=True)
        args.write_md.write_text(text_out, encoding="utf-8")
        print(f"Wrote {args.write_md}", flush=True)


if __name__ == "__main__":
    main()
