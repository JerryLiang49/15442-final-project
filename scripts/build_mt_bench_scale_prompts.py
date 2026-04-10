#!/usr/bin/env python3
"""Build `data/mt_bench_scale_prompts.json` for larger-model / longer-context sweeps.

Source material:
  - Starts from `data/mt_bench_subset.json` (same 17 prompts as the v4 paper sweep).
  - Appends synthetic **long** prompts (repeated technical prose) so token counts land in
    medium vs long buckets after **the scale sweep's** `short_token_max` / `medium_token_max`.

Default target model for length checks: **gpt2-xl** (same tokenizer as `gpt2`). GPT-2 models
have **n_positions=1024**, so each prompt is capped so that ``prompt_tokens + max_new_tokens``
fits in 1024 when using ``max_new_tokens=32`` in the scale configs (we target ~560–920 prompt tokens).

Usage::

    PYTHONPATH=src python scripts/build_mt_bench_scale_prompts.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBSET = PROJECT_ROOT / "data" / "mt_bench_subset.json"
OUT = PROJECT_ROOT / "data" / "mt_bench_scale_prompts.json"

# One paragraph (~90–110 tokens on gpt2 BPE); repeated to reach target lengths.
_PARA = (
    "Transformer decoding maintains a key-value cache per layer so attention can attend to prior "
    "tokens without recomputing earlier activations. Speculative decoding proposes multiple future "
    "tokens from a draft model or head, then verifies them in parallel against a target model. "
    "Quantization reduces the memory footprint of stored activations but may require dequantization "
    "before matmuls unless kernels are fused. Sparsity retains only a subset of tokens in the draft "
    "cache, trading compute for retention quality. "
)


def _grow_to_token_range(
    tokenizer,
    *,
    min_tokens: int,
    max_tokens: int,
    prefix: str,
) -> str:
    """Repeat ``_PARA`` until length is in [min_tokens, max_tokens] (inclusive)."""
    text = prefix.strip() + "\n\n"
    while True:
        n = len(tokenizer.encode(text, add_special_tokens=True))
        if n >= min_tokens:
            break
        text += _PARA
    # Trim if overshoot max by adding fewer repeats — binary search not needed; pop paragraphs
    while len(tokenizer.encode(text, add_special_tokens=True)) > max_tokens:
        text = text[: -len(_PARA)]
    if len(tokenizer.encode(text, add_special_tokens=True)) < min_tokens:
        text += _PARA * 2
    return text.strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokenizer",
        default="gpt2-xl",
        help="HF id used only to count tokens (default: gpt2-xl, same as gpt2).",
    )
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=920,
        help="Hard cap on prompt tokens (leave room for max_new_tokens=32 under 1024).",
    )
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    base = json.loads(SUBSET.read_text(encoding="utf-8"))
    if not isinstance(base, list):
        raise SystemExit(f"Expected list in {SUBSET}")

    extras: list[dict[str, str]] = []

    # Medium-length (~260–400 tokens): lands in "medium" when short_max=256
    extras.append(
        {
            "id": "scale-med-01",
            "category": "scale_medium",
            "text": _grow_to_token_range(
                tok,
                min_tokens=280,
                max_tokens=380,
                prefix="Task: answer in one sentence after the context. Context:",
            ),
        }
    )
    extras.append(
        {
            "id": "scale-med-02",
            "category": "scale_medium",
            "text": _grow_to_token_range(
                tok,
                min_tokens=300,
                max_tokens=420,
                prefix="Summarize the following repeated notes on KV caches and speculative decoding.",
            ),
        }
    )

    # Long bucket (> medium_max when medium_max=1024): target ~650–900 tokens
    for i in range(1, 7):
        extras.append(
            {
                "id": f"scale-long-{i:02d}",
                "category": "scale_long",
                "text": _grow_to_token_range(
                    tok,
                    min_tokens=640 + (i * 25),
                    max_tokens=min(900 + i * 5, args.max_prompt_tokens),
                    prefix=f"[Long-context benchmark {i}] Read the technical background, then reply OK. "
                    f"Background:",
                ),
            }
        )

    merged = base + extras
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(merged)} prompts to {OUT}")
    for row in extras:
        n = len(tok.encode(row["text"], add_special_tokens=True))
        print(f"  {row['id']}: {n} tokens")


if __name__ == "__main__":
    main()
