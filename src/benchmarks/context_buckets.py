"""Token-length buckets (short / medium / long) for sweep filtering and reporting."""

from __future__ import annotations

from enum import Enum

from transformers import PreTrainedTokenizerBase


class ContextBucket(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


def prompt_token_length(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count prompt tokens (no special-token tricks; matches AR prefill length style)."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    return int(enc["input_ids"].shape[1])


def classify_context_bucket(
    num_tokens: int,
    *,
    short_max: int,
    medium_max: int,
) -> ContextBucket:
    """Assign bucket using upper bounds ``(short_max]``, ``(short_max, medium_max]``, else ``long``."""
    if num_tokens <= short_max:
        return ContextBucket.SHORT
    if num_tokens <= medium_max:
        return ContextBucket.MEDIUM
    return ContextBucket.LONG
