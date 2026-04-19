"""Per-request state for speculative decoding (serving lifecycle + metrics hooks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GraphCaptureHints:
    """Document what is static-shape for future ``torch.compile`` / CUDA graph capture.

    * ``gamma`` fixed for all rounds → block forward ``input_ids`` shape ``[1, gamma]`` (except tail).
    * Verifier past length grows deterministically with accepted tokens.
    """

    fixed_gamma: int
    """Effective γ when not tail-truncated."""

    static_block_forward: bool
    """True when every round uses the same ``gamma`` (``max_new_tokens`` divisible policy optional)."""


@dataclass
class SpeculativeRequestState:
    """One speculative decode session (single prompt + generation)."""

    request_id: str
    prompt_len: int
    max_new_tokens: int
    gamma: int
    total_rounds: int = 0
    total_committed_tokens: int = 0
    logical_kv_len: int = 0
    """Verifier sequence length (tokens) after last committed update."""

    device: torch.device | None = None
    graph_hints: GraphCaptureHints | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def reset_request_state(
    *,
    request_id: str,
    prompt_len: int,
    max_new_tokens: int,
    gamma: int,
    device: torch.device | None = None,
) -> SpeculativeRequestState:
    """Initialize state after tokenizer prefill."""
    gh = GraphCaptureHints(fixed_gamma=gamma, static_block_forward=True)
    return SpeculativeRequestState(
        request_id=request_id,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        logical_kv_len=prompt_len,
        device=device,
        graph_hints=gh,
    )


def update_request_state_after_round(
    st: SpeculativeRequestState,
    *,
    num_committed: int,
    new_kv_len: int,
) -> None:
    """Update after one speculative round (host-side bookkeeping only)."""
    st.total_rounds += 1
    st.total_committed_tokens += num_committed
    st.logical_kv_len = new_kv_len
