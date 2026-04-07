"""Unit tests for Phase 9 parallel block greedy verification (batch size 1 logic)."""

from __future__ import annotations

import pytest
import torch

from mlsys_kv.decoding.speculative import (
    first_greedy_speculative_mismatch,
    greedy_speculative_correction_token,
)


def test_mismatch_index_full_accept() -> None:
    v, k = 8, 2
    start_logits = torch.zeros(1, v)
    start_logits[0, 3] = 10.0
    block_logits = torch.zeros(1, k, v)
    block_logits[0, 0, 5] = 10.0
    proposals = torch.tensor([[3, 5]], dtype=torch.long)
    assert first_greedy_speculative_mismatch(start_logits, block_logits, proposals) == k


def test_mismatch_index_reject_first_token() -> None:
    v, k = 8, 3
    start_logits = torch.zeros(1, v)
    start_logits[0, 0] = 10.0
    block_logits = torch.zeros(1, k, v)
    proposals = torch.tensor([[7, 1, 2]], dtype=torch.long)
    assert first_greedy_speculative_mismatch(start_logits, block_logits, proposals) == 0
    corr = greedy_speculative_correction_token(start_logits, block_logits, 0)
    assert corr.shape == (1, 1)
    assert int(corr[0, 0].item()) == 0


def test_mismatch_index_partial_accept() -> None:
    v, k = 10, 4
    start_logits = torch.zeros(1, v)
    start_logits[0, 1] = 10.0
    block_logits = torch.zeros(1, k, v)
    block_logits[0, 0, 2] = 10.0
    block_logits[0, 1, 3] = 10.0
    block_logits[0, 2, 4] = 10.0
    proposals = torch.tensor([[1, 2, 3, 9]], dtype=torch.long)
    assert first_greedy_speculative_mismatch(start_logits, block_logits, proposals) == 3
    corr = greedy_speculative_correction_token(start_logits, block_logits, 3)
    assert int(corr[0, 0].item()) == 4


def test_mismatch_batch_not_one_raises() -> None:
    start = torch.zeros(2, 5)
    block = torch.zeros(2, 3, 5)
    prop = torch.zeros(2, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        first_greedy_speculative_mismatch(start, block, prop)
