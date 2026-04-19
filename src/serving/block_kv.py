"""Logical block / page table for long-context KV (vLLM-style integration hook).

This module does **not** allocate GPU memory. It records how logical token positions map to abstract
**block ids** so a future backend (vLLM paged attention, custom block pool) can attach without changing
the speculative algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockPoolConfig:
    """Block size in **tokens** per logical block (attention positions)."""

    block_tokens: int = 16
    """Must be > 0. Typical values: 16, 32, 64 (backend-specific)."""


@dataclass
class LogicalBlockTable:
    """Maps token index → block id (contiguous zero-based ids for a single request)."""

    config: BlockPoolConfig
    num_blocks: int
    """``ceil(seq_len / block_tokens)`` for the current logical length."""

    def block_id_for_token(self, token_index: int) -> int:
        if token_index < 0:
            raise ValueError("token_index must be >= 0")
        bt = self.config.block_tokens
        return token_index // bt


def logical_blocks_for_length(seq_len: int, config: BlockPoolConfig) -> LogicalBlockTable:
    """How many blocks cover ``[0 .. seq_len-1]``."""
    if seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    bt = config.block_tokens
    n = (seq_len + bt - 1) // bt if seq_len > 0 else 0
    return LogicalBlockTable(config=config, num_blocks=n)
