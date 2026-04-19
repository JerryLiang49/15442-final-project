"""Rollback-safe double FP16 recent buffer (CF1 / CF2) for QuantSpec-style KV.

**Why two buffers**

* **CF1** holds *committed* recent context in full precision so the verifier sees accurate KV
  on the settled prefix (better acceptance than quantizing immediately).
* **CF2** holds *speculative* tokens only. Rejection **trims CF2** — no historical INT4 rewrite and
  no re-quantize of a discarded suffix.

**Rollback**

* Cheap: slice or drop the CF2 tensors only (``trim_cf2``).
* Explicit: ``reject_speculative_suffix(keep_prefix)`` removes ``cf2_len - keep_prefix`` tokens.

**Rollover (exact rule)**

1. Quantize **only** the current CF1 FP16 block to upper/lower INT4 and **concatenate** onto history.
2. ``hist_len += cf1_len``.
3. Move **CF2 → CF1** (pointer/tensor move), set **cf2_len ← 0**.

Speculative tokens are never quantized until they have been promoted into CF1 and a rollover runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .cache_mutation_profile import CacheMutationProfile
from .hierarchical_kv_store import HierarchicalKVStore


@dataclass
class RecentBufferStats:
    """Counters for decoding instrumentation."""

    rollover_count: int = 0
    """Number of :meth:`RecentBufferManager.rollover` calls that advanced state."""

    reject_trim_events: int = 0
    """How many times speculative suffix was trimmed (CF2-only)."""

    rejected_tokens_total: int = 0
    """Sum of CF2 tokens dropped across all trims."""

    def reset(self) -> None:
        self.rollover_count = 0
        self.reject_trim_events = 0
        self.rejected_tokens_total = 0


@dataclass
class RecentBufferOccupancy:
    """Snapshot of FP16 + history lengths."""

    hist_len: int
    cf1_len: int
    cf2_len: int
    cf1_max_tokens: int
    logical_committed_seq_len: int
    logical_draft_seq_len: int


class RecentBufferManager:
    """First-class API over :class:`HierarchicalKVStore` for CF1/CF2 lifecycle + stats."""

    def __init__(self, store: HierarchicalKVStore) -> None:
        self._store = store
        self._stats = RecentBufferStats()

    @property
    def store(self) -> HierarchicalKVStore:
        return self._store

    @property
    def stats(self) -> RecentBufferStats:
        return self._stats

    def occupancy(self) -> RecentBufferOccupancy:
        s = self._store
        return RecentBufferOccupancy(
            hist_len=s.hist_len,
            cf1_len=s.cf1_len,
            cf2_len=s.cf2_len,
            cf1_max_tokens=s.cf1_max_tokens,
            logical_committed_seq_len=s.logical_committed_seq_len(),
            logical_draft_seq_len=s.logical_draft_seq_len(),
        )

    def instrumentation_dict(self) -> dict[str, Any]:
        o = self.occupancy()
        out: dict[str, Any] = {
            "rollover_count": self._stats.rollover_count,
            "reject_trim_events": self._stats.reject_trim_events,
            "rejected_tokens_total": self._stats.rejected_tokens_total,
            "hist_len": o.hist_len,
            "cf1_len": o.cf1_len,
            "cf2_len": o.cf2_len,
            "cf1_max_tokens": o.cf1_max_tokens,
            "logical_committed_seq_len": o.logical_committed_seq_len,
            "logical_draft_seq_len": o.logical_draft_seq_len,
        }
        mp = self._store.mutation_profile
        if mp is not None:
            out.update({f"mutation_{k}": v for k, v in mp.to_dict().items()})
        return out

    @property
    def mutation_profile(self) -> CacheMutationProfile | None:
        """Phase L timings when :class:`~cache.hierarchical_kv_store.HierarchicalKVStore` was constructed with profiling."""
        return self._store.mutation_profile

    def prefill_initialize(
        self,
        layers_k: list[torch.Tensor],
        layers_v: list[torch.Tensor],
        *,
        recent_tokens_cap: int | None = None,
    ) -> None:
        """Load prompt KV: INT4 history for the prefix, FP16 **CF1** for the tail.

        CF1 is filled with ``min(S, cap)`` tokens where ``cap`` defaults to ``cf1_max_tokens``,
        i.e. the largest committed FP16 window allowed by policy (full recent buffer when the
        prompt is long enough).
        """
        cap = int(recent_tokens_cap if recent_tokens_cap is not None else self._store.cf1_max_tokens)
        self._store.prefill_from_fp16(layers_k, layers_v, recent_tokens_cap=cap)

    def append_draft(self, layers_k: list[torch.Tensor], layers_v: list[torch.Tensor]) -> None:
        """Append newly drafted tokens to **CF2** only."""
        self._store.append_cf2_fp16(layers_k, layers_v)

    def accept_verified_prefix(self, num_tokens: int) -> None:
        """Move the first ``num_tokens`` of CF2 onto CF1 (verification acceptance)."""
        self._store.commit_cf2_prefix_to_cf1(num_tokens)

    def reject_speculative_suffix(self, keep_prefix_tokens: int) -> None:
        """Trim CF2 to the first ``keep_prefix_tokens`` tokens; drop the rest.

        Only CF2 is modified. Historical INT4 and CF1 are unchanged.
        Records instrumentation when tokens are actually removed.
        """
        before = self._store.cf2_len
        if keep_prefix_tokens >= before:
            return
        dropped = before - keep_prefix_tokens
        self._store.trim_cf2(keep_prefix_tokens)
        self._stats.reject_trim_events += 1
        self._stats.rejected_tokens_total += dropped

    def clear_speculative(self) -> None:
        """Remove all CF2 tokens (full rejection)."""
        self.reject_speculative_suffix(0)

    def rollover(self) -> None:
        """Quantize CF1 into history, shift CF2 → CF1, clear CF2. See module docstring."""
        if self._store.cf1_len == 0 and self._store.cf2_len == 0:
            return
        self._store.rollover()
        self._stats.rollover_count += 1

    def draft_view(self):
        return self._store.draft_view()

    def draft_view_without_cf2(self):
        """History + CF1 only (same as :meth:`draft_view` when CF2 is empty)."""
        return self._store.draft_view_without_cf2()

    def target_view(self):
        return self._store.target_view()

    def target_view_without_cf2(self):
        """Verifier past **before** a draft block (no CF2 speculative tail)."""
        return self._store.target_view_without_cf2()
