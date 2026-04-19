"""Serving-oriented speculative decoding (Phase G): request state, logical block layout, reduced orchestration.

**Design**

* :class:`SpeculativeRequestState` tracks one decode session (rounds, KV logical length).
* :class:`LogicalBlockTable` maps token ranges → block ids (vLLM-style pool hook; no GPU allocator yet).
* Dense / hierarchical decoders avoid **double** ``clone_past_key_values`` on the verifier: call sites
  pass ``past_key_values`` directly; :func:`decoding.speculative_dense.verify_block_and_commit` performs
  one clone per round. Legacy double-clone can be re-enabled for benchmarking via
  ``legacy_double_clone_verifier`` on the decoder.

**Remaining Python overhead**

* Token concat, logits snapshots, hierarchical ``prefill_initialize`` from HF after commit (quant path).
* Full CUDA graph capture requires a custom forward; flags document fixed-γ / sync policy for future work.
"""

from .block_kv import BlockPoolConfig, LogicalBlockTable, logical_blocks_for_length
from .request_state import (
    GraphCaptureHints,
    SpeculativeRequestState,
    reset_request_state,
    update_request_state_after_round,
)

__all__ = [
    "BlockPoolConfig",
    "GraphCaptureHints",
    "LogicalBlockTable",
    "SpeculativeRequestState",
    "logical_blocks_for_length",
    "reset_request_state",
    "update_request_state_after_round",
]
