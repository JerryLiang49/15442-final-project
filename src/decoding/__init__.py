"""Decoding entrypoints for the active stack."""

from .autoregressive import (
    AutoregressiveDecodeResult,
    autoregressive_smoke_generate,
    decode_greedy_autoregressive,
    model_device,
)
from .speculative_dense import (
    DenseSpeculativeDecodeResult,
    DenseSpeculativeMetrics,
    SpeculativeDecoderDense,
    concat_committed,
    draft_cache_from_verifier_snapshot,
    draft_greedy_proposals,
    first_mismatch_index_greedy,
    greedy_correction_token,
    verify_block_and_commit,
)
from .speculative_dense_hierarchical import (
    HierarchicalRoundDebug,
    SpeculativeDecoderDenseHierarchical,
    SpeculativeHierarchicalDecodeResult,
    dump_hierarchical_debug_state,
    format_hierarchical_debug_line,
    sync_hierarchical_store_from_hf_past,
)

__all__ = [
    "AutoregressiveDecodeResult",
    "DenseSpeculativeDecodeResult",
    "DenseSpeculativeMetrics",
    "SpeculativeDecoderDense",
    "autoregressive_smoke_generate",
    "concat_committed",
    "decode_greedy_autoregressive",
    "draft_cache_from_verifier_snapshot",
    "draft_greedy_proposals",
    "first_mismatch_index_greedy",
    "greedy_correction_token",
    "model_device",
    "verify_block_and_commit",
    "HierarchicalRoundDebug",
    "SpeculativeDecoderDenseHierarchical",
    "SpeculativeHierarchicalDecodeResult",
    "format_hierarchical_debug_line",
    "dump_hierarchical_debug_state",
    "sync_hierarchical_store_from_hf_past",
]
