"""Decoding algorithms (autoregressive now; speculative in later phases)."""

from mlsys_kv.decoding.autoregressive import (
    autoregressive_smoke_generate,
    decode_greedy_autoregressive,
    model_device,
    reference_greedy_generate_ids,
)
from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.draft_factory import create_draft_cache, draft_cache_from_verifier_snapshot
from mlsys_kv.decoding.speculative import (
    SpeculativeDecodeResult,
    SpeculativeDecoder,
    SpeculativeMetrics,
    commit_tokens_to_sequence,
    propose_draft_tokens,
    verify_greedy_proposals,
)

__all__ = [
    "DraftCacheMode",
    "SpeculativeDecodeResult",
    "SpeculativeDecoder",
    "SpeculativeMetrics",
    "create_draft_cache",
    "draft_cache_from_verifier_snapshot",
    "autoregressive_smoke_generate",
    "commit_tokens_to_sequence",
    "decode_greedy_autoregressive",
    "model_device",
    "propose_draft_tokens",
    "reference_greedy_generate_ids",
    "verify_greedy_proposals",
]
