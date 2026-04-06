"""Abstract KV cache interface for verifier and draft paths.

**Hugging Face adapter semantics**

Causal LMs return full ``past_key_values`` after each forward (one new token when
``input_ids`` has shape ``[batch, 1]``). For this stack, *appending* one decode
step means:

1. Forward with ``past_key_values=self.get_attention_kv()``.
2. Call ``append_from_forward_output(outputs.past_key_values)``.

So the cache holds the **entire** prefix KV each time, not a delta only. Future
quantized/sparse backends may store a compressed representation but must still
implement :meth:`get_attention_kv` to supply whatever the model’s ``forward``
expects (or a wrapper around ``forward``).

**Tensor layout**

Layers follow Hugging Face order: index ``ℓ`` matches transformer layer ``ℓ``.
Keys/values are model-derived; typical LLaMA-like shape is
``[batch, n_heads, seq_len, head_dim]`` (layout may vary by model / cache class).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KVCacheBase(ABC):
    """Pluggable KV state for decoding (draft backends implement this; verifier uses FP16)."""

    @abstractmethod
    def append_from_forward_output(self, past_key_values: Any) -> None:
        """Absorb KV from a single ``model(...)`` call that advanced the sequence by one token.

        For native HF caches this typically **replaces** the full ``past_key_values``
        object returned by the model (not a sparse delta).
        """

    @abstractmethod
    def get_attention_kv(self) -> Any | None:
        """Attention-ready KV for the **next** forward pass.

        Returns:
            Value for ``past_key_values=`` on the next call, or ``None`` when no prefix
            has been stored yet (prefill will use ``past_key_values=None``).
        """

    @abstractmethod
    def memory_bytes(self) -> int:
        """Logical storage size of this cache in bytes (implementation-defined, for experiments)."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Small JSON-serializable summary (layer count, seq length estimate, mode, ...)."""
