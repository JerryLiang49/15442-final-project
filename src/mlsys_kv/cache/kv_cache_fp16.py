"""Full-precision KV cache wrapping Hugging Face ``past_key_values`` (verifier + FP16 draft)."""

from __future__ import annotations

from typing import Any

import torch

from mlsys_kv.cache.kv_cache_base import KVCacheBase


def _iter_kv_tensors(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Flatten HF ``past_key_values`` into ``(key, value)`` pairs per layer."""
    if past_key_values is None:
        return []
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    key_cache = getattr(past_key_values, "key_cache", None)
    val_cache = getattr(past_key_values, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(val_cache, list):
        for k, v in zip(key_cache, val_cache):
            if k is None or v is None:
                continue
            pairs.append((k, v))
        return pairs

    for layer in past_key_values:
        if layer is None:
            continue
        k_t, v_t = layer[0], layer[1]
        pairs.append((k_t, v_t))
    return pairs


class KVCacheFP16(KVCacheBase):
    """Stores **references** to HF ``past_key_values`` (same ownership as pre-refactor).

    The model owns the tensor graph; this wrapper holds the latest object returned from
    ``forward``. Callers must not mutate tensors in-place unless they match HF contracts.
    """

    __slots__ = ("_past",)

    def __init__(self) -> None:
        self._past: Any | None = None

    def append_from_forward_output(self, past_key_values: Any) -> None:
        self._past = past_key_values

    def get_attention_kv(self) -> Any | None:
        return self._past

    def memory_bytes(self) -> int:
        total = 0
        for k, v in _iter_kv_tensors(self._past):
            total += int(k.numel()) * int(k.element_size())
            total += int(v.numel()) * int(v.element_size())
        return int(total)

    def stats(self) -> dict[str, Any]:
        pairs = _iter_kv_tensors(self._past)
        seq_len: int | None = None
        if pairs:
            k0 = pairs[0][0]
            if k0.dim() >= 2:
                seq_len = int(k0.shape[-2])
        return {
            "type": "KVCacheFP16",
            "num_layers": len(pairs),
            "sequence_length_est": seq_len,
            "memory_bytes": self.memory_bytes(),
        }

    def sync_from_forward_output(self, past_key_values: Any) -> None:
        """Alias for :meth:`append_from_forward_output` (legacy name)."""

        self.append_from_forward_output(past_key_values)

    def past_key_values_for_forward(self) -> Any | None:
        """Alias for :meth:`get_attention_kv` (legacy name)."""

        return self.get_attention_kv()
