"""GPU memory measurement helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def reset_peak_memory_stats(device: torch.device) -> None:
    """Reset CUDA peak memory counters for ``device`` (no-op on CPU)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def max_memory_allocated_bytes(device: torch.device) -> int:
    """Return peak bytes allocated by the CUDA allocator since last reset (0 on CPU)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(device))


@dataclass
class MemorySnapshot:
    """Bundled memory fields for structured logging."""

    peak_cuda_allocated_bytes: int
    logical_kv_cache_bytes: int
    prompt_len_tokens: int
    generated_len_tokens: int

    def to_jsonable(self) -> dict[str, int]:
        return {
            "peak_cuda_allocated_bytes": int(self.peak_cuda_allocated_bytes),
            "logical_kv_cache_bytes": int(self.logical_kv_cache_bytes),
            "prompt_len_tokens": int(self.prompt_len_tokens),
            "generated_len_tokens": int(self.generated_len_tokens),
        }
