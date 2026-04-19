"""Kernel backend selection (reference vs Triton)."""

from __future__ import annotations

from enum import Enum


class KVKernelBackend(str, Enum):
    """Hot-path implementation for packed KV + Q·K attention scores on history."""

    REFERENCE = "reference"
    """Pure PyTorch / dequant path (correctness baseline)."""

    TRITON = "triton"
    """CUDA Triton kernels (explicit nibble loads in attention; Triton pack for append)."""


def normalize_backend(name: str) -> KVKernelBackend:
    n = name.strip().lower()
    if n in ("ref", "reference", "cpu"):
        return KVKernelBackend.REFERENCE
    if n in ("triton", "cuda"):
        return KVKernelBackend.TRITON
    raise ValueError(f"Unknown kernel backend: {name!r}")
