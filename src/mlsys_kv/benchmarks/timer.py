"""CUDA-aware timing helpers for microbenchmarks.

Timing policy (Phase 2):
    * Intervals bracket GPU work with :func:`torch.cuda.synchronize` when ``device`` is CUDA.
    * CPU runs use :func:`time.perf_counter` only.
    * Model load and tokenization are OUTSIDE timed regions unless a caller explicitly wraps them.

This module does not reset CUDA peak memory; see :mod:`mlsys_kv.benchmarks.memory`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TypeVar

import torch

T = TypeVar("T")


def cuda_synchronize(device: torch.device) -> None:
    """Block until CUDA work on ``device`` completes (no-op on CPU)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextmanager
def timed_cuda_interval(device: torch.device) -> Generator[list[float], None, None]:
    """Measure wall time for a region, synchronizing CUDA before and after.

    Yields:
        A one-element list ``out`` where ``out[0]`` is set to elapsed seconds on exit.
    """
    cuda_synchronize(device)
    t0 = time.perf_counter()
    out: list[float] = [0.0]
    try:
        yield out
    finally:
        cuda_synchronize(device)
        out[0] = time.perf_counter() - t0


def measure_cuda(
    device: torch.device,
    fn: Callable[[], T],
) -> tuple[T, float]:
    """Run ``fn()`` and return ``(result, elapsed_seconds)`` with CUDA sync brackets."""
    cuda_synchronize(device)
    t0 = time.perf_counter()
    try:
        out = fn()
    finally:
        cuda_synchronize(device)
    return out, time.perf_counter() - t0
