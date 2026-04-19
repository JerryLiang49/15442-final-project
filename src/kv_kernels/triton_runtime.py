"""Triton availability (optional dependency)."""

from __future__ import annotations

from typing import Any

_triton: Any | None
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _triton = triton
except ImportError:
    _triton = None


def triton_available() -> bool:
    return _triton is not None


def require_triton():
    if _triton is None:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")
