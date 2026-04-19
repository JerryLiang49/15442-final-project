"""Optional runtime flags (Phase O): CUDA graphs intent, static workspace reservation.

Full **decode-loop** CUDA Graph capture is model- and shape-specific; these flags record benchmark intent
and gate **microbenchmark** helpers only unless extended."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePerfFlags:
    """Process-wide perf hints (explicit, reproducible via env + YAML)."""

    cuda_graphs_enabled: bool = False
    static_workspace_enabled: bool = False
    reduce_python_dispatch: bool = False


def flags_from_env() -> RuntimePerfFlags:
    return RuntimePerfFlags(
        cuda_graphs_enabled=os.environ.get("KV_CUDA_GRAPHS", "").strip().lower() in ("1", "true", "yes"),
        static_workspace_enabled=os.environ.get("KV_STATIC_WS", "").strip().lower() in ("1", "true", "yes"),
        reduce_python_dispatch=os.environ.get("KV_LEAN_DISPATCH", "").strip().lower()
        in ("1", "true", "yes"),
    )
