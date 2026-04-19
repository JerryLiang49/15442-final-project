"""Profiling hooks for hierarchical KV cache mutation (Phase L).

Timers use :func:`time.perf_counter`. When ``sync_cuda`` is True and a CUDA device is active,
:func:`torch.cuda.synchronize` runs before each stop so GPU work is included in the interval.

**Semantics**

* **append** — time spent appending draft FP16 K/V into CF2 (copy into preallocated buffer or cat).
* **pack** — time to quantize FP16 CF1 into INT4 upper/lower codes + metadata (``quantize_fp16_kv_to_upper_lower`` path).
* **rollback** — CF2 suffix trim (metadata-only when using the fast preallocated CF2 path).
* **rollover** — full rollover (quantize CF1 → hist + CF2→CF1 pointer move + clear CF2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class CacheMutationProfile:
    """Last-call and cumulative seconds for cache hot-path operations."""

    last_append_s: float = 0.0
    last_pack_s: float = 0.0
    last_rollback_s: float = 0.0
    last_rollover_s: float = 0.0

    total_append_s: float = 0.0
    total_pack_s: float = 0.0
    total_rollback_s: float = 0.0
    total_rollover_s: float = 0.0

    n_append: int = 0
    n_pack: int = 0
    n_rollback: int = 0
    n_rollover: int = 0

    sync_cuda: bool = False

    def _maybe_sync(self, device: torch.device | None) -> None:
        if self.sync_cuda and device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    def record_append(self, dt_s: float, *, device: torch.device | None = None) -> None:
        self._maybe_sync(device)
        self.last_append_s = float(dt_s)
        self.total_append_s += float(dt_s)
        self.n_append += 1

    def record_pack(self, dt_s: float, *, device: torch.device | None = None) -> None:
        self._maybe_sync(device)
        self.last_pack_s = float(dt_s)
        self.total_pack_s += float(dt_s)
        self.n_pack += 1

    def record_rollback(self, dt_s: float, *, device: torch.device | None = None) -> None:
        self._maybe_sync(device)
        self.last_rollback_s = float(dt_s)
        self.total_rollback_s += float(dt_s)
        self.n_rollback += 1

    def record_rollover(self, dt_s: float, *, device: torch.device | None = None) -> None:
        self._maybe_sync(device)
        self.last_rollover_s = float(dt_s)
        self.total_rollover_s += float(dt_s)
        self.n_rollover += 1

    def to_dict(self) -> dict[str, float | int]:
        return {
            "last_append_s": self.last_append_s,
            "last_pack_s": self.last_pack_s,
            "last_rollback_s": self.last_rollback_s,
            "last_rollover_s": self.last_rollover_s,
            "total_append_s": self.total_append_s,
            "total_pack_s": self.total_pack_s,
            "total_rollback_s": self.total_rollback_s,
            "total_rollover_s": self.total_rollover_s,
            "n_append": self.n_append,
            "n_pack": self.n_pack,
            "n_rollback": self.n_rollback,
            "n_rollover": self.n_rollover,
        }


@dataclass
class _Timer:
    """Context manager: records elapsed time into a callback."""

    on_done: Callable[..., Any]
    device: torch.device | None
    t0: float = field(default_factory=time.perf_counter)

    def __enter__(self) -> _Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        dt = time.perf_counter() - self.t0
        self.on_done(dt, device=self.device)


def _noop_timer_cb(*_a: object, **_k: object) -> None:
    return None


def timer_append(profile: CacheMutationProfile | None, *, device: torch.device | None) -> _Timer:
    if profile is None:
        return _Timer(_noop_timer_cb, device)
    return _Timer(profile.record_append, device)


def timer_pack(profile: CacheMutationProfile | None, *, device: torch.device | None) -> _Timer:
    if profile is None:
        return _Timer(_noop_timer_cb, device)
    return _Timer(profile.record_pack, device)


def timer_rollback(profile: CacheMutationProfile | None, *, device: torch.device | None) -> _Timer:
    if profile is None:
        return _Timer(_noop_timer_cb, device)
    return _Timer(profile.record_rollback, device)


def timer_rollover(profile: CacheMutationProfile | None, *, device: torch.device | None) -> _Timer:
    if profile is None:
        return _Timer(_noop_timer_cb, device)
    return _Timer(profile.record_rollover, device)
