"""Phase O — explicit, reproducible Triton tuning for QuantSpec fused kernels.

Separate defaults for **draft** (`q_len=1` fused decode), **verifier** (`q_len=γ` block), and **hist Q·K**
(score) micro-kernels. Selection is by named profile and/or GPU name substring.

**Usage**

* Benchmarks: set ``kernel_tuning_profile`` in YAML; :func:`set_kernel_tuning_from_spec` at run start.
* Tests: :func:`kernel_tuning_scope` or direct :func:`set_active_kernel_tuning`.

CUDA graphs / static workspace: see :mod:`kv_kernels.runtime_options` (flags only; full graph capture is model-specific).
"""

from __future__ import annotations

import copy
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class KernelTuningConfig:
    """Immutable tuning snapshot for logging and A/B comparisons."""

    profile_id: str = "default"
    # Fused draft decode (q_len=1)
    draft_block_d: int = 64
    draft_num_warps: int = 4
    # Fused verifier block (q_len=gamma)
    verifier_block_d: int = 64
    verifier_num_warps: int = 4
    # Hist Q·K overlay kernels (per-seq position program)
    qk_hist_block_d: int = 128
    qk_hist_num_warps: int = 4
    # Documentation
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> KernelTuningConfig:
        d = json.loads(s)
        keys = {f.name for f in KernelTuningConfig.__dataclass_fields__.values()}  # type: ignore[misc]
        return KernelTuningConfig(**{k: v for k, v in d.items() if k in keys})


# Named profiles (tune on target hardware; these are conservative starting points).
_PROFILES: dict[str, KernelTuningConfig] = {
    "default": KernelTuningConfig(profile_id="default"),
    # Slightly larger D-tiles on verifier (γ queries amortize launch)
    "verifier_wide": KernelTuningConfig(
        profile_id="verifier_wide",
        draft_block_d=64,
        draft_num_warps=4,
        verifier_block_d=128,
        verifier_num_warps=8,
        qk_hist_block_d=128,
        qk_hist_num_warps=4,
        notes="Higher verifier BLOCK_D / num_warps; validate on GPU.",
    ),
    # Emphasize draft-only throughput (q_len=1)
    "draft_aggressive": KernelTuningConfig(
        profile_id="draft_aggressive",
        draft_block_d=128,
        draft_num_warps=8,
        verifier_block_d=64,
        verifier_num_warps=4,
        qk_hist_block_d=128,
        qk_hist_num_warps=8,
        notes="Tune draft fused kernel; validate numerics.",
    ),
    # Typical A10-class: balance occupancy
    "a10g_balanced": KernelTuningConfig(
        profile_id="a10g_balanced",
        draft_block_d=64,
        draft_num_warps=4,
        verifier_block_d=64,
        verifier_num_warps=8,
        qk_hist_block_d=128,
        qk_hist_num_warps=4,
        notes="Preset for NVIDIA A10-class; adjust after profiling.",
    ),
    "a100_high_throughput": KernelTuningConfig(
        profile_id="a100_high_throughput",
        draft_block_d=128,
        draft_num_warps=8,
        verifier_block_d=128,
        verifier_num_warps=8,
        qk_hist_block_d=128,
        qk_hist_num_warps=8,
        notes="Preset for large SM count; validate memory pressure.",
    ),
}

# Optional env override for CI/Modal
_ENV_PROFILE = os.environ.get("KV_KERNEL_TUNING_PROFILE", "").strip()


def list_tuning_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES.keys()))


def get_preset_config(profile_id: str) -> KernelTuningConfig:
    """Return a **copy** of a named preset (safe to mutate in tests)."""
    if profile_id not in _PROFILES:
        raise KeyError(f"Unknown kernel tuning profile {profile_id!r}; known: {list_tuning_profiles()}")
    return copy.deepcopy(_PROFILES[profile_id])


_GPU_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"a10", re.I), "a10g_balanced"),
    (re.compile(r"a100", re.I), "a100_high_throughput"),
    (re.compile(r"l40", re.I), "a10g_balanced"),
]


def resolve_tuning_profile(
    explicit: str | None,
    gpu_torch_name: str | None,
) -> KernelTuningConfig:
    """Resolve profile: explicit name > ``KV_KERNEL_TUNING_PROFILE`` > GPU heuristic > default."""
    chosen = (explicit or _ENV_PROFILE or "").strip()
    if chosen and chosen in _PROFILES:
        return get_preset_config(chosen)
    if chosen:
        # Allow direct JSON path in future; for now treat unknown as default + note
        pass
    gname = gpu_torch_name or ""
    for pat, prof in _GPU_HINTS:
        if pat.search(gname):
            return get_preset_config(prof)
    return get_preset_config("default")


_active: KernelTuningConfig | None = None


def set_active_kernel_tuning(cfg: KernelTuningConfig | None) -> None:
    """Set process-wide tuning (``None`` = use preset *default* only inside :func:`active_kernel_tuning`)."""
    global _active
    _active = cfg


def active_kernel_tuning() -> KernelTuningConfig:
    """Current tuning; if unset, returns **default** preset."""
    return _active if _active is not None else get_preset_config("default")


def set_kernel_tuning_from_spec(profile_name: str | None, gpu_torch_name: str | None) -> KernelTuningConfig:
    """Resolve and install active tuning; returns the config applied."""
    cfg = resolve_tuning_profile(profile_name, gpu_torch_name)
    set_active_kernel_tuning(cfg)
    return cfg


@contextmanager
def kernel_tuning_scope(cfg: KernelTuningConfig | None) -> Iterator[None]:
    """Temporarily override active tuning (tests)."""
    prev = _active
    set_active_kernel_tuning(cfg)
    try:
        yield
    finally:
        set_active_kernel_tuning(prev)
