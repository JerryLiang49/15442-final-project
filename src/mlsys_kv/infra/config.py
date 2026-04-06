"""YAML-backed run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class RunConfig:
    """Configuration for a single decoding run or smoke test."""

    model_name: str
    seed: int = 42
    max_new_tokens: int = 10
    device: str = "auto"
    output_dir: str = "outputs/raw"
    prompt: str = "Hello, world"
    torch_dtype: str = "float16"
    warmup_runs: int = 1
    num_trials: int = 1
    spec_k: int = 4
    draft_cache_mode: str = "fp16"
    sparse_recent_window: int = 32
    sparse_heavy_hitter_budget: int = 32
    sparse_refresh_interval: int = 4
    sparse_scoring: str = "key_norm"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (e.g. for logging)."""
        return {
            "model_name": self.model_name,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "device": self.device,
            "output_dir": self.output_dir,
            "prompt": self.prompt,
            "torch_dtype": self.torch_dtype,
            "warmup_runs": self.warmup_runs,
            "num_trials": self.num_trials,
            "spec_k": self.spec_k,
            "draft_cache_mode": self.draft_cache_mode,
            "sparse_recent_window": self.sparse_recent_window,
            "sparse_heavy_hitter_budget": self.sparse_heavy_hitter_budget,
            "sparse_refresh_interval": self.sparse_refresh_interval,
            "sparse_scoring": self.sparse_scoring,
        }


def _merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out


def load_run_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """Load ``RunConfig`` from a YAML file, optionally overridden by CLI keyvals."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    if overrides:
        raw = _merge_dict(raw, overrides)
    return RunConfig(
        model_name=str(raw["model_name"]),
        seed=int(raw.get("seed", 42)),
        max_new_tokens=int(raw.get("max_new_tokens", 10)),
        device=str(raw.get("device", "auto")),
        output_dir=str(raw.get("output_dir", "outputs/raw")),
        prompt=str(raw.get("prompt", "")),
        torch_dtype=str(raw.get("torch_dtype", "float16")),
        warmup_runs=int(raw.get("warmup_runs", 1)),
        num_trials=int(raw.get("num_trials", 1)),
        spec_k=int(raw.get("spec_k", 4)),
        draft_cache_mode=str(raw.get("draft_cache_mode", "fp16")),
        sparse_recent_window=int(raw.get("sparse_recent_window", 32)),
        sparse_heavy_hitter_budget=int(raw.get("sparse_heavy_hitter_budget", 32)),
        sparse_refresh_interval=int(raw.get("sparse_refresh_interval", 4)),
        sparse_scoring=str(raw.get("sparse_scoring", "key_norm")),
    )
