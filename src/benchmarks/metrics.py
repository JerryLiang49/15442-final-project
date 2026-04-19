"""Structured metric payloads for JSONL experiment logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AutoregressiveRunMetrics:
    """One autoregressive decoding measurement (single prompt, single trial).

    * ``prefill_time_s``: forward over the full prompt (includes computing logits for the first new token).
    * ``decode_step_times_s``: per-forward times for the remaining ``max_new_tokens-1`` greedy steps
      (empty when ``max_new_tokens <= 1``). The first generated token does not add a decode step.
    * ``total_decode_time_s``: sum of ``decode_step_times_s``.
    * ``end_to_end_generation_s``: ``prefill_time_s + total_decode_time_s`` (tokenization / model load excluded).
    * ``new_tokens_per_sec_e2e``: ``new_tokens_generated / end_to_end_generation_s`` when denominator > 0.
    """

    prefill_time_s: float
    decode_step_times_s: list[float]
    total_decode_time_s: float
    end_to_end_generation_s: float
    new_tokens_per_sec_e2e: float | None

    prompt_len_tokens: int
    max_new_tokens: int
    new_tokens_generated: int

    peak_cuda_allocated_bytes: int
    logical_kv_cache_bytes: int

    warmup: bool
    trial_index: int

    def to_jsonable(self) -> dict[str, Any]:
        """Serialize for JSONL (lists and scalars only)."""
        d = asdict(self)
        # Keep a stable key order for readability in raw logs
        return {
            "prefill_time_s": d["prefill_time_s"],
            "decode_step_times_s": d["decode_step_times_s"],
            "total_decode_time_s": d["total_decode_time_s"],
            "end_to_end_generation_s": d["end_to_end_generation_s"],
            "new_tokens_per_sec_e2e": d["new_tokens_per_sec_e2e"],
            "prompt_len_tokens": d["prompt_len_tokens"],
            "max_new_tokens": d["max_new_tokens"],
            "new_tokens_generated": d["new_tokens_generated"],
            "peak_cuda_allocated_bytes": d["peak_cuda_allocated_bytes"],
            "logical_kv_cache_bytes": d["logical_kv_cache_bytes"],
            "warmup": d["warmup"],
            "trial_index": d["trial_index"],
        }


def summarize_decode_latencies(step_times_s: list[float]) -> dict[str, float]:
    """Return simple summary stats for per-step decode latencies (may be empty)."""
    if not step_times_s:
        return {"decode_steps": 0.0, "mean_step_s": 0.0, "max_step_s": 0.0}
    return {
        "decode_steps": float(len(step_times_s)),
        "mean_step_s": float(sum(step_times_s) / len(step_times_s)),
        "max_step_s": float(max(step_times_s)),
    }


@dataclass
class AutoregressiveSummary:
    """Aggregated metrics across trials for one prompt."""

    prompt_id: str
    num_trials: int
    mean_prefill_s: float
    mean_total_decode_s: float
    mean_e2e_s: float
    mean_new_tokens_per_sec: float | None
    mean_peak_cuda_bytes: float
    mean_logical_kv_bytes: float
    per_trial: list[dict[str, Any]] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "num_trials": self.num_trials,
            "mean_prefill_s": self.mean_prefill_s,
            "mean_total_decode_s": self.mean_total_decode_s,
            "mean_e2e_s": self.mean_e2e_s,
            "mean_new_tokens_per_sec": self.mean_new_tokens_per_sec,
            "mean_peak_cuda_bytes": self.mean_peak_cuda_bytes,
            "mean_logical_kv_bytes": self.mean_logical_kv_bytes,
            "per_trial": list(self.per_trial),
        }


def summarize_prompt_trials(prompt_id: str, trials: list[AutoregressiveRunMetrics]) -> AutoregressiveSummary:
    """Aggregate trial metrics for one prompt (excludes warmup rows where ``warmup`` is True)."""
    measured = [t for t in trials if not t.warmup]
    if not measured:
        measured = list(trials)
    n = len(measured)
    if n == 0:
        raise ValueError("summarize_prompt_trials requires at least one trial metric")
    mean_prefill = sum(t.prefill_time_s for t in measured) / n
    mean_decode = sum(t.total_decode_time_s for t in measured) / n
    mean_e2e = sum(t.end_to_end_generation_s for t in measured) / n
    tps_vals = [t.new_tokens_per_sec_e2e for t in measured if t.new_tokens_per_sec_e2e is not None]
    mean_tps = sum(tps_vals) / len(tps_vals) if tps_vals else None
    mean_peak = sum(t.peak_cuda_allocated_bytes for t in measured) / n
    mean_logical = sum(t.logical_kv_cache_bytes for t in measured) / n
    return AutoregressiveSummary(
        prompt_id=prompt_id,
        num_trials=n,
        mean_prefill_s=float(mean_prefill),
        mean_total_decode_s=float(mean_decode),
        mean_e2e_s=float(mean_e2e),
        mean_new_tokens_per_sec=mean_tps,
        mean_peak_cuda_bytes=float(mean_peak),
        mean_logical_kv_bytes=float(mean_logical),
        per_trial=[t.to_jsonable() for t in measured],
    )
