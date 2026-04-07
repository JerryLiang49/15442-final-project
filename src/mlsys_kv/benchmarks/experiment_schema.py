"""Frozen experiment-facing schema for Phase 14–15 (do not rename fields casually).

**Phase 14 gate:** ``pytest -m benchmark_gate --slow`` (``tests/test_benchmark_gate_phase14.py``).

**Phase 15:** YAML ``modes:`` may use legacy names (:data:`SWEEP_LOGICAL_MODES`) or explicit
:data:`BENCHMARK_LABEL_*` aliases (e.g. ``ar``, ``spec_fp16``) — see :func:`canonical_sweep_mode`.

Each CSV row includes :data:`benchmark_label` for publishable charts and ``mode`` as the **canonical**
internal key (stable for resume). See ``docs/BENCHMARK_PHASE15.md`` for the staged matrix.
"""

from __future__ import annotations

from typing import Any, Final

# --- Canonical sweep modes (internal; used with resolve_speculative_mode) ---

SWEEP_MODE_AUTOREGRESSIVE: Final[str] = "autoregressive"
SWEEP_MODE_SPECULATIVE_FP16: Final[str] = "speculative_fp16"
SWEEP_MODE_QUANT_ONLY: Final[str] = "quant_only"
SWEEP_MODE_SPARSE_ONLY: Final[str] = "sparse_only"
SWEEP_MODE_SPARSE_QUANT: Final[str] = "sparse_quant"

SWEEP_LOGICAL_MODES: Final[frozenset[str]] = frozenset(
    {
        SWEEP_MODE_AUTOREGRESSIVE,
        SWEEP_MODE_SPECULATIVE_FP16,
        SWEEP_MODE_QUANT_ONLY,
        SWEEP_MODE_SPARSE_ONLY,
        SWEEP_MODE_SPARSE_QUANT,
    }
)

# --- Phase 15 explicit benchmark labels (YAML aliases → same canonical mode) ---

BENCHMARK_LABEL_AR: Final[str] = "ar"
BENCHMARK_LABEL_SPEC_FP16: Final[str] = "spec_fp16"
BENCHMARK_LABEL_SPEC_QUANT_MEMONLY: Final[str] = "spec_quant_memonly"
BENCHMARK_LABEL_SPEC_SPARSE: Final[str] = "spec_sparse"
BENCHMARK_LABEL_SPEC_SPARSE_QUANT_MEMONLY: Final[str] = "spec_sparse_quant_memonly"
# Reserved for future native INT8/INT4 attention (not implemented):
BENCHMARK_LABEL_SPEC_QUANT_RUNTIME: Final[str] = "spec_quant_runtime"
BENCHMARK_LABEL_SPEC_SPARSE_QUANT_RUNTIME: Final[str] = "spec_sparse_quant_runtime"

_MODE_ALIASE_TO_CANONICAL: Final[dict[str, str]] = {
    BENCHMARK_LABEL_AR: SWEEP_MODE_AUTOREGRESSIVE,
    BENCHMARK_LABEL_SPEC_FP16: SWEEP_MODE_SPECULATIVE_FP16,
    BENCHMARK_LABEL_SPEC_QUANT_MEMONLY: SWEEP_MODE_QUANT_ONLY,
    BENCHMARK_LABEL_SPEC_SPARSE: SWEEP_MODE_SPARSE_ONLY,
    BENCHMARK_LABEL_SPEC_SPARSE_QUANT_MEMONLY: SWEEP_MODE_SPARSE_QUANT,
}

BENCHMARK_LABEL_BY_CANONICAL: Final[dict[str, str]] = {v: k for k, v in _MODE_ALIASE_TO_CANONICAL.items()}

SWEEP_INPUT_MODES: Final[frozenset[str]] = frozenset(SWEEP_LOGICAL_MODES) | frozenset(
    _MODE_ALIASE_TO_CANONICAL.keys()
)

# Mirrors :class:`~mlsys_kv.cache.draft_cache_mode.DraftCacheMode` ``.value`` strings.
DRAFT_CACHE_MODE_VALUES: Final[frozenset[str]] = frozenset(
    {"fp16", "quant_only", "sparse_only", "sparse_quant"}
)

QUANTIZATION_TYPE_NONE: Final[str] = "none"
QUANTIZATION_TYPE_MEMORY_ONLY: Final[str] = "memory_only"
QUANTIZATION_TYPE_RUNTIME_ACCELERATED: Final[str] = "runtime_accelerated"  # reserved / not used

SPARSE_INTEGRATION_VERSION: Final[str] = "SparseHFCacheIntegrator"

EXPERIMENT_SCHEMA_VERSION: Final[str] = "2"


def canonical_sweep_mode(raw: str) -> str:
    """Map YAML ``modes`` entry to canonical mode string (:data:`SWEEP_LOGICAL_MODES`)."""
    s = (raw or "").strip()
    if s in _MODE_ALIASE_TO_CANONICAL:
        return _MODE_ALIASE_TO_CANONICAL[s]
    if s in SWEEP_LOGICAL_MODES:
        return s
    raise ValueError(
        f"Unknown benchmark mode {raw!r}; use one of {sorted(SWEEP_INPUT_MODES)} "
        f"(see experiment_schema / docs/BENCHMARK_PHASE15.md)"
    )


def benchmark_label_for_canonical_mode(canonical: str) -> str:
    """Return stable Phase 15 label for CSV ``benchmark_label`` column."""
    if canonical not in BENCHMARK_LABEL_BY_CANONICAL:
        raise ValueError(f"Not a canonical sweep mode: {canonical!r}")
    return BENCHMARK_LABEL_BY_CANONICAL[canonical]


def quantization_type_for_row(*, canonical_mode: str, quant_bits_effective: int) -> str:
    """Phase 13 semantics: draft INT quant is memory-only; FP16-equivalent draft has ``none``."""
    if canonical_mode == SWEEP_MODE_AUTOREGRESSIVE:
        return QUANTIZATION_TYPE_NONE
    if canonical_mode in (SWEEP_MODE_SPECULATIVE_FP16, SWEEP_MODE_SPARSE_ONLY):
        return QUANTIZATION_TYPE_NONE
    if canonical_mode == SWEEP_MODE_QUANT_ONLY:
        if int(quant_bits_effective) == 16:
            return QUANTIZATION_TYPE_NONE
        return QUANTIZATION_TYPE_MEMORY_ONLY
    if canonical_mode == SWEEP_MODE_SPARSE_QUANT:
        if int(quant_bits_effective) == 16:
            return QUANTIZATION_TYPE_NONE
        return QUANTIZATION_TYPE_MEMORY_ONLY
    return QUANTIZATION_TYPE_NONE


# --- Main sweep CSV columns (stable order; append-only for v2+) ---

BENCHMARK_CSV_FIELDNAMES: list[str] = [
    "sweep_id",
    "timestamp_utc",
    "status",
    "failure_reason",
    "prompt_id",
    "prompt_idx",
    "context_bucket",
    "prompt_len_tokens",
    "mode",
    "benchmark_label",
    "spec_k",
    "sparsity_budget",
    "quant_bits_requested",
    "quant_bits_effective",
    "recent_window",
    "heavy_hitter_budget",
    "sparse_refresh_interval",
    "sparse_scoring",
    "sparse_config_json",
    "draft_cache_mode_resolved",
    "quant_notes",
    "trial_index",
    "warmup",
    "latency_e2e_s",
    "latency_per_new_token_s",
    "acceptance_rate",
    "tokens_per_sec",
    "logical_draft_kv_bytes",
    "logical_verifier_kv_bytes",
    "gpu_torch_name",
    "modal_resource_tag",
    "model_name",
    "max_new_tokens",
    "verify_match",
    "device_type",
    "experiment_schema_version",
    "is_parallel_verification",
    "quantization_type",
    "sparse_integration_version",
    "gpu_peak_memory_bytes_before_run",
    "gpu_peak_memory_bytes_after_run",
    "model_weights_gb",
    "kv_cache_size_gb",
    "effective_memory_bandwidth_gb_s",
    "memory_throughput_gb_s",
    "draft_latency_total_s",
    "verify_latency_total_s",
]

# --- SpeculativeDecoder metrics JSON (see SpeculativeMetrics.to_jsonable) ---

SPECULATIVE_METRICS_JSON_KEYS: Final[tuple[str, ...]] = (
    "acceptance_rate",
    "total_accepted_tokens",
    "total_draft_proposals",
    "avg_accepted_tokens_per_round",
    "rejection_events",
    "total_rounds",
    "total_runtime_s",
    "total_new_tokens",
    "draft_cache_mode",
    "draft_dequant_time_s_total",
    "draft_refresh_time_s_total",
    "draft_mean_sparsity_ratio",
    "draft_quantization_kv_bits",
    "draft_cache_end_stats",
    "draft_phase_time_s_total",
    "verify_phase_time_s_total",
    "draft_kv_quantization_semantics",
    "draft_runtime_accelerated_quant_attention",
)


def assert_speculative_metrics_jsonable_shape(d: dict[str, Any]) -> None:
    """Strict check for benchmark gate: every key the schema expects must be present."""
    missing = [k for k in SPECULATIVE_METRICS_JSON_KEYS if k not in d]
    if missing:
        raise AssertionError(f"metrics JSON missing keys (schema v{EXPERIMENT_SCHEMA_VERSION}): {missing}")
