"""Phase H — QuantSpec-style evaluation CSV schema (plot/table friendly).

**Definitions**

* **E2E timed** — ``prefill_time_s + decode_phase_time_s`` (CUDA-synchronized intervals where available).
  Use ``tokens_per_sec_e2e_timed`` = ``max_new_tokens / e2e_timed_s``.
* **Decode-only** — ``decode_phase_time_s`` excludes prefill (draft + verify + hierarchical HF resync).
  Use ``tokens_per_sec_decode_phase`` = ``max_new_tokens / decode_phase_time_s``.
* **AR baseline** — ``decode_phase_time_s`` = ``end_to_end_generation_s - prefill_time_s`` (= incremental
  greedy forwards after prefill; first new token comes from prefill logits).
* **Bytes per committed token** — ``logical_verifier_kv_bytes / full_seq_len`` (footprint proxy; not a HW
  memory traffic counter).
* **Speedup** — always relative to a named baseline column (e.g. ``speedup_vs_ar_e2e``); document in sweep YAML.

Stratify plots by ``gamma`` (empty for ``baseline=ar``) — do not aggregate across γ without bucketing.

* **Retries** — ``row_retries`` = YAML ``row_retries`` (number of **extra** attempts after a failure;
  ``max_attempts = 1 + row_retries``). ``attempts_used`` is 1–``max_attempts`` on success; on failure
  equals ``max_attempts``.

* **Phase N (schema v4+)** — ``comparison_mode`` buckets runs for plotting: ``hf_ar`` (SDPA on Llama),
  ``hf_ar_eager`` (eager matmul attention on Llama), ``dense_self_spec``, ``hierarchical_ref``, ``hierarchical_fused``. ``context_length_target_tokens`` is the YAML-driven prefill
  length target (``context_length_tokens_values`` sweep). ``cache_mutation_time_s_total`` mirrors
  ``quant_resync_time_s_total`` (hierarchical resync / cache reconciliation). Traffic proxies are **not**
  hardware counters — see :mod:`benchmarks.roofline_estimates`.

* **Phase O (schema v5)** — ``kernel_tuning_profile`` + ``kernel_tuning_config_json`` reproduce Triton tile/warp
  settings (:mod:`kv_kernels.tuning`). ``runtime_perf_flags_json`` records CUDA-graph / workspace / dispatch flags.
"""

from __future__ import annotations

from typing import Final

PHASE_H_SCHEMA_VERSION: Final[str] = "5"

# Stable column order for CSV (append-only for v2+).
PHASE_H_CSV_FIELDNAMES: list[str] = [
    "phase_h_schema_version",
    "sweep_id",
    "timestamp_utc",
    "status",
    "failure_reason",
    "trial_index",
    "warmup",
    "row_retries",
    "max_attempts",
    "attempts_used",
    # Experiment identity
    "baseline",
    "benchmark_label",
    "gamma",
    "G",
    "cf1_max_tokens",
    "quant_group_size",
    "kv_kernel_backend",
    "batch_size",
    # Environment
    "model_name",
    "gpu_torch_name",
    "device_type",
    "dtype",
    "torch_version",
    "modal_resource_tag",
    # Prompt / regime
    "prompt_id",
    "prompt_len_tokens",
    "context_bucket",
    "max_new_tokens",
    "decode_heavy_ratio",
    # Timings (seconds)
    "prefill_time_s",
    "decode_phase_time_s",
    "e2e_timed_s",
    "total_runtime_wall_s",
    "draft_latency_total_s",
    "verify_latency_total_s",
    "quant_resync_time_s_total",
    # Throughput (tokens/s)
    "tokens_per_sec_e2e_timed",
    "tokens_per_sec_decode_phase",
    "tokens_per_sec_wall",
    # Quality
    "acceptance_rate",
    "total_rounds",
    # Memory / bytes
    "logical_verifier_kv_bytes",
    "logical_quant_store_bytes",
    "logical_bytes_per_output_token",
    "gpu_peak_memory_bytes_after_run",
    "gpu_peak_memory_bytes_before_run",
    # Phase N — Modal roofline / comparison (append-only; see docs string above)
    "comparison_mode",
    "context_length_target_tokens",
    "cache_mutation_time_s_total",
    "estimated_kv_traffic_bytes_per_output_token",
    "estimated_hist_kv_read_proxy_bytes",
    "effective_kv_kernel_backend",
    "fused_path_active",
    "quant_attention_patch_applied",
    "cuda_graphs_enabled",
    "timing_sync",
    "implied_bandwidth_gbps_decode_proxy",
    # Phase O — kernel tuning & runtime flags
    "kernel_tuning_profile",
    "kernel_tuning_config_json",
    "runtime_perf_flags_json",
    # Documentation
    "speedup_definition_note",
    "notes",
]
