"""Phase H — QuantSpec-style benchmark & ablation (long-context, decode-heavy, low batch).

**Usage** (repo root)::

    PYTHONPATH=src python -m benchmarks.phase_h_runner --config configs/phase_h_smoke.yaml

See :mod:`benchmarks.phase_h_schema` for CSV column definitions and throughput semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
import time
from datetime import datetime, timezone
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml

from benchmarks.context_buckets import classify_context_bucket, prompt_token_length
from benchmarks.memory import max_memory_allocated_bytes, reset_peak_memory_stats
from benchmarks.phase_h_schema import PHASE_H_CSV_FIELDNAMES, PHASE_H_SCHEMA_VERSION
from benchmarks.roofline_estimates import (
    estimate_attention_hist_kv_read_bytes_proxy,
    implied_bandwidth_gbps,
    traffic_proxy_bytes_per_output_token_from_row,
)
from benchmarks.timer import cuda_synchronize
from kv_kernels.runtime_options import RuntimePerfFlags
from kv_kernels.tuning import set_active_kernel_tuning, set_kernel_tuning_from_spec
from kv_kernels.triton_runtime import triton_available
from decoding.autoregressive import decode_greedy_autoregressive, model_device
from decoding.speculative_dense import SpeculativeDecoderDense
from decoding.speculative_dense_hierarchical import SpeculativeDecoderDenseHierarchical
from mlsys_kv.datasets.mt_bench import load_mt_bench_subset
from mlsys_kv.infra.device import resolve_device
from mlsys_kv.infra.seed import set_seed
from mlsys_kv.models.hf_loader import load_causal_lm

BASELINE_AR = "ar"
BASELINE_AR_EAGER = "ar_eager"
BASELINE_DENSE_SPEC = "dense_spec"
BASELINE_QUANT_SPEC = "quant_spec"


def _comparison_mode(baseline: str, kv: str | None) -> str:
    """Phase N plot bucket: dense HF AR vs dense self-spec vs hierarchical ref vs fused."""
    if baseline == BASELINE_AR:
        return "hf_ar"
    if baseline == BASELINE_AR_EAGER:
        return "hf_ar_eager"
    if baseline == BASELINE_DENSE_SPEC:
        return "dense_self_spec"
    if baseline == BASELINE_QUANT_SPEC:
        if (kv or "").strip().lower() == "triton":
            return "hierarchical_fused"
        return "hierarchical_ref"
    return baseline


def _should_apply_quant_spec_attention_patch(cfg: dict[str, Any], model_name: str, kv: str | None) -> bool:
    """Llama needs QuantSpec attention patch for Triton hierarchical attention; GPT-2 stays HF core."""
    explicit = cfg.get("apply_quant_spec_attention_patch")
    if explicit is not None:
        return bool(explicit)
    return (kv or "").strip().lower() == "triton" and "llama" in model_name.lower()


def _infer_layers_heads_head_dim(model: torch.nn.Module) -> tuple[int, int, int]:
    c = model.config
    n = int(getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", 0))
    nh = int(getattr(c, "num_attention_heads", None) or getattr(c, "n_head", 0))
    hs = int(getattr(c, "hidden_size", None) or getattr(c, "n_embd", 0))
    if nh <= 0 or hs <= 0 or hs % nh != 0:
        return n, max(nh, 1), max(hs // max(nh, 1), 1)
    return n, nh, hs // nh


def _approx_verifier_kv_fp16_bytes(model: torch.nn.Module, seq_len: int) -> int:
    """Rough FP16 K+V size for all layers (batch=1), for reporting when only HF-style KV exists."""
    c = model.config
    n = int(getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", 0))
    nh = int(getattr(c, "num_attention_heads", None) or getattr(c, "n_head", 0))
    hs = int(getattr(c, "hidden_size", None) or getattr(c, "n_embd", 0))
    if nh <= 0 or hs <= 0:
        return 0
    hd = hs // nh
    # Two tensors (K,V), FP16 = 2 bytes per element; shape [1, nh, S, hd]
    per_layer = 2 * (2 * nh * max(seq_len, 0) * hd)
    return n * per_layer


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_min_prompt_tokens(tok, text: str, min_tokens: int) -> str:
    if min_tokens <= 0:
        return text
    cur = text
    while prompt_token_length(tok, cur) < min_tokens:
        cur = cur + "\n\n" + text
    return cur


def _model_max_context_tokens(model: torch.nn.Module) -> int:
    c = model.config
    n = getattr(c, "max_position_embeddings", None) or getattr(c, "n_positions", None)
    if n is None:
        return 2048
    return int(n)


def _truncate_prompt_to_fit(tok, text: str, max_prompt_tokens: int) -> str:
    """Keep the tail of the prompt so token count <= max_prompt_tokens (matches HF prefill style).

    Decode/re-encode can change length; iterate until stable so we never exceed the budget
    (avoids 2049 vs 2048 style tokenizer warnings on tight contexts).
    """
    if max_prompt_tokens <= 0:
        return text
    for _ in range(12):
        enc = tok(text, return_tensors="pt", add_special_tokens=True)
        n = int(enc["input_ids"].shape[1])
        if n <= max_prompt_tokens:
            return text
        tail = enc["input_ids"][:, -max_prompt_tokens:]
        text = tok.decode(tail[0], skip_special_tokens=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    if int(enc["input_ids"].shape[1]) > max_prompt_tokens:
        ids = enc["input_ids"][0, -max_prompt_tokens:].tolist()
        text = tok.decode(ids, skip_special_tokens=True)
    return text


def _expand_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    baselines = [str(b) for b in (cfg.get("baselines") or [BASELINE_AR, BASELINE_DENSE_SPEC, BASELINE_QUANT_SPEC])]
    gammas = [int(x) for x in (cfg.get("gamma_values") or [4])]
    Gs = [int(x) for x in (cfg.get("G_values") or [64])]
    qgs_list = cfg.get("quant_group_size_values", [None])
    kvs = [str(x) for x in (cfg.get("kv_kernel_backend_values") or ["reference"])]

    rows: list[dict[str, Any]] = []
    for b in baselines:
        if b in (BASELINE_AR, BASELINE_AR_EAGER):
            rows.append(
                {
                    "baseline": b,
                    "gamma": None,
                    "G": None,
                    "quant_group_size": None,
                    "kv_kernel_backend": None,
                }
            )
        elif b == BASELINE_DENSE_SPEC:
            for g in gammas:
                rows.append(
                    {
                        "baseline": b,
                        "gamma": int(g),
                        "G": None,
                        "quant_group_size": None,
                        "kv_kernel_backend": None,
                    }
                )
        elif b == BASELINE_QUANT_SPEC:
            for g in gammas:
                for G in Gs:
                    for qgs in qgs_list:
                        for kv in kvs:
                            rows.append(
                                {
                                    "baseline": b,
                                    "gamma": int(g),
                                    "G": int(G),
                                    "quant_group_size": qgs,
                                    "kv_kernel_backend": kv,
                                }
                            )
        else:
            raise ValueError(f"Unknown baseline {b!r}; use ar | ar_eager | dense_spec | quant_spec")
    return rows


def _parse_csv_int(s: str | None) -> int | None:
    t = (s or "").strip()
    if t == "" or t.lower() == "none":
        return None
    return int(t)


def _row_key(
    prompt_id: str,
    cell: dict[str, Any],
    max_new_tokens: int,
    trial_index: int,
    context_target_tokens: int,
) -> tuple[Any, ...]:
    return (
        prompt_id,
        cell["baseline"],
        cell.get("gamma"),
        cell.get("G"),
        cell.get("quant_group_size"),
        cell.get("kv_kernel_backend"),
        int(max_new_tokens),
        int(trial_index),
        int(context_target_tokens),
    )


def _load_done_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.is_file():
        return set()
    done: set[tuple[Any, ...]] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            try:
                mnt = int(row.get("max_new_tokens", 0))
            except ValueError:
                mnt = 0
            ctx_t = row.get("context_length_target_tokens")
            try:
                ctx_int = int(ctx_t) if ctx_t not in (None, "") else -1
            except ValueError:
                ctx_int = -1
            key = (
                row.get("prompt_id"),
                row.get("baseline"),
                _parse_csv_int(row.get("gamma")),
                _parse_csv_int(row.get("G")),
                row.get("quant_group_size") or None,
                row.get("kv_kernel_backend") or None,
                mnt,
                int(row.get("trial_index", 0)),
                ctx_int,
            )
            done.add(key)
    return done


VolumeCommitFn = Callable[[], None]


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PHASE_H_CSV_FIELDNAMES, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in PHASE_H_CSV_FIELDNAMES})
        f.flush()


def run_phase_h_benchmark(
    config_path: str | Path,
    *,
    volume_commit_fn: VolumeCommitFn | None = None,
) -> int:
    p = Path(config_path)
    cfg: dict[str, Any] = dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})

    sweep_id = str(cfg.get("sweep_id", p.stem))
    model_name = str(cfg["model_name"])
    device_s = str(cfg.get("device", "auto"))
    dtype_s = str(cfg.get("dtype", "float16"))
    seed = int(cfg.get("seed", 42))
    warmup_trials = int(cfg.get("warmup_trials", 0))
    num_trials = int(cfg.get("num_trials", 1))
    verify_match = bool(cfg.get("verify_match", False))
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    min_prompt_tokens = int(cfg.get("min_prompt_tokens", 128))
    short_max = int(cfg.get("short_token_max", 64))
    medium_max = int(cfg.get("medium_token_max", 256))
    batch_size = int(cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("Phase H runner currently supports batch_size=1 only")

    mt_path = Path(cfg.get("mt_bench_path", "data/mt_bench_subset.json"))
    max_prompts = cfg.get("max_prompts")
    speedup_note = str(
        cfg.get(
            "speedup_definition_note",
            "tokens_per_sec_e2e_timed = max_new_tokens / (prefill_time_s + decode_phase_time_s); "
            "decode_phase excludes prefill (see phase_h_schema).",
        )
    )
    notes = str(cfg.get("notes", ""))
    out_csv = Path(cfg.get("output_csv", f"outputs/phase_h/{sweep_id}.csv"))
    raw_jsonl = Path(cfg.get("raw_jsonl", f"outputs/phase_h/{sweep_id}.jsonl"))
    resume = bool(cfg.get("resume", True))
    modal_tag = str(cfg.get("modal_resource_tag") or os.environ.get("MODAL_RESOURCE_TAG", ""))
    row_retries = int(cfg.get("row_retries", 1))
    max_attempts = 1 + max(0, row_retries)

    context_length_tokens_values = cfg.get("context_length_tokens_values")
    if context_length_tokens_values is not None:
        context_targets = [int(x) for x in context_length_tokens_values]
    else:
        context_targets = [min_prompt_tokens]
    timing_sync_policy = str(cfg.get("timing_sync", "cuda_synchronize_end"))
    cuda_graphs_flag = "true" if bool(cfg.get("cuda_graphs_enabled", False)) else "false"
    validate_triton_flag = bool(cfg.get("validate_triton_kernels_at_start", False))

    set_seed(seed)
    device = resolve_device(device_s)
    grid = _expand_grid(cfg)
    done_keys = _load_done_keys(out_csv) if resume else set()

    print(f"[phase_h] loading model {model_name} on {device} …", flush=True)
    loaded = load_causal_lm(model_name, device=device, dtype=dtype_s)
    tok = loaded.tokenizer
    model = loaded.model
    model.eval()
    m_dev = model_device(model)

    prompts_all = load_mt_bench_subset(mt_path)
    if max_prompts is not None:
        prompts_all = prompts_all[: int(max_prompts)]

    torch_version = torch.__version__
    gpu_torch_name = ""
    if m_dev.type == "cuda" and torch.cuda.is_available():
        gpu_torch_name = torch.cuda.get_device_name(m_dev)

    tune_prof = cfg.get("kernel_tuning_profile")
    rp_yaml = cfg.get("runtime_perf") or {}
    rp_flags = RuntimePerfFlags(
        cuda_graphs_enabled=bool(rp_yaml.get("cuda_graphs_enabled", cfg.get("cuda_graphs_enabled", False))),
        static_workspace_enabled=bool(rp_yaml.get("static_workspace_enabled", False)),
        reduce_python_dispatch=bool(rp_yaml.get("reduce_python_dispatch", False)),
    )
    runtime_perf_json = json.dumps(asdict(rp_flags), sort_keys=True)

    applied_tune = set_kernel_tuning_from_spec(
        str(tune_prof).strip() if tune_prof else None,
        gpu_torch_name or None,
    )
    try:
        _run_inner_loop(
            context_targets,
            prompts_all,
            grid,
            tok,
            model,
            model_name,
            m_dev,
            gpu_torch_name,
            applied_tune,
            runtime_perf_json,
            short_max,
            medium_max,
            max_new_tokens,
            speedup_note,
            notes,
            sweep_id,
            seed,
            warmup_trials,
            num_trials,
            verify_match,
            batch_size,
            modal_tag,
            row_retries,
            max_attempts,
            torch_version,
            dtype_s,
            done_keys,
            resume,
            out_csv,
            raw_jsonl,
            volume_commit_fn,
            cfg,
            timing_sync_policy,
            cuda_graphs_flag,
            validate_triton_flag,
        )
    finally:
        set_active_kernel_tuning(None)

    print(f"[phase_h] wrote {out_csv}", flush=True)
    return 0


def _run_inner_loop(
    context_targets: list[int],
    prompts_all: list[Any],
    grid: list[dict[str, Any]],
    tok: Any,
    model: torch.nn.Module,
    model_name: str,
    m_dev: torch.device,
    gpu_torch_name: str,
    applied_tune: Any,
    runtime_perf_json: str,
    short_max: int,
    medium_max: int,
    max_new_tokens: int,
    speedup_note: str,
    notes: str,
    sweep_id: str,
    seed: int,
    warmup_trials: int,
    num_trials: int,
    verify_match: bool,
    batch_size: int,
    modal_tag: str,
    row_retries: int,
    max_attempts: int,
    torch_version: str,
    dtype_s: str,
    done_keys: set[tuple[Any, ...]],
    resume: bool,
    out_csv: Path,
    raw_jsonl: Path,
    volume_commit_fn: VolumeCommitFn | None,
    cfg: dict[str, Any],
    timing_sync_policy: str,
    cuda_graphs_flag: str,
    validate_triton_flag: bool,
) -> None:
    max_ctx = _model_max_context_tokens(model)
    # One token of slack: decode/encode round-trip and prefill/generation boundary vs ``max_position_embeddings``.
    max_prompt_for_decode = max(1, max_ctx - int(max_new_tokens) - 1)
    for ctx_target in context_targets:
        for prompt in prompts_all:
            text = _ensure_min_prompt_tokens(tok, prompt.text, ctx_target)
            text = _truncate_prompt_to_fit(tok, text, max_prompt_for_decode)
            prompt_len = prompt_token_length(tok, text)
            bucket = classify_context_bucket(prompt_len, short_max=short_max, medium_max=medium_max)
            decode_heavy_ratio = float(max_new_tokens) / float(max_new_tokens + prompt_len)

            for cell in grid:
                for trial in range(warmup_trials + num_trials):
                    is_warmup = trial < warmup_trials
                    trial_index = trial - warmup_trials
                    key = _row_key(prompt.id, cell, max_new_tokens, trial_index, ctx_target)
                    if not is_warmup and resume and key in done_keys:
                        print(f"[phase_h] skip (done) {key}", flush=True)
                        continue

                    gamma = cell.get("gamma")
                    G = cell.get("G")
                    qgs = cell.get("quant_group_size")
                    kv = cell.get("kv_kernel_backend")
                    baseline = cell["baseline"]

                    label = baseline
                    if gamma is not None:
                        label = f"{baseline}_g{gamma}"
                    if baseline == BASELINE_QUANT_SPEC:
                        qgs_lbl = "default" if qgs is None else str(qgs)
                        label = f"{label}_G{G}_qgs{qgs_lbl}_kv{kv}"

                    row_base: dict[str, Any] = {
                        "phase_h_schema_version": PHASE_H_SCHEMA_VERSION,
                        "sweep_id": sweep_id,
                        "timestamp_utc": _utc_ts(),
                        "trial_index": trial_index if not is_warmup else -1,
                        "warmup": is_warmup,
                        "row_retries": row_retries,
                        "max_attempts": max_attempts,
                        "attempts_used": "",
                        "baseline": baseline,
                        "benchmark_label": label,
                        "gamma": gamma if gamma is not None else "",
                        "G": G if G is not None else "",
                        "cf1_max_tokens": (2 * int(G)) if G is not None else "",
                        "quant_group_size": qgs if qgs is not None else "",
                        "kv_kernel_backend": kv if kv is not None else "",
                        "batch_size": batch_size,
                        "model_name": model_name,
                        "gpu_torch_name": gpu_torch_name,
                        "device_type": m_dev.type,
                        "dtype": dtype_s,
                        "torch_version": torch_version,
                        "modal_resource_tag": modal_tag,
                        "prompt_id": prompt.id,
                        "prompt_len_tokens": prompt_len,
                        "context_bucket": bucket.value,
                        "max_new_tokens": max_new_tokens,
                        "decode_heavy_ratio": f"{decode_heavy_ratio:.4f}",
                        "speedup_definition_note": speedup_note,
                        "notes": notes,
                        "kernel_tuning_profile": applied_tune.profile_id,
                        "kernel_tuning_config_json": applied_tune.to_json(),
                        "runtime_perf_flags_json": runtime_perf_json,
                    }

                    for attempt in range(1, max_attempts + 1):
                        try:
                            reset_peak_memory_stats(m_dev)
                            peak_before = max_memory_allocated_bytes(m_dev)
                            t_wall0 = time.perf_counter()

                            if baseline in (BASELINE_AR, BASELINE_AR_EAGER):
                                prev_attn = getattr(model.config, "_attn_implementation", None)
                                mt = str(getattr(model.config, "model_type", "") or "").lower()
                                llama_like = "llama" in mt
                                try:
                                    if llama_like and hasattr(model.config, "_attn_implementation"):
                                        if baseline == BASELINE_AR_EAGER:
                                            model.config._attn_implementation = "eager"
                                        else:
                                            model.config._attn_implementation = "sdpa"
                                    res = decode_greedy_autoregressive(
                                        model,
                                        tok,
                                        text,
                                        max_new_tokens=max_new_tokens,
                                        warmup=is_warmup,
                                        trial_index=trial_index,
                                    )
                                finally:
                                    if prev_attn is not None and hasattr(model.config, "_attn_implementation"):
                                        model.config._attn_implementation = prev_attn
                                m = res.metrics
                                prefill_s = float(m.prefill_time_s)
                                decode_phase_s = float(m.end_to_end_generation_s - m.prefill_time_s)
                                e2e_timed = float(m.end_to_end_generation_s)
                                draft_s = 0.0
                                verify_s = 0.0
                                resync_s = 0.0
                                acceptance = ""
                                total_rounds = ""
                                log_ver = int(m.logical_kv_cache_bytes)
                                log_quant = ""
                                ver_kv = log_ver
                            elif baseline == BASELINE_DENSE_SPEC:
                                assert gamma is not None
                                dec = SpeculativeDecoderDense(
                                    model,
                                    tok,
                                    gamma=int(gamma),
                                    verify_match_autoregressive=verify_match and not is_warmup,
                                    serving_mode=True,
                                    legacy_double_clone_verifier=False,
                                )
                                res = dec.decode(text, max_new_tokens=max_new_tokens, benchmark_profile=True)
                                m = res.metrics
                                prefill_s = float(m.prefill_time_s)
                                decode_phase_s = float(m.decode_phase_time_s)
                                e2e_timed = prefill_s + decode_phase_s
                                draft_s = float(m.draft_phase_time_s_total)
                                verify_s = float(m.verify_phase_time_s_total)
                                resync_s = float(m.quant_resync_time_s_total)
                                acceptance = float(m.acceptance_rate)
                                total_rounds = int(m.total_rounds)
                                log_ver = int(res.verifier_kv.memory_bytes())
                                log_quant = ""
                                ver_kv = log_ver
                            elif baseline == BASELINE_QUANT_SPEC:
                                assert gamma is not None and G is not None and kv is not None
                                qgs_int = int(qgs) if qgs is not None else None
                                apply_patch = _should_apply_quant_spec_attention_patch(cfg, model_name, kv)
                                dec = SpeculativeDecoderDenseHierarchical(
                                    model,
                                    tok,
                                    gamma=int(gamma),
                                    G=int(G),
                                    quant_group_size=qgs_int,
                                    verify_match_autoregressive=verify_match and not is_warmup,
                                    serving_mode=True,
                                    kv_kernel_backend=kv,
                                    validate_triton_kernels_at_start=validate_triton_flag
                                    and (kv or "").strip().lower() == "triton",
                                    apply_quant_spec_attention_patch=apply_patch,
                                )
                                res = dec.decode(text, max_new_tokens=max_new_tokens, benchmark_profile=True)
                                m = res.metrics
                                prefill_s = float(m.prefill_time_s)
                                decode_phase_s = float(m.decode_phase_time_s)
                                e2e_timed = prefill_s + decode_phase_s
                                draft_s = float(m.draft_phase_time_s_total)
                                verify_s = float(m.verify_phase_time_s_total)
                                resync_s = float(m.quant_resync_time_s_total)
                                acceptance = float(m.acceptance_rate)
                                total_rounds = int(m.total_rounds)
                                log_quant = int(res.mgr.store.memory_bytes_estimate())
                                full_len_est = int(res.full_token_ids.shape[-1])
                                log_ver = _approx_verifier_kv_fp16_bytes(model, full_len_est)
                                ver_kv = log_ver
                            else:
                                raise RuntimeError(f"unreachable baseline {baseline}")

                            cuda_synchronize(m_dev)
                            peak_after = max_memory_allocated_bytes(m_dev)
                            wall_s = time.perf_counter() - t_wall0

                            full_len = int(prompt_len + max_new_tokens)
                            bytes_per_tok: float | str = ""
                            if baseline == BASELINE_QUANT_SPEC and log_quant != "":
                                bytes_per_tok = float(log_quant) / float(full_len)
                            elif isinstance(ver_kv, int) and full_len > 0:
                                bytes_per_tok = float(ver_kv) / float(full_len)

                            tps_e2e: float | str = float(max_new_tokens) / e2e_timed if e2e_timed > 0 else ""
                            tps_dec: float | str = (
                                float(max_new_tokens) / decode_phase_s if decode_phase_s > 0 else ""
                            )
                            tps_wall: float | str = float(max_new_tokens) / wall_s if wall_s > 0 else ""

                            triton_ok = (
                                m_dev.type == "cuda"
                                and torch.cuda.is_available()
                                and triton_available()
                            )
                            fused_active = (
                                baseline == BASELINE_QUANT_SPEC
                                and (kv or "").strip().lower() == "triton"
                                and triton_ok
                            )
                            apply_patch_effective = _should_apply_quant_spec_attention_patch(cfg, model_name, kv)
                            n_l, n_h, n_hd = _infer_layers_heads_head_dim(model)
                            est_hist_read = estimate_attention_hist_kv_read_bytes_proxy(
                                num_layers=n_l,
                                num_heads=n_h,
                                head_dim=n_hd,
                                hist_seq_len=full_len,
                                kv_kernel_backend=str(kv or "reference"),
                            )
                            traffic_est = traffic_proxy_bytes_per_output_token_from_row(
                                baseline=baseline,
                                logical_quant_store_bytes=log_quant if log_quant != "" else None,
                                logical_verifier_kv_bytes=ver_kv if isinstance(ver_kv, int) else None,
                                full_seq_len=full_len,
                            )
                            tps_dec_f = float(tps_dec) if isinstance(tps_dec, float) else None
                            gbps_est = implied_bandwidth_gbps(tps_dec_f, traffic_est)

                            out_row = {
                                **row_base,
                                "status": "ok",
                                "failure_reason": "",
                                "attempts_used": attempt,
                                "prefill_time_s": f"{prefill_s:.6f}",
                                "decode_phase_time_s": f"{decode_phase_s:.6f}",
                                "e2e_timed_s": f"{e2e_timed:.6f}",
                                "total_runtime_wall_s": f"{wall_s:.6f}",
                                "draft_latency_total_s": f"{draft_s:.6f}",
                                "verify_latency_total_s": f"{verify_s:.6f}",
                                "quant_resync_time_s_total": f"{resync_s:.6f}",
                                "tokens_per_sec_e2e_timed": f"{tps_e2e}" if tps_e2e != "" else "",
                                "tokens_per_sec_decode_phase": f"{tps_dec}" if tps_dec != "" else "",
                                "tokens_per_sec_wall": f"{tps_wall}" if tps_wall != "" else "",
                                "acceptance_rate": f"{acceptance}" if acceptance != "" else "",
                                "total_rounds": total_rounds if total_rounds != "" else "",
                                "logical_verifier_kv_bytes": ver_kv if isinstance(ver_kv, int) else "",
                                "logical_quant_store_bytes": log_quant if log_quant != "" else "",
                                "logical_bytes_per_output_token": f"{bytes_per_tok}"
                                if bytes_per_tok != ""
                                else "",
                                "gpu_peak_memory_bytes_after_run": peak_after,
                                "gpu_peak_memory_bytes_before_run": peak_before,
                                "comparison_mode": _comparison_mode(baseline, kv if isinstance(kv, str) else None),
                                "context_length_target_tokens": ctx_target,
                                "cache_mutation_time_s_total": f"{resync_s:.6f}",
                                "estimated_kv_traffic_bytes_per_output_token": f"{traffic_est}"
                                if traffic_est is not None
                                else "",
                                "estimated_hist_kv_read_proxy_bytes": f"{est_hist_read:.0f}",
                                "effective_kv_kernel_backend": (kv or "")
                                if baseline == BASELINE_QUANT_SPEC
                                else "",
                                "fused_path_active": "true" if fused_active else "false",
                                "quant_attention_patch_applied": "true"
                                if (baseline == BASELINE_QUANT_SPEC and apply_patch_effective)
                                else "false",
                                "cuda_graphs_enabled": cuda_graphs_flag,
                                "timing_sync": timing_sync_policy,
                                "implied_bandwidth_gbps_decode_proxy": f"{gbps_est}"
                                if gbps_est is not None
                                else "",
                            }

                            if not is_warmup:
                                # One-line summary for logs: separates e2e vs decode-only tok/s and confirms Triton fused path.
                                print(
                                    "[phase_h] metrics "
                                    f"prompt_id={prompt.id} baseline={baseline} "
                                    f"comparison_mode={out_row.get('comparison_mode', '')} "
                                    f"context_target={ctx_target} "
                                    f"fused_path_active={out_row.get('fused_path_active', '')} "
                                    f"effective_kv_backend={out_row.get('effective_kv_kernel_backend', '')} "
                                    f"tps_e2e={out_row.get('tokens_per_sec_e2e_timed', '')} "
                                    f"tps_decode_only={out_row.get('tokens_per_sec_decode_phase', '')}",
                                    flush=True,
                                )
                                _append_csv(out_csv, out_row)
                                done_keys.add(
                                    _row_key(prompt.id, cell, max_new_tokens, trial_index, ctx_target)
                                )
                            raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
                            with raw_jsonl.open("a", encoding="utf-8") as jf:
                                jf.write(json.dumps(out_row, default=str, sort_keys=True) + "\n")
                                jf.flush()
                            if not is_warmup and volume_commit_fn is not None:
                                volume_commit_fn()
                            break

                        except Exception as exc:
                            if attempt < max_attempts:
                                print(
                                    f"[phase_h] RETRY {attempt}/{max_attempts} {prompt.id} {baseline}: {exc!r}",
                                    flush=True,
                                )
                                continue
                            empty_phase_n = {
                                "comparison_mode": _comparison_mode(baseline, kv if isinstance(kv, str) else None),
                                "context_length_target_tokens": str(ctx_target),
                                "cache_mutation_time_s_total": "",
                                "estimated_kv_traffic_bytes_per_output_token": "",
                                "estimated_hist_kv_read_proxy_bytes": "",
                                "effective_kv_kernel_backend": "",
                                "fused_path_active": "",
                                "quant_attention_patch_applied": "",
                                "cuda_graphs_enabled": cuda_graphs_flag,
                                "timing_sync": timing_sync_policy,
                                "implied_bandwidth_gbps_decode_proxy": "",
                            }
                            err_row = {
                                **row_base,
                                **empty_phase_n,
                                "status": "error",
                                "failure_reason": repr(exc),
                                "attempts_used": max_attempts,
                                "prefill_time_s": "",
                                "decode_phase_time_s": "",
                                "e2e_timed_s": "",
                                "total_runtime_wall_s": "",
                                "draft_latency_total_s": "",
                                "verify_latency_total_s": "",
                                "quant_resync_time_s_total": "",
                                "tokens_per_sec_e2e_timed": "",
                                "tokens_per_sec_decode_phase": "",
                                "tokens_per_sec_wall": "",
                                "acceptance_rate": "",
                                "total_rounds": "",
                                "logical_verifier_kv_bytes": "",
                                "logical_quant_store_bytes": "",
                                "logical_bytes_per_output_token": "",
                                "gpu_peak_memory_bytes_after_run": "",
                                "gpu_peak_memory_bytes_before_run": "",
                            }
                            if not is_warmup:
                                _append_csv(out_csv, err_row)
                                if volume_commit_fn is not None:
                                    volume_commit_fn()
                            print(f"[phase_h] ERROR {prompt.id} {baseline}: {exc}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase H QuantSpec benchmark runner")
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    args = ap.parse_args()
    # Allow `python benchmarks/phase_h_runner.py` from repo root without PYTHONPATH
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    raise SystemExit(run_phase_h_benchmark(args.config, volume_commit_fn=None))


if __name__ == "__main__":
    main()
