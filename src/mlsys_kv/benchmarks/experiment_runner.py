"""Phase 15 benchmark sweep: MT-Bench subset, staged grids, CSV/JSONL, resume, Modal-ready."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import statistics
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from mlsys_kv.benchmarks.context_buckets import classify_context_bucket, prompt_token_length
from mlsys_kv.benchmarks.experiment_schema import (
    BENCHMARK_CSV_FIELDNAMES,
    EXPERIMENT_SCHEMA_VERSION,
    SPARSE_INTEGRATION_VERSION,
    SWEEP_INPUT_MODES,
    SWEEP_MODE_AUTOREGRESSIVE,
    SWEEP_MODE_QUANT_ONLY,
    SWEEP_MODE_SPARSE_ONLY,
    SWEEP_MODE_SPARSE_QUANT,
    SWEEP_MODE_SPECULATIVE_FP16,
    benchmark_label_for_canonical_mode,
    canonical_sweep_mode,
    quantization_type_for_row,
)
from mlsys_kv.benchmarks.memory import max_memory_allocated_bytes, reset_peak_memory_stats
from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.datasets.mt_bench import MBPrompt, load_mt_bench_subset
from mlsys_kv.decoding.autoregressive import decode_greedy_autoregressive, model_device
from mlsys_kv.decoding.speculative import SpeculativeDecoder
from mlsys_kv.infra.device import resolve_device
from mlsys_kv.infra.seed import set_seed
from mlsys_kv.models.hf_loader import load_causal_lm

VolumeCommitFn = Callable[[], None]


def model_weights_bytes(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) * int(p.element_size()) for p in model.parameters())


def build_sparse_config_for_prompt(
    prompt_len_tokens: int,
    *,
    sparsity_budget: float,
    recent_window_fraction: float,
    refresh_interval: int,
    scoring: str,
) -> SparseRetentionConfig:
    """Map ``sparsity_budget`` (fraction of non-recent pool for heavy hitters, or ≥1 for full pool)."""

    rw = max(1, int(prompt_len_tokens * recent_window_fraction))
    rw = min(rw, prompt_len_tokens)
    pool = max(0, prompt_len_tokens - rw)
    if pool <= 0:
        hb = 0
    elif sparsity_budget >= 1.0:
        hb = pool
    else:
        hb = max(1, int(pool * float(sparsity_budget)))
    ss = scoring if scoring in ("attention", "key_norm") else "key_norm"
    return SparseRetentionConfig(
        recent_window=rw,
        heavy_hitter_budget=int(hb),
        refresh_interval=int(refresh_interval),
        scoring=ss,  # type: ignore[arg-type]
    )


def resolve_speculative_mode(
    logical_mode: str,
    quant_bits: int,
) -> tuple[DraftCacheMode, int, str, str]:
    """Return ``(draft_mode, quant_bits_effective, draft_cache_mode_resolved, notes)`` for speculative paths."""
    if logical_mode == "speculative_fp16":
        return (
            DraftCacheMode.FP16,
            16,
            "fp16",
            "matrix_quant_bits_preserved_for_log_not_used_by_fp16_draft",
        )
    if logical_mode == "quant_only":
        if quant_bits == 16:
            return DraftCacheMode.FP16, 16, "fp16", "quant_matrix_16bit_maps_to_fp16_draft"
        if quant_bits == 4:
            return (
                DraftCacheMode.QUANT_ONLY,
                4,
                "quant_only",
                "symmetric_int4_per_group_packed_kv",
            )
        return DraftCacheMode.QUANT_ONLY, 8, "quant_only", ""
    if logical_mode == "sparse_only":
        return DraftCacheMode.SPARSE_ONLY, 16, "sparse_only", "sparse_fp16_retained_kv"
    if logical_mode == "sparse_quant":
        if quant_bits == 16:
            return DraftCacheMode.SPARSE_ONLY, 16, "sparse_only", "joint_matrix_16bit_maps_to_sparse_fp16"
        if quant_bits == 4:
            return (
                DraftCacheMode.SPARSE_QUANT,
                4,
                "sparse_quant",
                "joint_sparse_int4_retained_kv",
            )
        return DraftCacheMode.SPARSE_QUANT, 8, "sparse_quant", ""
    raise ValueError(f"Unknown speculative logical mode {logical_mode!r}")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_sweep_grid(
    modes_canon: list[str],
    k_values: list[int],
    sparsity_budgets: list[float],
    quant_bits_list: list[int],
    *,
    strict_labeled_grid: bool,
) -> list[tuple[str, int, float, int]]:
    """Yield ``(canonical_mode, spec_k, sparsity_budget, quant_bits_requested)`` cells.

    With ``strict_labeled_grid=True`` (Phase 15 default), dimensions that a mode ignores are fixed
    to sentinels so every row is labeled consistently and we do not duplicate identical runs under
    different hyperparameter keys (e.g. AR × every ``quant_bits``).
    """

    ks = list(k_values) if k_values else [1]
    sbs = list(sparsity_budgets) if sparsity_budgets else [0.0]
    qbs = list(quant_bits_list) if quant_bits_list else [8]

    if not strict_labeled_grid:
        return list(product(modes_canon, ks, sbs, qbs))

    out: list[tuple[str, int, float, int]] = []
    for mode in modes_canon:
        if mode == SWEEP_MODE_AUTOREGRESSIVE:
            out.append((mode, 0, 0.0, -1))
        elif mode == SWEEP_MODE_SPECULATIVE_FP16:
            for spec_k in ks:
                out.append((mode, int(spec_k), 0.0, 16))
        elif mode == SWEEP_MODE_QUANT_ONLY:
            for spec_k, qb in product(ks, qbs):
                out.append((mode, int(spec_k), 0.0, int(qb)))
        elif mode == SWEEP_MODE_SPARSE_ONLY:
            for spec_k, sb in product(ks, sbs):
                out.append((mode, int(spec_k), float(sb), 16))
        elif mode == SWEEP_MODE_SPARSE_QUANT:
            for spec_k, sb, qb in product(ks, sbs, qbs):
                out.append((mode, int(spec_k), float(sb), int(qb)))
        else:
            raise ValueError(f"Unknown canonical mode {mode!r}")
    return out


def append_benchmark_csv_row(
    path: Path,
    row: dict[str, Any],
    *,
    fieldnames: list[str] | None = None,
    volume_commit_fn: VolumeCommitFn | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = fieldnames or BENCHMARK_CSV_FIELDNAMES
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})
        f.flush()
    if volume_commit_fn is not None:
        volume_commit_fn()


def load_completed_keys(path: Path, *, retry_failures: bool) -> set[tuple[Any, ...]]:
    """Keys to skip: all ``ok`` rows; ``error``/``skipped`` rows skip only if ``not retry_failures``."""
    if not path.is_file():
        return set()
    keys: set[tuple[Any, ...]] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = row.get("status", "")
            try:
                mode_c = canonical_sweep_mode(str(row.get("mode", "")))
            except ValueError:
                mode_c = str(row.get("mode", ""))
            key = (
                row.get("prompt_id"),
                mode_c,
                row.get("spec_k"),
                row.get("sparsity_budget"),
                row.get("quant_bits_requested"),
                row.get("context_bucket"),
                row.get("trial_index"),
            )
            if st == "ok":
                keys.add(key)
            elif st in ("error", "skipped") and not retry_failures:
                keys.add(key)
    return keys


def append_raw_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
        f.flush()


def run_benchmark_sweep(
    sweep_yaml: str | Path,
    *,
    volume_commit_fn: VolumeCommitFn | None = None,
    modal_resource_tag: str = "",
) -> int:
    """Append one CSV row per measured prompt×trial; flush/commit after each row."""

    p = Path(sweep_yaml)
    cfg: dict[str, Any] = dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})

    sweep_id = str(cfg.get("sweep_id", p.stem))
    model_name = str(cfg["model_name"])
    max_new_tokens = int(cfg.get("max_new_tokens", 32))
    device_s = str(cfg.get("device", "auto"))
    dtype_s = str(cfg.get("dtype", cfg.get("torch_dtype", "float16")))
    seed = int(cfg.get("seed", 42))
    warmup_runs = int(cfg.get("warmup_runs", 1))
    num_trials = int(cfg.get("num_trials", 1))
    verify_match = bool(cfg.get("verify_match", False))
    mt_path = Path(cfg.get("mt_bench_path", "data/mt_bench_subset.json"))
    max_prompts = cfg.get("max_prompts")
    short_max = int(cfg.get("short_token_max", 64))
    medium_max = int(cfg.get("medium_token_max", 256))
    buckets_filter = {str(b).lower() for b in (cfg.get("context_buckets") or ["short", "medium", "long"])}
    modes_raw = [str(m) for m in (cfg.get("modes") or ["autoregressive", "speculative_fp16"])]
    unknown_modes = set(modes_raw) - SWEEP_INPUT_MODES
    if unknown_modes:
        raise ValueError(
            f"Unknown benchmark sweep mode(s) {sorted(unknown_modes)}; "
            f"must be subset of {sorted(SWEEP_INPUT_MODES)} (see experiment_schema / BENCHMARK_PHASE15.md)"
        )
    modes_canon = [canonical_sweep_mode(m) for m in modes_raw]
    print(
        f"[sweep] experiment_schema_version={EXPERIMENT_SCHEMA_VERSION} "
        "pre-flight gate: pytest -m benchmark_gate (scripts/benchmark_presweep_gate.py; docs/BENCHMARK_READINESS.md)",
        flush=True,
    )
    k_values = [int(x) for x in (cfg.get("k_values") or [1, 3, 5, 7])]
    sparsity_budgets = [float(x) for x in (cfg.get("sparsity_budgets") or [0.1, 0.2, 0.4, 1.0])]
    quant_bits_list = [int(x) for x in (cfg.get("quant_bits") or [4, 8, 16])]
    recent_frac = float(cfg.get("recent_window_fraction", 0.15))
    refresh_iv = int(cfg.get("sparse_refresh_interval", 2))
    sparse_scoring = str(cfg.get("sparse_scoring", "key_norm"))
    out_csv = Path(cfg.get("output_csv", "outputs/benchmarks/sweep.csv"))
    raw_jsonl = Path(cfg.get("raw_jsonl_path", f"outputs/benchmarks/raw/{sweep_id}.jsonl"))
    processed_json = Path(cfg.get("processed_json", f"outputs/benchmarks/processed/{sweep_id}_rollup.json"))
    resume = bool(cfg.get("resume", True))
    retry_failures = bool(cfg.get("retry_failures", False))
    strict_labeled_grid = bool(cfg.get("strict_labeled_grid", True))

    if out_csv.is_file():
        hdr = out_csv.open(encoding="utf-8").readline()
        if hdr and "benchmark_label" not in hdr:
            raise ValueError(
                f"CSV {out_csv} predates experiment_schema_version 2 (missing benchmark_label). "
                "Use a new output_csv path for Phase 15 sweeps."
            )

    dump_bundle = Path(cfg.get("config_bundle_json", f"outputs/benchmarks/{sweep_id}_config.json"))
    dump_sweep_config_bundle(p, dump_bundle)

    prompts_all = load_mt_bench_subset(mt_path)
    if max_prompts is not None:
        prompts_all = prompts_all[: int(max_prompts)]

    set_seed(seed)
    device = resolve_device(device_s)
    print(
        f"[sweep] experiment_schema_version={EXPERIMENT_SCHEMA_VERSION} "
        f"modes={modes_canon} strict_labeled_grid={strict_labeled_grid}",
        flush=True,
    )
    print(f"[sweep] loading model {model_name} on {device} …", flush=True)
    loaded = load_causal_lm(model_name, device=device, dtype=dtype_s)
    tok = loaded.tokenizer
    if getattr(tok, "pad_token_id", None) is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = loaded.model
    model.eval()
    m_dev = model_device(model)
    gpu_torch = ""
    if m_dev.type == "cuda" and torch.cuda.is_available():
        try:
            gpu_torch = torch.cuda.get_device_name(m_dev)
        except Exception:
            gpu_torch = str(m_dev)

    completed = load_completed_keys(out_csv, retry_failures=retry_failures) if resume else set()

    indexed_prompts: list[tuple[int, MBPrompt, Any, int]] = []
    for i, pr in enumerate(prompts_all):
        ntok = prompt_token_length(tok, pr.text)
        bucket = classify_context_bucket(ntok, short_max=short_max, medium_max=medium_max)
        if bucket.value not in buckets_filter:
            continue
        indexed_prompts.append((i, pr, bucket, ntok))

    grid = expand_sweep_grid(
        modes_canon,
        k_values,
        sparsity_budgets,
        quant_bits_list,
        strict_labeled_grid=strict_labeled_grid,
    )

    mw_bytes = model_weights_bytes(model)
    mw_gb = float(mw_bytes) / 1e9

    for mode, spec_k, sparsity_budg, qb_req in grid:
        b_label = benchmark_label_for_canonical_mode(mode)
        qb_req_i = int(qb_req)
        if mode == SWEEP_MODE_AUTOREGRESSIVE:
            draft_label = "n/a"
            qb_eff = -1
            qnote = ""
            draft_mode = DraftCacheMode.FP16
        else:
            draft_mode, qb_eff, draft_label, qnote = resolve_speculative_mode(mode, qb_req_i)

        for pi, pr, bucket, ntok in indexed_prompts:
            sparse_cfg: SparseRetentionConfig | None = None
            sparse_json = ""
            if mode in ("sparse_only", "sparse_quant"):
                sparse_cfg = build_sparse_config_for_prompt(
                    ntok,
                    sparsity_budget=sparsity_budg,
                    recent_window_fraction=recent_frac,
                    refresh_interval=refresh_iv,
                    scoring=sparse_scoring,
                )
                sparse_json = json.dumps(dataclasses.asdict(sparse_cfg), sort_keys=True)
            rw_disp = sparse_cfg.recent_window if sparse_cfg else ""
            hb_disp = sparse_cfg.heavy_hitter_budget if sparse_cfg else ""

            for trial in range(num_trials):
                key = (pr.id, mode, str(spec_k), str(sparsity_budg), str(qb_req_i), bucket.value, str(trial))
                if resume and key in completed:
                    continue

                for _ in range(warmup_runs):
                    reset_peak_memory_stats(m_dev)
                    try:
                        if mode == SWEEP_MODE_AUTOREGRESSIVE:
                            decode_greedy_autoregressive(
                                model,
                                tok,
                                pr.text,
                                max_new_tokens=max_new_tokens,
                                warmup=True,
                                trial_index=-1,
                            )
                        else:
                            SpeculativeDecoder(
                                model,
                                tok,
                                int(spec_k),
                                draft_mode=draft_mode,
                                verbose=False,
                                verify_match=verify_match,
                                sparse_config=sparse_cfg,
                                kv_quant_bits=qb_req_i,
                            ).decode(pr.text, max_new_tokens=max_new_tokens)
                    except Exception:
                        pass

                reset_peak_memory_stats(m_dev)
                if m_dev.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(m_dev)
                peak_before = max_memory_allocated_bytes(m_dev)
                status = "ok"
                fail = ""
                lat: float | str = ""
                acc: float | str = ""
                tps: float | str = ""
                draft_b: int | str = ""
                ver_b: int | str = ""
                metrics_blob: dict[str, Any] | None = None
                gen_n = 0
                draft_lat_s: float | str = ""
                verify_lat_s: float | str = ""
                peak_after = 0

                try:
                    if mode == SWEEP_MODE_AUTOREGRESSIVE:
                        res = decode_greedy_autoregressive(
                            model,
                            tok,
                            pr.text,
                            max_new_tokens=max_new_tokens,
                            warmup=False,
                            trial_index=trial,
                        )
                        lat = float(res.metrics.end_to_end_generation_s)
                        gen_n = int(res.metrics.new_tokens_generated)
                        tps = float(gen_n / float(lat)) if lat else ""
                        draft_b = int(res.metrics.logical_kv_cache_bytes)
                        ver_b = draft_b
                        metrics_blob = res.metrics.to_jsonable()
                    else:
                        out = SpeculativeDecoder(
                            model,
                            tok,
                            int(spec_k),
                            draft_mode=draft_mode,
                            verbose=False,
                            verify_match=verify_match,
                            sparse_config=sparse_cfg,
                            kv_quant_bits=qb_req_i,
                        ).decode(pr.text, max_new_tokens=max_new_tokens)
                        lat = float(out.metrics.total_runtime_s)
                        acc = float(out.metrics.acceptance_rate)
                        gen_n = int(max_new_tokens)
                        tps = float(gen_n / float(lat)) if lat else ""
                        draft_b = int(out.draft_kv.memory_bytes())
                        ver_b = int(out.verifier_kv.memory_bytes())
                        metrics_blob = out.metrics.to_jsonable()
                        draft_lat_s = float(out.metrics.draft_phase_time_s_total)
                        verify_lat_s = float(out.metrics.verify_phase_time_s_total)
                except Exception as exc:
                    status = "error"
                    fail = f"{type(exc).__name__}: {exc}"

                if m_dev.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(m_dev)
                peak_after = max_memory_allocated_bytes(m_dev)

                kv_gb = float(ver_b) / 1e9 if isinstance(ver_b, int) else float("nan")
                lat_per_tok: float | str = ""
                eff_bw: float | str = ""
                mem_tp: float | str = ""
                if status == "ok" and isinstance(lat, float) and lat > 0 and gen_n > 0:
                    lat_per_tok = float(lat) / float(gen_n)
                    eff_bw = (mw_gb + kv_gb) / lat_per_tok
                    mem_tp = eff_bw

                q_type = quantization_type_for_row(canonical_mode=mode, quant_bits_effective=int(qb_eff))
                parallel_ver = mode != SWEEP_MODE_AUTOREGRESSIVE
                _parallel_cell = "true" if parallel_ver else "false"
                sparse_ver_cell = SPARSE_INTEGRATION_VERSION if mode in ("sparse_only", "sparse_quant") else ""

                row_dict: dict[str, Any] = {
                    "sweep_id": sweep_id,
                    "timestamp_utc": _utc_ts(),
                    "status": status,
                    "failure_reason": fail,
                    "prompt_id": pr.id,
                    "prompt_idx": pi,
                    "context_bucket": bucket.value,
                    "prompt_len_tokens": ntok,
                    "mode": mode,
                    "benchmark_label": b_label,
                    "spec_k": spec_k,
                    "sparsity_budget": sparsity_budg,
                    "quant_bits_requested": qb_req_i,
                    "quant_bits_effective": qb_eff,
                    "recent_window": rw_disp,
                    "heavy_hitter_budget": hb_disp,
                    "sparse_refresh_interval": refresh_iv,
                    "sparse_scoring": sparse_scoring,
                    "sparse_config_json": sparse_json,
                    "draft_cache_mode_resolved": draft_label,
                    "quant_notes": qnote if mode != SWEEP_MODE_AUTOREGRESSIVE else "",
                    "trial_index": trial,
                    "warmup": "false",
                    "latency_e2e_s": lat,
                    "latency_per_new_token_s": lat_per_tok,
                    "acceptance_rate": acc,
                    "tokens_per_sec": tps,
                    "logical_draft_kv_bytes": draft_b,
                    "logical_verifier_kv_bytes": ver_b,
                    "gpu_torch_name": gpu_torch,
                    "modal_resource_tag": modal_resource_tag,
                    "model_name": model_name,
                    "max_new_tokens": max_new_tokens,
                    "verify_match": str(verify_match).lower(),
                    "device_type": str(m_dev.type),
                    "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
                    "is_parallel_verification": _parallel_cell,
                    "quantization_type": q_type,
                    "sparse_integration_version": sparse_ver_cell,
                    "gpu_peak_memory_bytes_before_run": peak_before,
                    "gpu_peak_memory_bytes_after_run": peak_after,
                    "model_weights_gb": mw_gb,
                    "kv_cache_size_gb": kv_gb if status == "ok" and isinstance(ver_b, int) else "",
                    "effective_memory_bandwidth_gb_s": eff_bw,
                    "memory_throughput_gb_s": mem_tp,
                    "draft_latency_total_s": draft_lat_s if mode != SWEEP_MODE_AUTOREGRESSIVE else "",
                    "verify_latency_total_s": verify_lat_s if mode != SWEEP_MODE_AUTOREGRESSIVE else "",
                }
                if mode == SWEEP_MODE_AUTOREGRESSIVE:
                    row_dict["draft_cache_mode_resolved"] = "n/a"
                    row_dict["quant_bits_effective"] = -1

                append_benchmark_csv_row(out_csv, row_dict, volume_commit_fn=volume_commit_fn)
                append_raw_jsonl(
                    raw_jsonl,
                    {"row": row_dict, "full_metrics": metrics_blob},
                )

    summ_path = Path(cfg.get("summary_csv", str(out_csv).replace(".csv", "_summary.csv")))
    try:
        write_sweep_summary_csv(out_csv, summ_path)
        print(f"[sweep] summary → {summ_path}", flush=True)
    except Exception as exc:
        print(f"[sweep] summary skipped: {exc}", flush=True)

    try:
        write_processed_rollup_json(
            out_csv,
            processed_json,
            sweep_id=sweep_id,
            model_name=model_name,
            sweep_yaml_resolved=str(p.resolve()),
            strict_labeled_grid=strict_labeled_grid,
        )
        print(f"[sweep] processed rollup → {processed_json}", flush=True)
    except Exception as exc:
        print(f"[sweep] processed rollup skipped: {exc}", flush=True)

    print(f"[sweep] done sweep_id={sweep_id} csv={out_csv}", flush=True)
    return 0


def _csv_float(x: str | None) -> float:
    try:
        return float(x or "")
    except (TypeError, ValueError):
        return float("nan")


def write_sweep_summary_csv(raw_csv: Path, summary_csv: Path) -> None:
    if not raw_csv.is_file():
        return
    with raw_csv.open(encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("status") == "ok"]
    if not rows:
        return

    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("benchmark_label", ""),
            row.get("mode"),
            row.get("spec_k"),
            row.get("sparsity_budget"),
            row.get("quant_bits_requested"),
            row.get("context_bucket"),
            row.get("draft_cache_mode_resolved"),
        )
        groups.setdefault(key, []).append(row)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fields = [
        "benchmark_label",
        "mode",
        "spec_k",
        "sparsity_budget",
        "quant_bits_requested",
        "context_bucket",
        "draft_cache_mode_resolved",
        "n_rows",
        "is_parallel_verification",
        "quantization_type",
        "mean_latency_e2e_s",
        "std_latency_e2e_s",
        "mean_latency_per_new_token_s",
        "mean_memory_throughput_gb_s",
        "mean_gpu_peak_memory_bytes_after_run",
        "mean_draft_latency_total_s",
        "mean_verify_latency_total_s",
        "mean_acceptance_rate",
        "mean_tokens_per_sec",
        "mean_logical_draft_kv_bytes",
        "gpu_torch_name",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for key, g_rows in sorted(groups.items(), key=lambda x: str(x[0])):
            lats = [_csv_float(r.get("latency_e2e_s")) for r in g_rows]
            lats = [x for x in lats if x == x]
            lat_tr = [_csv_float(r.get("latency_per_new_token_s")) for r in g_rows]
            lat_tr = [x for x in lat_tr if x == x]
            mem_tp = [_csv_float(r.get("memory_throughput_gb_s")) for r in g_rows]
            mem_tp = [x for x in mem_tp if x == x]
            peaks = [_csv_float(r.get("gpu_peak_memory_bytes_after_run")) for r in g_rows]
            peaks = [x for x in peaks if x == x]
            dlat = [_csv_float(r.get("draft_latency_total_s")) for r in g_rows if r.get("draft_latency_total_s", "") != ""]
            dlat = [x for x in dlat if x == x]
            vlat = [_csv_float(r.get("verify_latency_total_s")) for r in g_rows if r.get("verify_latency_total_s", "") != ""]
            vlat = [x for x in vlat if x == x]
            accs = [_csv_float(r.get("acceptance_rate")) for r in g_rows if r.get("acceptance_rate", "") != ""]
            accs = [x for x in accs if x == x]
            tpss = [_csv_float(r.get("tokens_per_sec")) for r in g_rows if r.get("tokens_per_sec", "") != ""]
            tpss = [x for x in tpss if x == x]
            drafts = [_csv_float(r.get("logical_draft_kv_bytes")) for r in g_rows if r.get("logical_draft_kv_bytes", "") != ""]
            drafts = [x for x in drafts if x == x]
            gpu = next((r.get("gpu_torch_name", "") for r in g_rows if r.get("gpu_torch_name")), "")
            par = next((r.get("is_parallel_verification", "") for r in g_rows if r.get("is_parallel_verification", "")), "")
            qtyp = next((r.get("quantization_type", "") for r in g_rows if r.get("quantization_type", "")), "")
            w.writerow(
                {
                    "benchmark_label": key[0],
                    "mode": key[1],
                    "spec_k": key[2],
                    "sparsity_budget": key[3],
                    "quant_bits_requested": key[4],
                    "context_bucket": key[5],
                    "draft_cache_mode_resolved": key[6],
                    "n_rows": len(g_rows),
                    "is_parallel_verification": par,
                    "quantization_type": qtyp,
                    "mean_latency_e2e_s": statistics.mean(lats) if lats else "",
                    "std_latency_e2e_s": statistics.stdev(lats) if len(lats) > 1 else 0.0,
                    "mean_latency_per_new_token_s": statistics.mean(lat_tr) if lat_tr else "",
                    "mean_memory_throughput_gb_s": statistics.mean(mem_tp) if mem_tp else "",
                    "mean_gpu_peak_memory_bytes_after_run": statistics.mean(peaks) if peaks else "",
                    "mean_draft_latency_total_s": statistics.mean(dlat) if dlat else "",
                    "mean_verify_latency_total_s": statistics.mean(vlat) if vlat else "",
                    "mean_acceptance_rate": statistics.mean(accs) if accs else "",
                    "mean_tokens_per_sec": statistics.mean(tpss) if tpss else "",
                    "mean_logical_draft_kv_bytes": statistics.mean(drafts) if drafts else "",
                    "gpu_torch_name": gpu,
                }
            )


def write_processed_rollup_json(
    raw_csv: Path,
    out_json: Path,
    *,
    sweep_id: str,
    model_name: str,
    sweep_yaml_resolved: str,
    strict_labeled_grid: bool,
) -> None:
    """Aggregate ``status=ok`` rows for charts / Phase 16 tables (resume-safe: overwrites file)."""

    if not raw_csv.is_file():
        return
    with raw_csv.open(encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("status") == "ok"]
    if not rows:
        return

    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        bl = str(row.get("benchmark_label", ""))
        by_label.setdefault(bl, []).append(row)

    def _mean(key: str, group: list[dict[str, str]]) -> float | None:
        xs = [_csv_float(r.get(key)) for r in group]
        xs = [x for x in xs if x == x]
        return float(statistics.mean(xs)) if xs else None

    rollup_labels: dict[str, Any] = {}
    for label, g in sorted(by_label.items()):
        rollup_labels[label] = {
            "n_rows": len(g),
            "mean_latency_e2e_s": _mean("latency_e2e_s", g),
            "mean_latency_per_new_token_s": _mean("latency_per_new_token_s", g),
            "mean_memory_throughput_gb_s": _mean("memory_throughput_gb_s", g),
            "mean_effective_memory_bandwidth_gb_s": _mean("effective_memory_bandwidth_gb_s", g),
            "mean_gpu_peak_memory_bytes_after_run": _mean("gpu_peak_memory_bytes_after_run", g),
            "mean_draft_latency_total_s": _mean("draft_latency_total_s", g),
            "mean_verify_latency_total_s": _mean("verify_latency_total_s", g),
            "mean_acceptance_rate": _mean("acceptance_rate", g),
            "mean_tokens_per_sec": _mean("tokens_per_sec", g),
            "is_parallel_verification": next(
                (r.get("is_parallel_verification") for r in g if r.get("is_parallel_verification")), ""
            ),
            "quantization_type": next((r.get("quantization_type") for r in g if r.get("quantization_type")), ""),
            "gpu_torch_name": next((r.get("gpu_torch_name") for r in g if r.get("gpu_torch_name")), ""),
        }

    first = rows[0]
    bundle = {
        "sweep_id": sweep_id,
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "generated_at_utc": _utc_ts(),
        "source_csv": str(raw_csv.resolve()),
        "sweep_yaml_resolved": sweep_yaml_resolved,
        "model_name": model_name,
        "strict_labeled_grid": strict_labeled_grid,
        "speculative_verifier": "parallel_block_verification",
        "rows_ok_total": len(rows),
        "by_benchmark_label": rollup_labels,
        "hardware": {
            "gpu_torch_name": first.get("gpu_torch_name", ""),
            "device_type": first.get("device_type", ""),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")


def dump_sweep_config_bundle(sweep_yaml: str | Path, out_json: Path) -> None:
    p = Path(sweep_yaml)
    cfg: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    if cfg is None:
        cfg = {}
    bundle = {
        "sweep_yaml_path": str(p.resolve()),
        "config": cfg,
        "env": {
            "MODAL_RESOURCE_TAG": os.environ.get("MODAL_RESOURCE_TAG", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
