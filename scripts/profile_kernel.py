#!/usr/bin/env python3
"""QuantSpec / hierarchical KV — kernel profiling harness (CUDA + Triton).

**Why warmup matters (read before interpreting traces)**

Short prompts on small models make Triton kernel **body time** (~10–50 µs) comparable to **launch** and
**JIT compile** time. Profiling **without** warming imports, allocator state, and the Triton cache
captures **importlib**, **profiler startup**, and first-kernel compile — not steady-state decode.

This script therefore supports:

* **Heavy pre-profiler warmup** (``--profiler-warmup-iters``) — JIT, allocator, repeated kernel work.
* **``--profile-scope``** — ``qk_kernel`` (isolated Q·K hist microbench) vs ``decode_ar_steady`` (greedy
  AR decode loop only; **HF baseline path**, not quant — useful for steady GPU timeline without Triton
  quant orchestration).

**Modes**

1. **Microbench** — Triton vs ref Q·K hist; ``triton.testing.do_bench``; analytical roofline.
2. **torch.profiler** — Only after warmup; captures steady-state ``work()`` repeats.
3. **Optional** ``--include-fused-verifier`` — toy fused verifier block timing.

**Run**

  PYTHONPATH=src python scripts/profile_kernel.py --device cuda

  PYTHONPATH=src python scripts/profile_kernel.py --device cuda --trace-out outputs/profile/trace.json \\
    --profiler-warmup-iters 50 --profile-scope qk_kernel

**Modal**

  modal run modal_phase_o_tools.py --tool profile_kernel --profile-kernel-extra "--profiler-warmup-iters 50"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch


def _ensure_src_path() -> None:
    src = _REPO_ROOT / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _warm_kv_imports() -> None:
    """Import Triton/kv_kernels before profiling so the trace is not dominated by importlib."""
    _ensure_src_path()
    from kv_kernels.triton_runtime import triton_available  # noqa: F401
    from kv_kernels.triton_attention import (  # noqa: F401
        qk_draft_dispatch,
        qk_target_dispatch,
    )
    from kv_kernels.tuning import get_preset_config, kernel_tuning_scope  # noqa: F401


def _allocator_warmup(device: torch.device) -> None:
    """Touch CUDA allocator / L2 so first real iteration is closer to steady state."""
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # A few moderate allocations + sync (no huge tensors — avoid OOM on small GPUs)
    for _ in range(3):
        _ = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    del _
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _make_hist_tensors(
    *,
    device: torch.device,
    d_model: int,
    seq_len: int,
    group_size: int,
    dtype_q: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, ...]:
    if d_model % group_size != 0:
        raise ValueError("d_model must divide group_size")
    torch.manual_seed(0)
    q = torch.randn(d_model, device=device, dtype=dtype_q)
    packed = torch.randint(0, 256, (seq_len, d_model), device=device, dtype=torch.uint8)
    n_g = d_model // group_size
    su = torch.randn(seq_len, n_g, device=device)
    zu = torch.randn(seq_len, n_g, device=device)
    sl = torch.randn(seq_len, n_g, device=device)
    zl = torch.randn(seq_len, n_g, device=device)
    return q, packed, su, zu, sl, zl


def _bench_torch_sync(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
            torch.cuda.synchronize()
        import time

        times: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    else:
        import time

        for _ in range(warmup):
            fn()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
    std = math.sqrt(var)
    return mean, std


def _triton_do_bench(fn: Callable[[], Any], *, warmup: int, rep: int) -> float | None:
    try:
        import triton.testing as triton_testing
    except Exception:
        return None
    try:
        r = triton_testing.do_bench(fn, warmup=warmup, rep=rep)
        if isinstance(r, (list, tuple)):
            return float(r[0])
        if hasattr(r, "item"):
            return float(r.item())
        return float(r)
    except Exception:
        try:
            r = triton_testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5])
            if isinstance(r, (list, tuple)):
                return float(r[0])
            return float(r)
        except Exception:
            return None


def _roofline_qk_hist_draft(
    *,
    s: int,
    d: int,
    group_size: int,
    bytes_per_elem_fp16: int = 2,
) -> dict[str, float]:
    n_g = d // group_size
    bytes_per_score = (
        d + 2 * n_g * 4 + d * bytes_per_elem_fp16
    )
    total_bytes = bytes_per_score * s
    flops = 2.0 * s * d
    ai = flops / max(total_bytes, 1.0)
    return {
        "total_bytes_est": float(total_bytes),
        "total_flops_est": float(flops),
        "arithmetic_intensity_flops_per_byte": float(ai),
    }


def _run_profiler_qk(
    *,
    device: torch.device,
    d_model: int,
    seq_len: int,
    group_size: int,
    backend: str,
    path: str,
    profiler_warmup_iters: int,
    repeat: int,
    trace_out: Path | None,
    tuning_profile: str,
) -> None:
    """Profile Q·K hist only. **Profiler window excludes first-time import** (call `_warm_kv_imports` first)."""
    _ensure_src_path()
    from kv_kernels.triton_attention import qk_draft_dispatch, qk_target_dispatch
    from kv_kernels.tuning import get_preset_config, kernel_tuning_scope

    q, packed, su, zu, sl, zl = _make_hist_tensors(
        device=device, d_model=d_model, seq_len=seq_len, group_size=group_size
    )
    cfg = get_preset_config(tuning_profile)

    def work() -> None:
        with kernel_tuning_scope(cfg):
            if path == "draft":
                _ = qk_draft_dispatch(
                    q, packed, su, zu, group_size=group_size, backend=backend
                )
            else:
                _ = qk_target_dispatch(
                    q,
                    packed,
                    su,
                    zu,
                    sl,
                    zl,
                    group_size=group_size,
                    backend=backend,
                )

    _allocator_warmup(device)
    print(
        f"[profile_kernel] pre-profiler warmup: {profiler_warmup_iters} iters (Triton JIT + allocator + kernel)",
        flush=True,
    )
    for _ in range(profiler_warmup_iters):
        work()
        if device.type == "cuda":
            torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(repeat):
            work()
            if device.type == "cuda":
                torch.cuda.synchronize()

    if trace_out is not None:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_out))
        print(f"[profile_kernel] Chrome trace (steady-state Q·K region): {trace_out.resolve()}", flush=True)

    key_avgs = prof.key_averages()
    rows = sorted(
        key_avgs,
        key=lambda e: getattr(e, "cuda_time_total", 0) or getattr(e, "cpu_time_total", 0) or 0,
        reverse=True,
    )[:25]
    print(f"\n[profile_kernel] Top events (path={path}, backend={backend}):")
    for i, e in enumerate(rows, 1):
        cuda = getattr(e, "cuda_time_total", 0) or 0
        cpu = getattr(e, "cpu_time_total", 0) or 0
        print(f"  {i:2d}. {e.key[:90]:90s}  cuda_us={cuda:8.0f}  cpu_us={cpu:8.0f}")
    print(
        "[profile_kernel] For launch vs kernel body, filter trace for 'cuLaunchKernel' vs '_qk_*' / Triton.",
        flush=True,
    )


def _run_profiler_decode_ar(
    *,
    device: torch.device,
    model_name: str,
    max_new_tokens: int,
    profiler_warmup_iters: int,
    repeat: int,
    trace_out: Path | None,
) -> None:
    """Steady-state **HF autoregressive** decode (no quant path) — isolates generic decode GPU timeline."""
    _ensure_src_path()
    from mlsys_kv.models.hf_loader import load_causal_lm  # noqa: E402
    from decoding.autoregressive import decode_greedy_autoregressive

    _allocator_warmup(device)
    print(f"[profile_kernel] loading model {model_name!r} for decode_ar_steady …", flush=True)
    loaded = load_causal_lm(model_name, device=device, dtype="float16")
    model = loaded.model
    tok = loaded.tokenizer
    model.eval()
    prompt = "The quick brown fox jumps. " * 120

    def work() -> None:
        decode_greedy_autoregressive(
            model,
            tok,
            prompt,
            max_new_tokens=max_new_tokens,
            warmup=False,
            trial_index=0,
        )

    print(
        f"[profile_kernel] pre-profiler warmup: {profiler_warmup_iters} decode runs (AR steady)",
        flush=True,
    )
    for _ in range(profiler_warmup_iters):
        work()
        if device.type == "cuda":
            torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(repeat):
            work()
            if device.type == "cuda":
                torch.cuda.synchronize()

    if trace_out is not None:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_out))
        print(f"[profile_kernel] Chrome trace (AR decode steady): {trace_out.resolve()}", flush=True)

    key_avgs = prof.key_averages()
    rows = sorted(
        key_avgs,
        key=lambda e: getattr(e, "cuda_time_total", 0) or getattr(e, "cpu_time_total", 0) or 0,
        reverse=True,
    )[:25]
    print("\n[profile_kernel] Top events (decode_ar_steady):")
    for i, e in enumerate(rows, 1):
        cuda = getattr(e, "cuda_time_total", 0) or 0
        cpu = getattr(e, "cpu_time_total", 0) or 0
        print(f"  {i:2d}. {e.key[:90]:90s}  cuda_us={cuda:8.0f}  cpu_us={cpu:8.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile QuantSpec Triton Q·K hist vs reference")
    ap.add_argument("--device", type=str, default="cuda", help="cuda | cpu (CPU skips Triton)")
    ap.add_argument("--d-model", type=int, default=1280, help="Hidden size D (gpt2-xl default 1280)")
    ap.add_argument("--seq-len", type=int, default=512, help="History length S (packed_hist rows)")
    ap.add_argument("--group-size", type=int, default=32, help="Quant group size")
    ap.add_argument("--warmup", type=int, default=10, help="Microbench warmup (mean/std section)")
    ap.add_argument("--iters", type=int, default=50, help="Timed iterations for manual mean/std")
    ap.add_argument("--triton-bench-warmup", type=int, default=25)
    ap.add_argument("--triton-bench-rep", type=int, default=100)
    ap.add_argument("--path", choices=("draft", "target", "both"), default="both")
    ap.add_argument("--profile-backend", choices=("ref", "triton", "both"), default="both")
    ap.add_argument("--trace-out", type=Path, default=None, help="Chrome trace JSON (torch.profiler)")
    ap.add_argument(
        "--profile-path",
        choices=("draft", "target"),
        default="draft",
        help="Which Q·K hist path when --profile-scope=qk_kernel",
    )
    ap.add_argument(
        "--profile-scope",
        choices=("qk_kernel", "decode_ar_steady"),
        default="qk_kernel",
        help="qk_kernel=isolated Q·K microbench; decode_ar_steady=HF greedy decode only (no quant)",
    )
    ap.add_argument(
        "--profiler-warmup-iters",
        type=int,
        default=40,
        help="Iterations before torch.profiler to warm JIT/allocator/kernel (not included in trace)",
    )
    ap.add_argument("--profile-repeat", type=int, default=8, help="Repeats inside profiler window")
    ap.add_argument(
        "--decode-model-name",
        type=str,
        default="gpt2",
        help="Model for decode_ar_steady (small default for quick GPU profiling)",
    )
    ap.add_argument(
        "--decode-max-new-tokens",
        type=int,
        default=40,
        help="max_new_tokens for decode_ar_steady",
    )
    ap.add_argument(
        "--include-fused-verifier",
        action="store_true",
        help="Also microbench fused_verifier_block_attention (Triton vs ref) with fixed small shapes",
    )
    ap.add_argument("--peak-gbps", type=float, default=600.0, help="Peak memory BW for roofline (A10G ~600)")
    ap.add_argument("--tuning-profile", type=str, default="default", help="kv_kernels.tuning preset id")
    args = ap.parse_args()

    _ensure_src_path()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[profile_kernel] ERROR: CUDA requested but not available.", file=sys.stderr)
        return 1

    # Warm imports before any timed section so microbench isn't dominated by importlib.
    _warm_kv_imports()

    from kv_kernels.triton_runtime import triton_available
    from kv_kernels.triton_attention import qk_draft_dispatch, qk_target_dispatch
    from kv_kernels.tuning import get_preset_config, kernel_tuning_scope

    if device.type == "cuda" and not triton_available():
        print("[profile_kernel] WARNING: Triton not installed; Triton sections will fail.", file=sys.stderr)

    d, s, gs = args.d_model, args.seq_len, args.group_size
    q, packed, su, zu, sl, zl = _make_hist_tensors(device=device, d_model=d, seq_len=s, group_size=gs)

    tune_cfg = get_preset_config(args.tuning_profile)
    print(f"[profile_kernel] tuning_profile={tune_cfg.profile_id}  D={d} S={s} GS={gs}")
    print(f"[profile_kernel] peak_mem_gbps (roofline ref)={args.peak_gbps}")

    roof = _roofline_qk_hist_draft(s=s, d=d, group_size=gs)
    print("\n[profile_kernel] Analytical roofline (draft Q·K hist, upper-only model):")
    print(f"  est_bytes_total={roof['total_bytes_est']:.0f}  est_flops={roof['total_flops_est']:.0f}")
    print(f"  arithmetic_intensity={roof['arithmetic_intensity_flops_per_byte']:.4f} FLOP/byte")
    peak = args.peak_gbps * 1e9
    t_mem_s = roof["total_bytes_est"] / peak if peak > 0 else float("nan")
    print(f"  if_memory_bound_at_{args.peak_gbps:.0f}_GB/s: min_time_sec={t_mem_s*1e6:.2f} µs (full S scores)")
    if roof["arithmetic_intensity_flops_per_byte"] < 50:
        print("  note: low AI → typically memory-bandwidth limited unless compute dominates elsewhere.")

    paths: list[str] = ["draft", "target"] if args.path == "both" else [args.path]
    backends: list[str] = ["ref", "triton"] if args.profile_backend == "both" else [args.profile_backend]

    for path in paths:
        print(f"\n--- {path} path (microbench mean/std) ---")
        for backend in backends:
            if backend == "triton" and not triton_available():
                print(f"  [{path}] triton: SKIP (not available)")
                continue

            def make_fn() -> Callable[[], Any]:
                def fn() -> None:
                    with kernel_tuning_scope(tune_cfg):
                        if path == "draft":
                            _ = qk_draft_dispatch(
                                q, packed, su, zu, group_size=gs, backend=backend
                            )
                        else:
                            _ = qk_target_dispatch(
                                q,
                                packed,
                                su,
                                zu,
                                sl,
                                zl,
                                group_size=gs,
                                backend=backend,
                            )

                return fn

            fn = make_fn()
            mean, std = _bench_torch_sync(fn, warmup=args.warmup, iters=args.iters)
            print(
                f"  [{path}] {backend:6s}  mean_ms={mean*1e3:.4f}  std_ms={std*1e3:.4f}  "
                f"(warmup={args.warmup} iters={args.iters})"
            )
            print(
                f"  [{path}] {backend:6s}  label=kernel_e2e_ms_per_call (includes launch+sync; not NCU launch split)",
                flush=True,
            )

            if backend == "triton" and device.type == "cuda":
                ms = _triton_do_bench(fn, warmup=args.triton_bench_warmup, rep=args.triton_bench_rep)
                if ms is not None:
                    print(
                        f"  [{path}] triton  triton_do_bench_median_ms={ms:.4f}  "
                        f"(warmup={args.triton_bench_warmup} rep={args.triton_bench_rep})"
                    )

    if args.include_fused_verifier and device.type == "cuda":
        from kv_kernels.fused_verifier_block_attention import fused_verifier_block_attention

        torch.manual_seed(1)
        h, d_f, s_hist, s_rec, gamma = 2, min(d, 64), min(s, 128), 2, 3
        gs_k, gs_v = gs, 8
        n_gk = d_f // gs_k
        n_gv = (s_hist + gs_v - 1) // gs_v
        qf = torch.randn(1, h, gamma, d_f, device=device, dtype=torch.float32)
        k_uq = torch.randint(0, 16, (h, s_hist, d_f), device=device, dtype=torch.int8)
        k_lq = torch.randint(0, 16, (h, s_hist, d_f), device=device, dtype=torch.int8)
        v_uq = torch.randint(0, 16, (h, s_hist, d_f), device=device, dtype=torch.int8)
        v_lq = torch.randint(0, 16, (h, s_hist, d_f), device=device, dtype=torch.int8)
        k_su = torch.randn(h, s_hist, n_gk, device=device) * 0.05
        k_zu = torch.randn(h, s_hist, n_gk, device=device) * 0.05
        k_sl = torch.randn(h, s_hist, n_gk, device=device) * 0.05
        k_zl = torch.randn(h, s_hist, n_gk, device=device) * 0.05
        v_su = torch.randn(h, n_gv, d_f, device=device) * 0.05
        v_zu = torch.randn(h, n_gv, d_f, device=device) * 0.05
        v_sl = torch.randn(h, n_gv, d_f, device=device) * 0.05
        v_zl = torch.randn(h, n_gv, d_f, device=device) * 0.05
        k_rec = torch.randn(h, s_rec, d_f, device=device)
        v_rec = torch.randn(h, s_rec, d_f, device=device)
        k_blk = torch.randn(h, gamma, d_f, device=device)
        v_blk = torch.randn(h, gamma, d_f, device=device)
        tup = (
            qf,
            k_uq,
            k_lq,
            k_su,
            k_zu,
            k_sl,
            k_zl,
            v_uq,
            v_lq,
            v_su,
            v_zu,
            v_sl,
            v_zl,
            k_rec,
            v_rec,
            k_blk,
            v_blk,
        )
        print("\n--- fused_verifier_block_attention (fixed toy shape) ---")
        with kernel_tuning_scope(tune_cfg):
            for b in ("ref", "triton"):
                bb = b

                def make_fv(backend: str) -> Callable[[], None]:
                    def fv_fn() -> None:
                        fused_verifier_block_attention(
                            *tup, group_size_k=gs_k, group_size_v=gs_v, backend=backend
                        )

                    return fv_fn

                fv_fn = make_fv(bb)
                mean_f, std_f = _bench_torch_sync(fv_fn, warmup=args.warmup, iters=args.iters)
                print(
                    f"  fused  {bb:9s}  mean_ms={mean_f*1e3:.4f}  std_ms={std_f*1e3:.4f}"
                )
                if bb == "triton":
                    msf = _triton_do_bench(fv_fn, warmup=args.triton_bench_warmup, rep=args.triton_bench_rep)
                    if msf is not None:
                        print(f"  fused  triton    do_bench_ms={msf:.4f}")

    if args.trace_out and device.type == "cuda":
        if args.profile_scope == "decode_ar_steady":
            _run_profiler_decode_ar(
                device=device,
                model_name=args.decode_model_name,
                max_new_tokens=args.decode_max_new_tokens,
                profiler_warmup_iters=args.profiler_warmup_iters,
                repeat=args.profile_repeat,
                trace_out=args.trace_out,
            )
        else:
            tb = "triton"
            _run_profiler_qk(
                device=device,
                d_model=d,
                seq_len=s,
                group_size=gs,
                backend=tb,
                path=args.profile_path,
                profiler_warmup_iters=args.profiler_warmup_iters,
                repeat=args.profile_repeat,
                trace_out=args.trace_out,
                tuning_profile=args.tuning_profile,
            )

    summary = {
        "d_model": d,
        "seq_len": s,
        "group_size": gs,
        "tuning_profile": tune_cfg.profile_id,
        "roofline_draft": roof,
        "peak_gbps_assumption": args.peak_gbps,
        "device": str(device),
        "profile_scope": args.profile_scope,
        "profiler_warmup_iters": args.profiler_warmup_iters,
    }
    out_json = (
        Path(os.environ.get("PROFILE_KERNEL_SUMMARY_JSON", ""))
        if os.environ.get("PROFILE_KERNEL_SUMMARY_JSON")
        else None
    )
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[profile_kernel] wrote {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
