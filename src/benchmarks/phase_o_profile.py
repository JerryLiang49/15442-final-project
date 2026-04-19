"""Phase O — post-fusion profiling: find the next hotspots (o-proj, rollover, softmax, etc.).

Runs ``torch.profiler`` on a short user-supplied Python snippet or a default **attention tensor** microbench.

Usage::

    PYTHONPATH=src python -m benchmarks.phase_o_profile --out outputs/phase_o/trace.json

Requires CUDA for meaningful kernel rows. Chrome trace: chrome://tracing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _default_microbench() -> None:
    from kv_kernels.tuning import kernel_tuning_scope, get_preset_config
    from kv_kernels.triton_attention import qk_draft_hist_triton
    from kv_kernels.triton_runtime import triton_available

    if not torch.cuda.is_available() or not triton_available():
        return
    dev = torch.device("cuda")
    d, s, gs = 128, 256, 8
    torch.manual_seed(0)
    q = torch.randn(d, device=dev, dtype=torch.float16)
    packed = torch.randint(0, 256, (s, d), device=dev, dtype=torch.uint8)
    su = torch.randn(s, d // gs, device=dev)
    zu = torch.randn(s, d // gs, device=dev)
    with kernel_tuning_scope(get_preset_config("default")):
        for _ in range(4):
            _ = qk_draft_hist_triton(q, packed, su, zu, group_size=gs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase O CUDA profiler (torch.profiler)")
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_o/prof.json"))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        print(
            "[phase_o_profile] No CUDA device: the default microbench is skipped (no Triton hist Q·K). "
            "Trace will be CPU-only; for GPU kernel rows use an NVIDIA machine or Modal.\n",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def work() -> None:
        _default_microbench()

    for _ in range(max(0, args.warmup)):
        work()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    if not cuda_ok:
        activities = [torch.profiler.ProfilerActivity.CPU]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(max(1, args.repeat)):
            work()

    prof.export_chrome_trace(str(args.out))
    print(f"[phase_o_profile] wrote Chrome trace: {args.out.resolve()}")
    key_avgs = prof.key_averages()
    if cuda_ok:
        top = sorted(
            key_avgs,
            key=lambda e: getattr(e, "cuda_time_total", 0) or 0,
            reverse=True,
        )[:20]
        print("[phase_o_profile] top CUDA (by total cuda time):")
        for i, e in enumerate(top, 1):
            ct = getattr(e, "cuda_time_total", 0) or 0
            print(f"  {i:2d}. {e.key[:100]:100s}  cuda_us={ct}")
    else:
        top = sorted(
            key_avgs,
            key=lambda e: getattr(e, "cpu_time_total", 0) or 0,
            reverse=True,
        )[:15]
        print("[phase_o_profile] top CPU (no CUDA on this machine):")
        for i, e in enumerate(top, 1):
            ct = getattr(e, "cpu_time_total", 0) or 0
            print(f"  {i:2d}. {e.key[:100]:100s}  cpu_us={ct}")


if __name__ == "__main__":
    main()
