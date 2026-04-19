#!/usr/bin/env python3
"""Microbenchmark: kernel latency, bytes moved, correctness vs reference.

Run (CUDA): ``PYTHONPATH=src python benchmarks/kernels_microbench.py``

Reports:

* Wall time for reference vs Triton Q·K score passes (draft + target).
* Bytes read for packed history (approximate: ``S * D`` bytes + metadata).
* ``max |ref - triton|`` for both kernels.
"""

from __future__ import annotations

import argparse
import time

import torch

from kv_kernels.reference_attention import qk_scores_draft_upper_only, qk_scores_target_upper_plus_lower
from kv_kernels.triton_attention import qk_draft_hist_triton, qk_target_hist_triton
from kv_kernels.triton_runtime import triton_available


def _bench(name: str, fn, n_warmup: int, n_iter: int) -> float:
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--s", type=int, default=512, help="history length")
    p.add_argument("--d", type=int, default=64, help="head dim")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; microbench requires GPU.")
        return
    if not triton_available():
        print("Triton not installed; pip install triton")
        return

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    s, d, gs = args.s, args.d, args.group_size
    ng = d // gs
    q = torch.randn(d, device=dev, dtype=torch.float16)
    packed = torch.randint(0, 256, (s, d), device=dev, dtype=torch.uint8)
    su = torch.randn(s, ng, device=dev, dtype=torch.float16)
    zu = torch.randn(s, ng, device=dev, dtype=torch.float16)
    sl = torch.randn(s, ng, device=dev, dtype=torch.float16) * 0.05
    zl = torch.randn(s, ng, device=dev, dtype=torch.float16) * 0.05

    bytes_packed = packed.numel() * packed.element_size()
    bytes_meta = (su.numel() + zu.numel() + sl.numel() + zl.numel()) * 2
    print(f"shape S={s} D={d} group_size={gs}")
    print(f"bytes (packed K hist): {bytes_packed}")
    print(f"bytes (fp16 metadata upper+lower): {bytes_meta}")

    ref_d = qk_scores_draft_upper_only(q, packed, su, zu, group_size=gs)
    tri_d = qk_draft_hist_triton(q, packed, su, zu, group_size=gs)
    err_d = (ref_d - tri_d).abs().max().item()
    print(f"draft max|ref-triton|: {err_d}")

    ref_t = qk_scores_target_upper_plus_lower(q, packed, su, zu, sl, zl, group_size=gs)
    tri_t = qk_target_hist_triton(q, packed, su, zu, sl, zl, group_size=gs)
    err_t = (ref_t - tri_t).abs().max().item()
    print(f"target max|ref-triton|: {err_t}")

    t_ref_d = _bench(
        "ref_draft",
        lambda: qk_scores_draft_upper_only(q, packed, su, zu, group_size=gs),
        args.warmup,
        args.iters,
    )
    t_tri_d = _bench(
        "tri_draft",
        lambda: qk_draft_hist_triton(q, packed, su, zu, group_size=gs),
        args.warmup,
        args.iters,
    )
    t_ref_t = _bench(
        "ref_target",
        lambda: qk_scores_target_upper_plus_lower(q, packed, su, zu, sl, zl, group_size=gs),
        args.warmup,
        args.iters,
    )
    t_tri_t = _bench(
        "tri_target",
        lambda: qk_target_hist_triton(q, packed, su, zu, sl, zl, group_size=gs),
        args.warmup,
        args.iters,
    )

    print(f"latency ref draft (s): {t_ref_d:.6e}")
    print(f"latency triton draft (s): {t_tri_d:.6e}")
    print(f"latency ref target (s): {t_ref_t:.6e}")
    print(f"latency triton target (s): {t_tri_t:.6e}")


if __name__ == "__main__":
    main()
