"""Microbenchmark: Triton fused verifier timing vs active :mod:`kv_kernels.tuning` profile.

Does **not** replace end-to-end Phase N sweeps; isolates kernel launch + math for A/B tuning.

Usage::

    PYTHONPATH=src python -m benchmarks.kernel_tune_microbench --profile verifier_wide --iters 200
"""

from __future__ import annotations

import argparse
import time

import torch

from kv_kernels.fused_verifier_block_attention import fused_verifier_block_attention
from kv_kernels.triton_runtime import triton_available
from kv_kernels.tuning import get_preset_config, kernel_tuning_scope


def _tensors(dev: torch.device):
    torch.manual_seed(1)
    h, d, s_hist, s_rec, gamma = 2, 64, 12, 2, 3
    gs_k, gs_v = 8, 8
    n_gk = d // gs_k
    n_gv = (s_hist + gs_v - 1) // gs_v
    q = torch.randn(1, h, gamma, d, device=dev, dtype=torch.float32)
    k_uq = torch.randint(0, 16, (h, s_hist, d), device=dev, dtype=torch.int8)
    k_lq = torch.randint(0, 16, (h, s_hist, d), device=dev, dtype=torch.int8)
    v_uq = torch.randint(0, 16, (h, s_hist, d), device=dev, dtype=torch.int8)
    v_lq = torch.randint(0, 16, (h, s_hist, d), device=dev, dtype=torch.int8)
    k_su = torch.randn(h, s_hist, n_gk, device=dev) * 0.05
    k_zu = torch.randn(h, s_hist, n_gk, device=dev) * 0.05
    k_sl = torch.randn(h, s_hist, n_gk, device=dev) * 0.05
    k_zl = torch.randn(h, s_hist, n_gk, device=dev) * 0.05
    v_su = torch.randn(h, n_gv, d, device=dev) * 0.05
    v_zu = torch.randn(h, n_gv, d, device=dev) * 0.05
    v_sl = torch.randn(h, n_gv, d, device=dev) * 0.05
    v_zl = torch.randn(h, n_gv, d, device=dev) * 0.05
    k_rec = torch.randn(h, s_rec, d, device=dev)
    v_rec = torch.randn(h, s_rec, d, device=dev)
    k_blk = torch.randn(h, gamma, d, device=dev)
    v_blk = torch.randn(h, gamma, d, device=dev)
    return (
        q,
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
        gs_k,
        gs_v,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=str, default="default")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not triton_available():
        raise SystemExit(
            "CUDA + Triton required (NVIDIA GPU + Triton). "
            "This microbench does not run on CPU-only PyTorch (e.g. Mac). "
            "Use a Linux machine with CUDA, or run on Modal with a GPU image."
        )

    dev = torch.device("cuda")
    cfg = get_preset_config(args.profile)

    tup = _tensors(dev)
    *tensors, gs_k, gs_v = tup

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with kernel_tuning_scope(cfg):
        for _ in range(args.iters):
            fused_verifier_block_attention(
                *tensors, group_size_k=gs_k, group_size_v=gs_v, backend="triton"
            )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[kernel_tune_microbench] profile={cfg.profile_id} iters={args.iters} total_s={dt:.4f}")
    print(f"  per_iter_ms={1000.0 * dt / args.iters:.4f}")


if __name__ == "__main__":
    main()
