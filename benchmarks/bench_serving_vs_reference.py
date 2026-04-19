#!/usr/bin/env python3
"""Compare dense speculative **reference** vs **serving-oriented** orchestration (wall time).

**Reference path** (legacy): ``legacy_double_clone_verifier=True`` — extra
``clone_past_key_values`` on the verifier before each verify (old double-clone behavior).

**Serving path**: ``serving_mode=True``, ``legacy_double_clone_verifier=False`` — single clone inside
:func:`decoding.speculative_dense.verify_block_and_commit`, reused draft :class:`~cache.kv_cache_fp16.KVCacheFP16`
wrapper, request-state bookkeeping.

Both runs use the same model weights and should produce **identical** token ids (disable AR cross-check
for pure timing by passing ``--no-ar-check``).

Usage (from repo root)::

    PYTHONPATH=src python benchmarks/bench_serving_vs_reference.py --model gpt2 --max-new-tokens 64 --gamma 4

"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def _ensure_src_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def main() -> None:
    _ensure_src_path()

    from decoding.autoregressive import model_device
    from decoding.speculative_dense import SpeculativeDecoderDense
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="gpt2", help="HF model id (default: gpt2)")
    p.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt text")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1, help="Warmup decode runs before timing")
    p.add_argument("--repeats", type=int, default=3, help="Timed repeats per configuration")
    p.add_argument(
        "--no-ar-check",
        action="store_true",
        help="Skip speculative vs autoregressive equality (faster benchmark setup)",
    )
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    device = model_device(model)
    _ = device

    verify_ar = not args.no_ar_check

    def run(*, serving_mode: bool, legacy: bool) -> tuple[float, torch.Tensor]:
        dec = SpeculativeDecoderDense(
            model,
            tok,
            gamma=args.gamma,
            verify_match_autoregressive=verify_ar,
            serving_mode=serving_mode,
            legacy_double_clone_verifier=legacy,
        )
        t0 = time.perf_counter()
        out = dec.decode(args.prompt, max_new_tokens=args.max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed, out.full_token_ids

    # Warmup (same config as serving to compile kernels / caches)
    for _ in range(max(0, args.warmup)):
        run(serving_mode=True, legacy=False)

    ref_times: list[float] = []
    srv_times: list[float] = []

    for _ in range(args.repeats):
        t_r, ids_ref = run(serving_mode=False, legacy=True)
        t_s, ids_srv = run(serving_mode=True, legacy=False)
        ref_times.append(t_r)
        srv_times.append(t_s)
        if not torch.equal(ids_ref, ids_srv):
            raise RuntimeError("Token mismatch between reference and serving paths (correctness bug).")

    def mean(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    mr = mean(ref_times)
    ms = mean(srv_times)
    speedup = mr / ms if ms > 0 else float("nan")

    print(f"model={args.model} max_new_tokens={args.max_new_tokens} gamma={args.gamma} repeats={args.repeats}")
    print(f"reference (legacy double clone): mean {mr:.4f}s  (per run: {ref_times})")
    print(f"serving (single clone + draft reuse): mean {ms:.4f}s  (per run: {srv_times})")
    print(f"speedup (reference_time / serving_time): {speedup:.3f}x")


if __name__ == "__main__":
    main()
