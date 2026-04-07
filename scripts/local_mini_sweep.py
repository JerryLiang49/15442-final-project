#!/usr/bin/env python3
"""Local Phase 7 mini-sweep: compare autoregressive vs four speculative draft modes on GPT-2.

Runs entirely in-process (no subprocess). Uses the same building blocks as
``mlsys_kv.main.run_speculative`` / :class:`~mlsys_kv.decoding.speculative.SpeculativeDecoder`
(but loads the model **once** for all modes; ``run_speculative`` reloads per invocation and
writes JSONL — better for full experiments than for this tight grid).

Usage (from repo root, after ``pip install -e .``)::

    python scripts/local_mini_sweep.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

# Repo root on sys.path when running as script without install is fragile; prefer editable install.
_PROMPTS = [
    "The number one is",
    "Hello, I am",
    "Once upon a time",
]
_MODEL = "gpt2"
_MAX_NEW = 20
_SPEC_K = 3
_SEED = 42


@dataclass(frozen=True)
class RunRecord:
    mode: str
    prompt: str
    acceptance_rate: float | None
    tokens_per_second: float | None
    verify_match: bool


def _print_results_table(records: list[RunRecord]) -> None:
    """Aggregate by mode and print a Markdown summary."""

    modes: list[str] = []
    for r in records:
        if r.mode not in modes:
            modes.append(r.mode)

    print("\n## Mini-sweep summary (mean over prompts)\n")
    print("| Mode | Avg acceptance rate | Avg tokens/s | All prompts `verify_match` |")
    print("|------|---------------------|-------------|------------------------------|")

    for mode in modes:
        sub = [r for r in records if r.mode == mode]
        acc_vals = [r.acceptance_rate for r in sub if r.acceptance_rate is not None]
        tps_vals = [r.tokens_per_second for r in sub if r.tokens_per_second is not None]
        avg_acc = sum(acc_vals) / len(acc_vals) if acc_vals else None
        avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else None
        all_verify = all(r.verify_match for r in sub)

        acc_s = f"{avg_acc:.4f}" if avg_acc is not None else "—"
        tps_s = f"{avg_tps:.2f}" if avg_tps is not None else "—"
        ver_s = "✓" if all_verify else "✗"
        print(f"| {mode} | {acc_s} | {tps_s} | {ver_s} |")

    print("\n### Per-prompt rows\n")
    print("| Mode | Prompt (preview) | Acceptance rate | Tokens/s | verify_match |")
    print("|------|------------------|-----------------|----------|--------------|")
    for r in records:
        prev = (r.prompt[:28] + "…") if len(r.prompt) > 29 else r.prompt
        ar = f"{r.acceptance_rate:.4f}" if r.acceptance_rate is not None else "—"
        ts = f"{r.tokens_per_second:.2f}" if r.tokens_per_second is not None else "—"
        vm = "✓" if r.verify_match else "✗"
        print(f"| {r.mode} | {prev} | {ar} | {ts} | {vm} |")


def main() -> int:
    from mlsys_kv.benchmarks.memory import reset_peak_memory_stats
    from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
    from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
    from mlsys_kv.decoding.autoregressive import decode_greedy_autoregressive, model_device
    from mlsys_kv.decoding.speculative import SpeculativeDecoder
    from mlsys_kv.infra.device import resolve_device
    from mlsys_kv.infra.seed import set_seed
    from mlsys_kv.models.hf_loader import load_causal_lm

    set_seed(_SEED)
    device = resolve_device("auto")

    # float32 keeps GPT-2 stable on CPU/Mac; use float16 when you have CUDA/MPS + half speedups.
    loaded = load_causal_lm(_MODEL, device=device, dtype="float32")
    model = loaded.model
    tok = loaded.tokenizer
    model.eval()
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    mdev = model_device(model)
    sparse_cfg = SparseRetentionConfig(
        recent_window=8,
        heavy_hitter_budget=8,
        refresh_interval=2,
        scoring="key_norm",
    )

    spec_modes: list[tuple[str, DraftCacheMode, SparseRetentionConfig | None]] = [
        ("speculative (FP16)", DraftCacheMode.FP16, None),
        ("speculative (quant-only)", DraftCacheMode.QUANT_ONLY, None),
        ("speculative (sparse-only)", DraftCacheMode.SPARSE_ONLY, sparse_cfg),
        ("speculative (sparse + quant joint)", DraftCacheMode.SPARSE_QUANT, sparse_cfg),
    ]

    records: list[RunRecord] = []

    for prompt in _PROMPTS:
        reset_peak_memory_stats(mdev)
        try:
            ar_res = decode_greedy_autoregressive(
                model,
                tok,
                prompt,
                max_new_tokens=_MAX_NEW,
                warmup=False,
                trial_index=0,
            )
            m_ar = ar_res.metrics
            tps = m_ar.new_tokens_per_sec_e2e
            if tps is None and m_ar.end_to_end_generation_s > 0 and m_ar.new_tokens_generated > 0:
                tps = m_ar.new_tokens_generated / m_ar.end_to_end_generation_s
            records.append(
                RunRecord(
                    mode="autoregressive (baseline)",
                    prompt=prompt,
                    acceptance_rate=None,
                    tokens_per_second=float(tps) if tps is not None else None,
                    verify_match=True,
                )
            )
        except Exception as exc:
            print(f"[baseline] FAIL prompt={prompt!r}: {exc}", file=sys.stderr)
            records.append(
                RunRecord(
                    mode="autoregressive (baseline)",
                    prompt=prompt,
                    acceptance_rate=None,
                    tokens_per_second=None,
                    verify_match=False,
                )
            )

        for label, draft_mode, sc in spec_modes:
            reset_peak_memory_stats(mdev)
            dec = SpeculativeDecoder(
                model,
                tok,
                _SPEC_K,
                draft_mode=draft_mode,
                verbose=False,
                verify_match=True,
                sparse_config=sc,
            )
            try:
                res = dec.decode(prompt, max_new_tokens=_MAX_NEW)
                m = res.metrics
                gen_rate = (
                    float(m.total_new_tokens) / m.total_runtime_s if m.total_runtime_s > 0 else None
                )
                records.append(
                    RunRecord(
                        mode=label,
                        prompt=prompt,
                        acceptance_rate=float(m.acceptance_rate),
                        tokens_per_second=gen_rate,
                        verify_match=True,
                    )
                )
            except RuntimeError as exc:
                print(f"[{label}] verify_match FAIL prompt={prompt!r}: {exc}", file=sys.stderr)
                records.append(
                    RunRecord(
                        mode=label,
                        prompt=prompt,
                        acceptance_rate=None,
                        tokens_per_second=None,
                        verify_match=False,
                    )
                )
            except Exception as exc:
                print(f"[{label}] ERROR prompt={prompt!r}: {exc}", file=sys.stderr)
                records.append(
                    RunRecord(
                        mode=label,
                        prompt=prompt,
                        acceptance_rate=None,
                        tokens_per_second=None,
                        verify_match=False,
                    )
                )

    _print_results_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
