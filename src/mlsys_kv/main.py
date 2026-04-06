"""High-level orchestration entrypoints (smoke, benchmarks)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from mlsys_kv.benchmarks.memory import reset_peak_memory_stats
from mlsys_kv.benchmarks.metrics import summarize_decode_latencies, summarize_prompt_trials
from mlsys_kv.decoding.autoregressive import autoregressive_smoke_generate, decode_greedy_autoregressive, model_device
from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.decoding.speculative import SpeculativeDecoder
from mlsys_kv.infra.config import RunConfig
from mlsys_kv.infra.device import device_metadata, resolve_device
from mlsys_kv.infra.env_meta import collect_env_metadata
from mlsys_kv.infra.logging_utils import RunLogContext
from mlsys_kv.infra.seed import set_seed
from mlsys_kv.models.hf_loader import load_causal_lm


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def run_smoke(config: RunConfig) -> int:
    """Execute Phase-1 autoregressive smoke: load model, generate, log metrics.

    Returns:
        ``0`` on success.

    Raises:
        Exception: Any propagation from model load / generation after structured logging
        in :class:`~mlsys_kv.infra.logging_utils.RunLogContext`.
    """
    out_dir = Path(config.output_dir)
    env_meta = collect_env_metadata()

    with RunLogContext(output_dir=out_dir, environment=env_meta) as run:
        set_seed(config.seed)
        device = resolve_device(config.device)
        run.log_event(
            "config_resolved",
            config=config.to_dict(),
            device=device_metadata(device),
            prompt_id=_prompt_id(config.prompt),
        )

        run.log_event("model_load_start", model_name=config.model_name)
        loaded = load_causal_lm(
            config.model_name,
            device=device,
            torch_dtype=config.torch_dtype,
        )
        run.log_event(
            "model_load_end",
            model_name=config.model_name,
            device=device_metadata(device),
        )

        m_dev = model_device(loaded.model)
        reset_peak_memory_stats(m_dev)

        run.log_event(
            "generate_start",
            prompt_preview=config.prompt[:200],
            prompt_id=_prompt_id(config.prompt),
            max_new_tokens=config.max_new_tokens,
        )
        text, gen_stats = autoregressive_smoke_generate(
            loaded.model,
            loaded.tokenizer,
            config.prompt,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )
        run.log_event("generate_end", **gen_stats)

        run.log_event(
            "result",
            generated_text=text,
            prompt_id=_prompt_id(config.prompt),
        )

        print("\n=== Generated text ===\n")
        print(text)
        print("\n=== Timing ===\n")
        print(f"wall_time_s: {gen_stats['wall_time_s']:.4f}")

    return 0


def run_baseline(config: RunConfig, prompts: list[str]) -> int:
    """Run instrumented autoregressive decoding for one or more prompts; log JSONL rows."""
    if config.num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if config.warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    out_dir = Path(config.output_dir)
    env_meta = collect_env_metadata()

    with RunLogContext(output_dir=out_dir, environment=env_meta) as run:
        set_seed(config.seed)
        device = resolve_device(config.device)
        run.log_event(
            "baseline_config",
            config=config.to_dict(),
            device=device_metadata(device),
            num_prompts=len(prompts),
        )

        run.log_event("model_load_start", model_name=config.model_name)
        loaded = load_causal_lm(
            config.model_name,
            device=device,
            torch_dtype=config.torch_dtype,
        )
        run.log_event("model_load_end", model_name=config.model_name, device=device_metadata(device))

        m_dev = model_device(loaded.model)
        dev_meta = device_metadata(m_dev)

        for prompt in prompts:
            pid = _prompt_id(prompt)
            run.log_event(
                "baseline_prompt_start",
                prompt_id=pid,
                prompt_preview=prompt[:200],
                prompt_len_chars=len(prompt),
            )

            for _ in range(config.warmup_runs):
                reset_peak_memory_stats(m_dev)
                decode_greedy_autoregressive(
                    loaded.model,
                    loaded.tokenizer,
                    prompt,
                    max_new_tokens=config.max_new_tokens,
                    warmup=True,
                    trial_index=-1,
                )

            measured = []
            last_text = ""
            for trial_idx in range(config.num_trials):
                reset_peak_memory_stats(m_dev)
                res = decode_greedy_autoregressive(
                    loaded.model,
                    loaded.tokenizer,
                    prompt,
                    max_new_tokens=config.max_new_tokens,
                    warmup=False,
                    trial_index=trial_idx,
                )
                last_text = res.text
                measured.append(res.metrics)
                latency_sum = summarize_decode_latencies(res.metrics.decode_step_times_s)
                run.log_event(
                    "baseline_trial",
                    phase="autoregressive_baseline",
                    prompt_id=pid,
                    model_name=config.model_name,
                    seed=config.seed,
                    device=str(m_dev),
                    gpu_name=dev_meta.get("gpu_name"),
                    prompt_preview=prompt[:200],
                    metrics=res.metrics.to_jsonable(),
                    decode_latency_summary=latency_sum,
                    generated_text=res.text,
                )

            summary = summarize_prompt_trials(pid, measured)
            run.log_event(
                "baseline_prompt_summary",
                phase="autoregressive_baseline",
                model_name=config.model_name,
                seed=config.seed,
                device=str(m_dev),
                gpu_name=dev_meta.get("gpu_name"),
                summary=summary.to_jsonable(),
            )

            print(f"\n=== Prompt {pid} ===\n")
            print(last_text)
            print("\n=== Mean trial metrics ===\n")
            print(f"mean_prefill_s: {summary.mean_prefill_s:.6f}")
            print(f"mean_total_decode_s: {summary.mean_total_decode_s:.6f}")
            print(f"mean_e2e_s: {summary.mean_e2e_s:.6f}")
            print(f"mean_new_tokens_per_sec: {summary.mean_new_tokens_per_sec}")
            print(f"mean_peak_cuda_bytes: {summary.mean_peak_cuda_bytes:.0f}")
            print(f"mean_logical_kv_bytes: {summary.mean_logical_kv_bytes:.0f}")

    return 0


def run_speculative(
    config: RunConfig,
    prompts: list[str],
    *,
    verbose: bool = False,
    verify_match: bool = True,
) -> int:
    """Run uncompressed FP16 self-speculative decode; log metrics to JSONL."""
    out_dir = Path(config.output_dir)
    env_meta = collect_env_metadata()

    with RunLogContext(output_dir=out_dir, environment=env_meta) as run:
        set_seed(config.seed)
        device = resolve_device(config.device)
        run.log_event(
            "speculative_config",
            config=config.to_dict(),
            device=device_metadata(device),
            num_prompts=len(prompts),
            verbose=verbose,
            verify_match=verify_match,
        )

        run.log_event("model_load_start", model_name=config.model_name)
        loaded = load_causal_lm(
            config.model_name,
            device=device,
            torch_dtype=config.torch_dtype,
        )
        run.log_event("model_load_end", model_name=config.model_name, device=device_metadata(device))

        m_dev = model_device(loaded.model)
        dev_meta = device_metadata(m_dev)

        for prompt in prompts:
            pid = _prompt_id(prompt)
            draft_mode = DraftCacheMode.from_string(config.draft_cache_mode)
            sparse_cfg: SparseRetentionConfig | None = None
            if draft_mode is DraftCacheMode.SPARSE_ONLY:
                ss = config.sparse_scoring
                if ss not in ("attention", "key_norm"):
                    raise ValueError(
                        f"sparse_scoring must be 'attention' or 'key_norm', got {ss!r}"
                    )
                sparse_cfg = SparseRetentionConfig(
                    recent_window=config.sparse_recent_window,
                    heavy_hitter_budget=config.sparse_heavy_hitter_budget,
                    refresh_interval=config.sparse_refresh_interval,
                    scoring=ss,  # narrowed
                )
            run.log_event(
                "speculative_prompt_start",
                prompt_id=pid,
                spec_k=config.spec_k,
                draft_cache_mode=draft_mode.value,
                max_new_tokens=config.max_new_tokens,
                prompt_preview=prompt[:200],
                sparse_config=asdict(sparse_cfg) if sparse_cfg else None,
            )

            decoder = SpeculativeDecoder(
                loaded.model,
                loaded.tokenizer,
                config.spec_k,
                draft_mode=draft_mode,
                verbose=verbose,
                verify_match=verify_match,
                sparse_config=sparse_cfg,
            )
            res = decoder.decode(prompt, max_new_tokens=config.max_new_tokens)

            run.log_event(
                "speculative_trial",
                phase="speculative_uncompressed_fp16",
                prompt_id=pid,
                model_name=config.model_name,
                seed=config.seed,
                device=str(m_dev),
                gpu_name=dev_meta.get("gpu_name"),
                logical_verifier_kv_bytes=res.verifier_kv.memory_bytes(),
                logical_draft_kv_bytes=res.draft_kv.memory_bytes(),
                metrics=res.metrics.to_jsonable(),
                generated_text=res.text,
            )

            print(f"\n=== Speculative prompt {pid} ===\n")
            print(res.text)
            print("\n=== Speculative metrics ===\n")
            print(res.metrics.to_jsonable())

    return 0
