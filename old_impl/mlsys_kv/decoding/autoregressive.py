"""Token-by-token greedy autoregressive decoding with explicit KV and timings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from mlsys_kv.benchmarks.memory import max_memory_allocated_bytes, reset_peak_memory_stats
from mlsys_kv.benchmarks.metrics import AutoregressiveRunMetrics
from mlsys_kv.benchmarks.timer import timed_cuda_interval
from mlsys_kv.cache.kv_cache_fp16 import KVCacheFP16


def model_device(model: PreTrainedModel) -> torch.device:
    """Return the device hosting model parameters (works with ``device_map`` models)."""
    return next(model.parameters()).device


@dataclass(frozen=True)
class AutoregressiveDecodeResult:
    """Outputs from one greedy autoregressive run."""

    text: str
    prompt_token_ids: torch.Tensor
    new_token_ids: torch.Tensor
    full_token_ids: torch.Tensor
    metrics: AutoregressiveRunMetrics


def decode_greedy_autoregressive(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int,
    warmup: bool,
    trial_index: int,
    kv_cache: KVCacheFP16 | None = None,
) -> AutoregressiveDecodeResult:
    """Greedy decoding with an explicit prefill pass and a per-token decode loop.

    **Timing semantics**

    * **Prefill** times one forward over the full prompt that materializes ``past_key_values``
      and logits for the final prompt position. The first new token is chosen from those logits;
      there is no separate forward for that token.
    * **Decode steps** time each incremental forward for tokens ``2..N`` (``N = max_new_tokens``).
      There are ``max(0, N-1)`` such forwards. Step latencies are stored in
      ``metrics.decode_step_times_s``.
    * **End-to-end generation time** is ``prefill + sum(decode steps)``. Tokenization,
      model load, and Python overhead outside the timed regions remain excluded unless you wrap
      this function more broadly.

    **Memory semantics**

    CUDA peak allocator stats are read after the run (caller should ``reset_peak_memory_stats``
    before calling for a clean peak). ``logical_kv_cache_bytes`` reflects the FP16 wrapper's
    accounting over ``past_key_values`` after the final forward.

    Args:
        model: Causal LM in eval mode.
        tokenizer: Matching tokenizer (``pad_token`` set).
        prompt: Prompt text.
        max_new_tokens: Count of new tokens to append after the prompt.
        warmup: Whether this execution is a warmup trial (logged on metrics only).
        trial_index: Trial id for logging aggregation.
        kv_cache: Optional cache instance; a fresh :class:`~mlsys_kv.cache.kv_cache_fp16.KVCacheFP16`
            is created when omitted.

    Returns:
        Decoded text, token id tensors, and structured metrics.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    device = model_device(model)
    kv = kv_cache or KVCacheFP16()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])
    decode_step_times: list[float] = []

    with torch.inference_mode():
        with timed_cuda_interval(device) as prefill_slot:
            prefill_out = model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=None,
            )
            kv.append_from_forward_output(prefill_out.past_key_values)
        prefill_s = float(prefill_slot[0])

        if max_new_tokens == 0:
            gen_chunk = input_ids.new_empty((1, 0), dtype=input_ids.dtype)
        else:
            next_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            chunks: list[torch.Tensor] = [next_id]

            for _ in range(max_new_tokens - 1):
                cur = chunks[-1]
                with timed_cuda_interval(device) as step_slot:
                    step_out = model(
                        input_ids=cur,
                        past_key_values=kv.get_attention_kv(),
                        use_cache=True,
                    )
                    kv.append_from_forward_output(step_out.past_key_values)
                decode_step_times.append(float(step_slot[0]))
                nxt = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                chunks.append(nxt)

            gen_chunk = torch.cat(chunks, dim=1)

    full = torch.cat([input_ids, gen_chunk], dim=1) if gen_chunk.numel() else input_ids
    text = tokenizer.decode(full[0], skip_special_tokens=True)

    total_decode_s = float(sum(decode_step_times))
    e2e_s = float(prefill_s + total_decode_s)
    new_n = int(gen_chunk.shape[1]) if gen_chunk.numel() else 0
    tps = (new_n / e2e_s) if e2e_s > 0 and new_n > 0 else None

    peak = max_memory_allocated_bytes(device)
    logical = int(kv.memory_bytes())

    metrics = AutoregressiveRunMetrics(
        prefill_time_s=prefill_s,
        decode_step_times_s=list(decode_step_times),
        total_decode_time_s=total_decode_s,
        end_to_end_generation_s=e2e_s,
        new_tokens_per_sec_e2e=tps,
        prompt_len_tokens=prompt_len,
        max_new_tokens=max_new_tokens,
        new_tokens_generated=new_n,
        peak_cuda_allocated_bytes=int(peak),
        logical_kv_cache_bytes=logical,
        warmup=warmup,
        trial_index=int(trial_index),
    )

    return AutoregressiveDecodeResult(
        text=text,
        prompt_token_ids=input_ids.detach().cpu(),
        new_token_ids=gen_chunk.detach().cpu() if gen_chunk.numel() else gen_chunk.cpu(),
        full_token_ids=full.detach().cpu(),
        metrics=metrics,
    )


def autoregressive_smoke_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, dict[str, Any]]:
    """Backward-compatible smoke helper using the manual decoder (Phase 2).

    Note:
        The ``device`` argument is accepted for API compatibility; tensors are placed on
        ``model``'s parameter device. CUDA peak memory is **not** reset here; for benchmark-grade
        numbers use :func:`decode_greedy_autoregressive` with an explicit reset in the runner.
    """
    _ = device  # noqa: F841 — tokenizer/model placement follows ``model_device(model)``.
    res = decode_greedy_autoregressive(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        warmup=False,
        trial_index=0,
    )
    m = res.metrics
    stats: dict[str, Any] = {
        "wall_time_s": m.end_to_end_generation_s,
        "prefill_time_s": m.prefill_time_s,
        "total_decode_time_s": m.total_decode_time_s,
        "max_new_tokens": max_new_tokens,
        "prompt_len_tokens": m.prompt_len_tokens,
        "output_len_tokens": int(res.full_token_ids.shape[-1]),
        "peak_cuda_allocated_bytes": m.peak_cuda_allocated_bytes,
        "logical_kv_cache_bytes": m.logical_kv_cache_bytes,
    }
    return res.text, stats


def reference_greedy_generate_ids(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy ``model.generate`` reference ids on the model device (for unit tests)."""
    device = model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    return out.detach().cpu()
