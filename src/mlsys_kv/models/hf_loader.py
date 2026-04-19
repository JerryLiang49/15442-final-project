"""Minimal Hugging Face loader for Llama-style causal language models."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


TorchDtypeName = Literal["float16", "bfloat16", "float32"]


def _parse_dtype(name: TorchDtypeName | str, device: torch.device) -> torch.dtype:
    n = str(name).lower()
    if n == "float16":
        return torch.float16
    if n == "bfloat16":
        return torch.bfloat16
    if n == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name!r}")


@dataclass(frozen=True)
class LoadedCausalLM:
    """Tokenizer + causal LM bundled for decoding loops."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    model_name: str

    def to_device(self, device: torch.device) -> None:
        """Move model weights to ``device`` when not using ``device_map``."""
        self.model.to(device)


def load_causal_lm(
    model_name: str,
    *,
    device: torch.device,
    dtype: TorchDtypeName | str = "float16",
    tokenizer_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> LoadedCausalLM:
    """Load ``AutoTokenizer`` and ``AutoModelForCausalLM`` for Llama-like models.

    Uses FP16/BF16 on CUDA by default for memory efficiency. On CPU, forces float32
    regardless of requested dtype (FP16 matmuls are not universally desirable on CPU).

    Args:
        model_name: Hugging Face hub id (e.g. ``meta-llama/Llama-2-7b-hf``).
        device: Resolved torch device.
        dtype: Model parameter dtype for GPU runs (string name; passed to HF as ``dtype=`` / ``torch_dtype=``).
        tokenizer_kwargs: Extra kwargs forwarded to ``AutoTokenizer.from_pretrained``.
        model_kwargs: Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained``.
    """
    # Default Hub timeout is ~10s; slow Wi‑Fi / VPN / HF load → ReadTimeout. Override via env.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    tok_kw: dict[str, Any] = dict(tokenizer_kwargs or {})
    tok_kw.setdefault("use_fast", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_t = _parse_dtype(dtype, device)
    m_kw: dict[str, Any] = dict(model_kwargs or {})
    # Transformers 5.x prefers ``dtype``; older versions use ``torch_dtype``. Passing the wrong
    # name or using ``device_map`` on a single GPU can leave some modules FP32 → matmul Half vs float.
    m_kw.pop("dtype", None)
    m_kw.pop("torch_dtype", None)
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    dtype_kw = "dtype" if "dtype" in sig.parameters else "torch_dtype"

    if device.type == "cuda":
        user_device_map = m_kw.get("device_map")
        # Default: no ``device_map`` so weights load consistently, then ``.to(device, dtype)`` matches
        # all parameters (Accelerate + ``device_map`` often keeps e.g. ``lm_head`` in FP32).
        if user_device_map is None:
            m_kw.pop("device_map", None)
        m_kw[dtype_kw] = dtype_t
        model = AutoModelForCausalLM.from_pretrained(model_name, **m_kw)
        if user_device_map is None:
            model.to(device=device, dtype=dtype_t)
    else:
        # CPU: load in float32 and place explicitly (avoid half precision surprises).
        m_kw.pop("device_map", None)
        m_kw[dtype_kw] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, **m_kw)
        model.to(device)

    model.eval()
    return LoadedCausalLM(tokenizer=tokenizer, model=model, model_name=model_name)
