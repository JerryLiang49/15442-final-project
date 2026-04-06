"""Minimal Hugging Face loader for Llama-style causal language models."""

from __future__ import annotations

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
    raise ValueError(f"Unsupported torch_dtype={name!r}")


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
    torch_dtype: TorchDtypeName | str = "float16",
    tokenizer_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> LoadedCausalLM:
    """Load ``AutoTokenizer`` and ``AutoModelForCausalLM`` for Llama-like models.

    Uses FP16/BF16 on CUDA by default for memory efficiency. On CPU, forces float32
    regardless of requested dtype (FP16 matmuls are not universally desirable on CPU).

    Args:
        model_name: Hugging Face hub id (e.g. ``meta-llama/Llama-2-7b-hf``).
        device: Resolved torch device.
        torch_dtype: Model parameter dtype for GPU runs.
        tokenizer_kwargs: Extra kwargs forwarded to ``AutoTokenizer.from_pretrained``.
        model_kwargs: Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained``.
    """
    tok_kw: dict[str, Any] = dict(tokenizer_kwargs or {})
    tok_kw.setdefault("use_fast", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _parse_dtype(torch_dtype, device)
    m_kw: dict[str, Any] = dict(model_kwargs or {})

    if device.type == "cuda":
        m_kw.setdefault("torch_dtype", dtype)
        # ``device_map`` expects integer GPU index or string device tags, not ``torch.device``.
        gpu_index = 0 if device.index is None else int(device.index)
        m_kw.setdefault("device_map", {"": gpu_index})
        model = AutoModelForCausalLM.from_pretrained(model_name, **m_kw)
    else:
        # CPU: load in float32 and place explicitly (avoid half precision surprises).
        m_kw.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, **m_kw)
        model.to(device)

    model.eval()
    return LoadedCausalLM(tokenizer=tokenizer, model=model, model_name=model_name)
