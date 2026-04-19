"""Replace Llama attention modules with :class:`QuantSpecLlamaAttention` (reversible)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_quant_spec_attention_patched"


def _align_module_device_dtype(dst: nn.Module, src: nn.Module) -> None:
    """Match floating-point parameter dtype (e.g. FP16 model + fresh submodule defaults to FP32 → matmul errors)."""
    ref: torch.Tensor | None = None
    for p in src.parameters(recurse=True):
        if p.is_floating_point():
            ref = p
            break
    if ref is None:
        p0 = next(src.parameters(), None)
        if p0 is not None:
            dst.to(device=p0.device)
        return
    dst.to(device=ref.device, dtype=ref.dtype)


def is_llama_causal_lm(model: nn.Module) -> bool:
    return model.__class__.__name__ == "LlamaForCausalLM" and hasattr(model, "model") and hasattr(model.model, "layers")


def patch_llama_model_with_quant_spec_attention(model: nn.Module) -> nn.Module:
    """Swap each ``layer.self_attn`` with :class:`QuantSpecLlamaAttention` (weights preserved).

    Idempotent: safe to call twice (no-op if already patched).
    """
    from quant_spec_attention.llama_attention import LLAMA_ATTENTION_AVAILABLE, QuantSpecLlamaAttention

    if not LLAMA_ATTENTION_AVAILABLE:
        raise RuntimeError("LlamaAttention not available in this transformers install; cannot patch")

    if getattr(model, _PATCH_ATTR, False):
        return model
    if not is_llama_causal_lm(model):
        raise TypeError(
            f"patch_llama_model_with_quant_spec_attention expects LlamaForCausalLM; got {type(model).__name__}"
        )

    for layer in model.model.layers:
        old = layer.self_attn
        if isinstance(old, QuantSpecLlamaAttention):
            continue
        new = QuantSpecLlamaAttention(old.config, old.layer_idx)
        new.load_state_dict(old.state_dict())
        _align_module_device_dtype(new, old)
        layer.self_attn = new

    setattr(model, _PATCH_ATTR, True)
    logger.info("QuantSpec: patched %d LlamaAttention layers with QuantSpecLlamaAttention", len(model.model.layers))
    return model


def unpatch_llama_model_quant_spec_attention(model: nn.Module) -> nn.Module:
    """Restore original ``LlamaAttention`` classes (requires same transformers version)."""
    if not getattr(model, _PATCH_ATTR, False):
        return model
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers LlamaAttention not importable; cannot unpatch") from exc

    for layer in model.model.layers:
        cur = layer.self_attn
        if not cur.__class__.__name__ == "QuantSpecLlamaAttention":
            continue
        old = LlamaAttention(cur.config, cur.layer_idx)
        old.load_state_dict(cur.state_dict())
        _align_module_device_dtype(old, cur)
        layer.self_attn = old

    setattr(model, _PATCH_ATTR, False)
    logger.info("QuantSpec: restored LlamaAttention modules")
    return model
