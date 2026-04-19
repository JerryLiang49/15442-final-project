"""Inject QuantSpec kernel dispatch into Llama attention (Phase I)."""

from quant_spec_attention.attention_execution_context import (
    AttentionExecutionContext,
    AttentionKernelDispatch,
    AttentionRole,
    KVLayout,
    attention_context_scope,
    get_attention_context,
    resolve_effective_dispatch,
    set_attention_context,
)
from quant_spec_attention.llama_attention import LLAMA_ATTENTION_AVAILABLE, QuantSpecLlamaAttention
from quant_spec_attention.patch import (
    is_llama_causal_lm,
    patch_llama_model_with_quant_spec_attention,
    unpatch_llama_model_quant_spec_attention,
)

__all__ = [
    "AttentionExecutionContext",
    "AttentionKernelDispatch",
    "AttentionRole",
    "KVLayout",
    "LLAMA_ATTENTION_AVAILABLE",
    "QuantSpecLlamaAttention",
    "attention_context_scope",
    "get_attention_context",
    "is_llama_causal_lm",
    "patch_llama_model_with_quant_spec_attention",
    "resolve_effective_dispatch",
    "set_attention_context",
    "unpatch_llama_model_quant_spec_attention",
]
