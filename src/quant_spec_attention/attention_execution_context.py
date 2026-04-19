"""Per-forward attention execution context (Phase I kernel dispatch).

Use :func:`attention_context_scope` around ``model(...)`` calls so :class:`QuantSpecLlamaAttention`
can read ``role``, ``kernel_dispatch``, and the hierarchical :class:`~cache.recent_buffer.RecentBufferManager`.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cache.recent_buffer import RecentBufferManager


class AttentionKernelDispatch(str, Enum):
    """Runtime kernel policy for QuantSpec attention (config / decoder → attention)."""

    HF_REFERENCE = "hf_reference"
    """Standard Hugging Face ``eager_attention_forward`` (SDPA/eager); no Triton hist overlay."""

    TRITON_DRAFT_DECODE = "triton_draft_decode"
    """Use Triton (or ref) Q·K on **packed INT4 history** for draft rounds (upper nibble path)."""

    TRITON_TARGET_VERIFY = "triton_target_verify"
    """Use Triton (or ref) Q·K on packed history for target/verify (upper + lower)."""

    TRITON_FUSED_VERIFIER = "triton_fused_verifier"
    """Fused full verifier attention (upper+lower hist + FP16 tail + block-causal draft block)."""

    AUTO = "auto"
    """Pick draft vs target Triton based on :class:`AttentionRole` when CUDA+Triton available."""


class AttentionRole(str, Enum):
    """Which speculative phase this forward belongs to."""

    DRAFT = "draft"
    TARGET = "target"


class KVLayout(str, Enum):
    DENSE = "dense"
    HIERARCHICAL = "hierarchical"


@dataclass
class AttentionExecutionContext:
    """Thread/async-safe context for one or more layer forwards (via :class:`contextvars.ContextVar`)."""

    role: AttentionRole
    kernel_dispatch: AttentionKernelDispatch
    kv_layout: KVLayout = KVLayout.HIERARCHICAL
    query_length: int = 1
    """Sequence length of ``query_states`` for this forward (1 for single-token decode, γ for verify block)."""

    recent_buffer_manager: Any | None = None
    """Optional :class:`~cache.recent_buffer.RecentBufferManager` (hierarchical KV)."""

    # Debug: last resolved backend per layer (filled by QuantSpecLlamaAttention)
    last_resolved: dict[int, str] = field(default_factory=dict)


_ctx: ContextVar[AttentionExecutionContext | None] = ContextVar("quant_spec_attention_ctx", default=None)


def get_attention_context() -> AttentionExecutionContext | None:
    return _ctx.get()


def set_attention_context(ctx: AttentionExecutionContext | None) -> Any:
    return _ctx.set(ctx)


def reset_attention_context(token: Any) -> None:
    _ctx.reset(token)


@contextmanager
def attention_context_scope(ctx: AttentionExecutionContext):
    """Bind ``ctx`` for the duration of the ``with`` block."""
    tok = set_attention_context(ctx)
    try:
        yield
    finally:
        reset_attention_context(tok)


def resolve_effective_dispatch(
    ctx: AttentionExecutionContext | None,
    *,
    cuda_available: bool,
    triton_available: bool,
) -> AttentionKernelDispatch:
    """Resolve AUTO; return concrete dispatch for logging / branching."""
    if ctx is None:
        return AttentionKernelDispatch.HF_REFERENCE
    kd = ctx.kernel_dispatch
    if kd == AttentionKernelDispatch.HF_REFERENCE:
        return AttentionKernelDispatch.HF_REFERENCE
    if kd == AttentionKernelDispatch.AUTO:
        if not cuda_available or not triton_available:
            return AttentionKernelDispatch.HF_REFERENCE
        return (
            AttentionKernelDispatch.TRITON_DRAFT_DECODE
            if ctx.role == AttentionRole.DRAFT
            else AttentionKernelDispatch.TRITON_TARGET_VERIFY
        )
    if kd in (
        AttentionKernelDispatch.TRITON_DRAFT_DECODE,
        AttentionKernelDispatch.TRITON_TARGET_VERIFY,
        AttentionKernelDispatch.TRITON_FUSED_VERIFIER,
    ):
        if not cuda_available or not triton_available:
            return AttentionKernelDispatch.HF_REFERENCE
        return kd
    return AttentionKernelDispatch.HF_REFERENCE


def qk_backend_string(effective: AttentionKernelDispatch) -> str:
    """``\"triton\"`` or ``\"ref\"`` for :mod:`kv_kernels.triton_attention` dispatch helpers."""
    if effective in (
        AttentionKernelDispatch.TRITON_DRAFT_DECODE,
        AttentionKernelDispatch.TRITON_TARGET_VERIFY,
        AttentionKernelDispatch.TRITON_FUSED_VERIFIER,
    ):
        return "triton"
    return "ref"
