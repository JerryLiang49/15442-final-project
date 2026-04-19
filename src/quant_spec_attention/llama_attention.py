"""QuantSpec-wrapped Llama attention: Triton Q·K on packed INT4 history in the real HF forward path.

**Pinned to** Transformers ``LlamaAttention`` API with ``position_embeddings`` (Transformers v5+).

When :class:`~quant_spec_attention.attention_execution_context.AttentionExecutionContext` is set and
``kernel_dispatch`` requests Triton, this module **overwrites** the attention-score block
``[..., :, :hist_len]`` with :mod:`kv_kernels` draft/target Q·K (reference on CPU), then runs the same
eager softmax / dropout / ``attn @ V`` as ``eager_attention_forward`` so the overlay is not discarded.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_spec_attention.attention_execution_context import (
    AttentionKernelDispatch,
    AttentionRole,
    KVLayout,
    get_attention_context,
    qk_backend_string,
    resolve_effective_dispatch,
)

logger = logging.getLogger(__name__)

try:
    from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
except ImportError:  # pragma: no cover
    LlamaAttention = nn.Module  # type: ignore[misc, assignment]
    _LLAMA_AVAILABLE = False
    apply_rotary_pos_emb = None  # type: ignore[misc, assignment]
    repeat_kv = None  # type: ignore[misc, assignment]
else:
    _LLAMA_AVAILABLE = True


def _triton_runtime_ok() -> bool:
    try:
        from kv_kernels.triton_runtime import triton_available

        return bool(triton_available())
    except ImportError:
        return False


def _cat_cf_kv(
    k_cf1: torch.Tensor | None,
    k_cf2: torch.Tensor | None,
    l1: int,
    l2: int,
) -> torch.Tensor:
    """Concatenate CF1 and CF2 along sequence dim (dim 2)."""
    parts: list[torch.Tensor] = []
    if l1 > 0 and k_cf1 is not None:
        parts.append(k_cf1[:, :, :l1, :])
    if l2 > 0 and k_cf2 is not None:
        parts.append(k_cf2[:, :, :l2, :])
    if not parts:
        raise ValueError("cat_cf_kv: empty recent (both lengths zero)")
    return torch.cat(parts, dim=2)


def _try_fused_verifier_attention(
    *,
    query_states: torch.Tensor,
    key_states_rep: torch.Tensor,
    value_states_rep: torch.Tensor,
    ctx: Any,
    layer_idx: int,
    effective: AttentionKernelDispatch,
    attention_mask: torch.Tensor | None,
    output_attentions: bool,
) -> torch.Tensor | None:
    """Return ``[B, H, γ, D]`` fused attention output, or ``None`` to use the HF matmul path."""
    if effective != AttentionKernelDispatch.TRITON_FUSED_VERIFIER:
        return None
    if ctx.role != AttentionRole.TARGET:
        return None
    if query_states.shape[2] <= 1:
        return None
    if query_states.shape[0] != 1:
        return None
    if output_attentions:
        return None
    if attention_mask is not None:
        return None

    mgr = ctx.recent_buffer_manager
    if mgr is None:
        return None
    store = mgr.store
    hist_len = int(store._hist_len)
    if hist_len <= 0:
        return None
    s_rec = int(store._cf1_len) + int(store._cf2_len)
    gamma = int(query_states.shape[2])
    seq_len = int(key_states_rep.shape[2])
    if seq_len != hist_len + s_rec + gamma:
        logger.warning(
            "QuantSpecLlamaAttention: fused verifier skipped (seq_len %s != hist+recent+gamma %s)",
            seq_len,
            hist_len + s_rec + gamma,
        )
        return None

    from kv_kernels.fused_verifier_block_attention import fused_verifier_block_attention
    from kv_kernels.triton_runtime import triton_available as _tri

    gs = int(store.quant_group_size)
    be = "triton" if query_states.device.type == "cuda" and _tri() else "ref"

    k_cf1 = store._cf1_k[layer_idx]
    k_cf2 = store._cf2_k[layer_idx]
    v_cf1 = store._cf1_v[layer_idx]
    v_cf2 = store._cf2_v[layer_idx]
    l1 = int(store._cf1_len)
    l2 = int(store._cf2_len)
    if (l1 > 0 and k_cf1 is None) or (l2 > 0 and k_cf2 is None):
        return None

    if s_rec > 0:
        k_recent = _cat_cf_kv(k_cf1, k_cf2, l1, l2)
        v_recent = _cat_cf_kv(v_cf1, v_cf2, l1, l2)
        k_recent = k_recent[0].contiguous()
        v_recent = v_recent[0].contiguous()
    else:
        d = int(query_states.shape[3])
        h = int(query_states.shape[1])
        dev = query_states.device
        k_recent = torch.zeros(h, 0, d, device=dev, dtype=key_states_rep.dtype)
        v_recent = torch.zeros(h, 0, d, device=dev, dtype=value_states_rep.dtype)

    ko = hist_len
    k_block = key_states_rep[0, :, ko + s_rec : ko + s_rec + gamma, :].contiguous()
    v_block = value_states_rep[0, :, ko + s_rec : ko + s_rec + gamma, :].contiguous()

    k_uq = store._upper_k[layer_idx][0, :, :hist_len, :].contiguous()
    k_lq = store._lower_k[layer_idx][0, :, :hist_len, :].contiguous()
    k_su = store._upper_k_scale[layer_idx][0, :, :hist_len, :].contiguous()
    k_zu = store._upper_k_zp[layer_idx][0, :, :hist_len, :].contiguous()
    k_sl = store._lower_k_scale[layer_idx][0, :, :hist_len, :].contiguous()
    k_zl = store._lower_k_zp[layer_idx][0, :, :hist_len, :].contiguous()

    v_uq = store._upper_v[layer_idx][0, :, :hist_len, :].contiguous()
    v_lq = store._lower_v[layer_idx][0, :, :hist_len, :].contiguous()
    v_su = store._upper_v_scale[layer_idx][0, :, :, :].contiguous()
    v_zu = store._upper_v_zp[layer_idx][0, :, :, :].contiguous()
    v_sl = store._lower_v_scale[layer_idx][0, :, :, :].contiguous()
    v_zl = store._lower_v_zp[layer_idx][0, :, :, :].contiguous()

    q = query_states.contiguous()

    return fused_verifier_block_attention(
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
        k_recent,
        v_recent,
        k_block,
        v_block,
        group_size_k=gs,
        group_size_v=gs,
        backend=be,
    )


def _apply_hist_qk_overlay(
    attn_weights: torch.Tensor,
    query_states: torch.Tensor,
    *,
    ctx: Any,
    layer_idx: int,
    scaling: float,
    effective: AttentionKernelDispatch,
    use_target_kernel: bool,
    num_kv_groups: int,
) -> None:
    """In-place: replace ``[..., :, :hist_len]`` scores with packed INT4 Q·K (Triton or ref)."""
    from kv_kernels.triton_attention import qk_draft_dispatch, qk_target_dispatch

    mgr = ctx.recent_buffer_manager
    if mgr is None:
        return
    store = mgr.store
    hist_len = int(store._hist_len)
    if hist_len <= 0:
        return

    gs = int(store.quant_group_size)
    be = qk_backend_string(effective)
    bsz, num_heads, q_len, _kv_len = attn_weights.shape
    if bsz != 1:
        logger.warning(
            "QuantSpecLlamaAttention: batch>1 unsupported for Triton hist overlay; skipping hist patch"
        )
        return

    n_rep = max(int(num_kv_groups), 1)

    upper_k = store._upper_k[layer_idx]
    su = store._upper_k_scale[layer_idx]
    zu = store._upper_k_zp[layer_idx]
    sl = store._lower_k_scale[layer_idx]
    zl = store._lower_k_zp[layer_idx]

    for h in range(num_heads):
        kv_h = h // n_rep
        for qpos in range(q_len):
            qvec = query_states[0, h, qpos, :].contiguous()
            packed = upper_k[0, kv_h, :hist_len, :].contiguous()
            ksu = su[0, kv_h, :hist_len, :].contiguous()
            kzu = zu[0, kv_h, :hist_len, :].contiguous()
            if use_target_kernel:
                scores = qk_target_dispatch(
                    qvec,
                    packed,
                    ksu,
                    kzu,
                    sl[0, kv_h, :hist_len, :].contiguous(),
                    zl[0, kv_h, :hist_len, :].contiguous(),
                    group_size=gs,
                    backend=be,
                )
            else:
                scores = qk_draft_dispatch(
                    qvec,
                    packed,
                    ksu,
                    kzu,
                    group_size=gs,
                    backend=be,
                )
            attn_weights[0, h, qpos, :hist_len] = (scores * scaling).to(attn_weights.dtype)


if _LLAMA_AVAILABLE:

    class QuantSpecLlamaAttention(LlamaAttention):
        """Drop-in replacement for ``LlamaAttention`` with optional Triton hist Q·K overlay."""

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            attention_mask: torch.Tensor | None = None,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ctx = get_attention_context()
            cuda_ok = hidden_states.device.type == "cuda" and torch.cuda.is_available()
            tri_ok = _triton_runtime_ok()
            effective = resolve_effective_dispatch(ctx, cuda_available=cuda_ok, triton_available=tri_ok)

            if (
                ctx is None
                or ctx.kv_layout != KVLayout.HIERARCHICAL
                or effective == AttentionKernelDispatch.HF_REFERENCE
                or ctx.recent_buffer_manager is None
            ):
                return super().forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            hist_len = int(ctx.recent_buffer_manager.store._hist_len)
            if hist_len <= 0:
                return super().forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings  # type: ignore[misc]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
            value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

            output_attentions = bool(kwargs.get("output_attentions", False))
            fused_out = _try_fused_verifier_attention(
                query_states=query_states,
                key_states_rep=key_states_rep,
                value_states_rep=value_states_rep,
                ctx=ctx,
                layer_idx=self.layer_idx,
                effective=effective,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if fused_out is not None:
                attn_output = fused_out.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                ctx.last_resolved[self.layer_idx] = f"{effective.value}|fused_verifier"
                logger.info(
                    "QuantSpecLlamaAttention layer=%s fused_verifier q_len=%s hist_len=%s",
                    self.layer_idx,
                    int(query_states.shape[2]),
                    hist_len,
                )
                return attn_output, None

            attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * self.scaling

            use_target = effective == AttentionKernelDispatch.TRITON_TARGET_VERIFY

            try:
                _apply_hist_qk_overlay(
                    attn_weights,
                    query_states,
                    ctx=ctx,
                    layer_idx=self.layer_idx,
                    scaling=self.scaling,
                    effective=effective,
                    use_target_kernel=use_target,
                    num_kv_groups=self.num_key_value_groups,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "QuantSpecLlamaAttention layer %s: hist Q·K overlay failed (%s); using matmul scores",
                    self.layer_idx,
                    exc,
                )

            ctx.last_resolved[self.layer_idx] = (
                f"{effective.value}|{'target' if use_target else 'draft'}|{qk_backend_string(effective)}"
            )
            logger.info(
                "QuantSpecLlamaAttention layer=%s backend=%s role=%s q_len=%s hist_len=%s",
                self.layer_idx,
                ctx.last_resolved[self.layer_idx],
                ctx.role.value,
                int(query_states.shape[2]),
                hist_len,
            )

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            dropout_p = 0.0 if not self.training else self.attention_dropout
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states_rep)
            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights


else:  # pragma: no cover

    class QuantSpecLlamaAttention(nn.Module):  # type: ignore[no-redef]
        """Placeholder when Transformers Llama is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "QuantSpecLlamaAttention requires transformers with LlamaAttention (install transformers>=4.40)"
            )


LLAMA_ATTENTION_AVAILABLE = _LLAMA_AVAILABLE
