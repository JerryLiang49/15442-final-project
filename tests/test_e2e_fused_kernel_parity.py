"""Phase M — end-to-end parity: HF reference attention vs Triton fused verifier (tiny Llama).

**Determinism**

* Two models initialized with the **same** RNG seed before construction so weights match.
* Greedy decoding is deterministic given identical logits.

**Tolerances**

* Token sequences: **exact** equality (``torch.equal``) — if kernel matches reference within
  layerwise bounds, greedy paths should agree.

**Requirements**

* CUDA + Triton for the fused path; skipped on CPU-only CI.
* Long enough prompt vs ``recent_tokens_cap`` so ``hist_len > 0`` and the fused verifier can run on
  target verification (γ>1).
"""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2TokenizerFast, LlamaConfig, LlamaForCausalLM

from decoding.speculative_dense_hierarchical import SpeculativeDecoderDenseHierarchical
from kv_kernels.triton_runtime import triton_available
from quant_spec_attention.attention_execution_context import AttentionKernelDispatch


def _tiny_llama(device: torch.device) -> tuple[LlamaForCausalLM, GPT2TokenizerFast]:
    """Vocab aligned with GPT-2 tokenizer (50257) for integration tests without extra downloads."""
    cfg = LlamaConfig(
        vocab_size=50257,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        hidden_act="silu",
    )
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    m = LlamaForCausalLM(cfg)
    m.to(device)
    m.eval()
    return m, tok


def _run_hierarchical_decode(
    *,
    attention_kernel_dispatch: AttentionKernelDispatch,
    device: torch.device,
    seed: int,
    prompt: str,
    max_new_tokens: int,
    gamma: int,
    g: int,
    recent_tokens_cap: int,
) -> tuple[torch.Tensor, object, tuple]:
    torch.manual_seed(seed)
    model, tok = _tiny_llama(device)
    dec = SpeculativeDecoderDenseHierarchical(
        model,
        tok,
        gamma=gamma,
        G=g,
        quant_group_size=8,
        verify_match_autoregressive=False,
        recent_tokens_cap=recent_tokens_cap,
        kv_kernel_backend="triton",
        apply_quant_spec_attention_patch=True,
        attention_kernel_dispatch=attention_kernel_dispatch,
    )
    out = dec.decode(prompt, max_new_tokens=max_new_tokens)
    return out.full_token_ids.cpu(), out.metrics, out.round_history


@pytest.mark.parity_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton fused e2e")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
def test_e2e_llama_tokens_and_acceptance_hf_vs_fused_verifier() -> None:
    """Logits path parity via identical greedy tokens + acceptance/rollback structure."""
    device = torch.device("cuda:0")
    # Force INT4 history (hist_len = prompt_len - recent_cap) > 0 for fused verifier eligibility.
    prompt = "The quick brown fox jumps over the lazy dog. " * 3
    max_new = 24
    gamma = 4
    g = 8
    recent_cap = 6

    seed = 2026
    ids_ref, met_ref, rounds_ref = _run_hierarchical_decode(
        attention_kernel_dispatch=AttentionKernelDispatch.HF_REFERENCE,
        device=device,
        seed=seed,
        prompt=prompt,
        max_new_tokens=max_new,
        gamma=gamma,
        g=g,
        recent_tokens_cap=recent_cap,
    )
    ids_fused, met_fused, rounds_fused = _run_hierarchical_decode(
        attention_kernel_dispatch=AttentionKernelDispatch.TRITON_FUSED_VERIFIER,
        device=device,
        seed=seed,
        prompt=prompt,
        max_new_tokens=max_new,
        gamma=gamma,
        g=g,
        recent_tokens_cap=recent_cap,
    )

    assert torch.equal(ids_ref, ids_fused), (
        f"token mismatch: ref={ids_ref.tolist()[:32]}... fused={ids_fused.tolist()[:32]}..."
    )
    assert met_ref.total_draft_proposals == met_fused.total_draft_proposals
    assert met_ref.total_accepted_draft_tokens == met_fused.total_accepted_draft_tokens
    assert met_ref.rejection_events == met_fused.rejection_events
    assert len(rounds_ref) == len(rounds_fused)
    for a, b in zip(rounds_ref, rounds_fused):
        assert a.rejected == b.rejected
        assert a.accepted_prefix_len == b.accepted_prefix_len
        assert a.gamma == b.gamma


@pytest.mark.parity_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not triton_available(), reason="Triton required")
def test_e2e_llama_matches_autoregressive_both_dispatch_modes() -> None:
    """Both dispatch modes still match greedy AR (sanity)."""
    device = torch.device("cuda:0")
    prompt = "Hello world " * 4
    max_new = 12
    for dispatch in (
        AttentionKernelDispatch.HF_REFERENCE,
        AttentionKernelDispatch.TRITON_FUSED_VERIFIER,
    ):
        torch.manual_seed(42)
        model, tok = _tiny_llama(device)
        dec = SpeculativeDecoderDenseHierarchical(
            model,
            tok,
            gamma=3,
            G=8,
            quant_group_size=8,
            verify_match_autoregressive=True,
            recent_tokens_cap=8,
            kv_kernel_backend="triton",
            apply_quant_spec_attention_patch=True,
            attention_kernel_dispatch=dispatch,
        )
        dec.decode(prompt, max_new_tokens=max_new)
