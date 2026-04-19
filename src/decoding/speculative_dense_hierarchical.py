"""Dense self-speculative decoding with hierarchical KV + double FP16 recent buffer (QuantSpec-style).

**Control flow (one round)**

1. Snapshot **verifier** ``past_key_values`` from :meth:`~cache.recent_buffer.RecentBufferManager.target_view_without_cf2`
   (sequence length ``S``; CF2 empty at round start).
2. **Draft:** ``γ`` greedy singles; each forward uses :meth:`~cache.recent_buffer.RecentBufferManager.draft_view`
   (upper-only history + FP16 CF1 + growing CF2). Append new K/V slices into **CF2** only.
3. **Verify:** One block forward with **same** snapshot past + proposal block (full-precision target path
   would use :meth:`~cache.recent_buffer.RecentBufferManager.target_view_without_cf2` — identical tensors
   to draft on CF1/hist when history is empty; differs when INT4 history is non-empty).
4. **Accept / correct:** Same greedy longest-prefix + correction as :mod:`decoding.speculative_dense`.
5. **Commit:** Hugging Face ``past_key_values`` from the verifier is authoritative; the hierarchical store
   is **resynced** via :meth:`~cache.recent_buffer.RecentBufferManager.prefill_initialize` so CF2 draft
   state is replaced (rollback is implicit; explicit CF2 trim is unnecessary once HF past is synced).

This matches Phase A semantics while using draft vs target *views* for draft vs verify forwards. Rollback of
a rejected speculative suffix is represented by **not** retaining draft CF2 after commit — the next
``prefill_initialize`` rebuilds hist / CF1 / empty CF2 from the verifier cache.

**Why this differs from :class:`SpeculativeDecoderDense`**

* Phase A keeps a single FP16 HF cache. Here, draft steps read **draft_view** (cheap upper dequant on history)
  and write FP16 into **CF2**; verification uses a snapshot **without** CF2, then state is reconciled from HF.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from benchmarks.timer import timed_cuda_interval
from cache.hf_kv_clone import clone_past_key_values
from cache.hf_past_adapters import (
    hf_past_to_layer_lists,
    hierarchical_view_to_past_key_values,
)
from cache.hf_kv_trim import past_contains_sliding_window_layer
from cache.hierarchical_kv_store import HierarchicalKVStore
from cache.recent_buffer import RecentBufferManager

from .autoregressive import decode_greedy_autoregressive, model_device
from .speculative_dense import (
    DenseSpeculativeMetrics,
    concat_committed,
    verify_block_and_commit,
)

from kv_kernels.backend import KVKernelBackend, normalize_backend
from kv_kernels.integration import validate_qk_kernels_cuda
from kv_kernels.triton_runtime import triton_available
from quant_spec_attention.attention_execution_context import AttentionRole

if TYPE_CHECKING:
    from quant_spec_attention.attention_execution_context import AttentionKernelDispatch
    from serving.block_kv import BlockPoolConfig

logger = logging.getLogger(__name__)


def _attention_dispatch_from_kv_backend(kv: KVKernelBackend) -> Any:
    """Map legacy ``kv_kernel_backend`` to Phase I :class:`~quant_spec_attention.attention_execution_context.AttentionKernelDispatch`.

    **Triton** uses :attr:`~quant_spec_attention.attention_execution_context.AttentionKernelDispatch.TRITON_FUSED_VERIFIER`
    so verify blocks run the fused attention kernel. Draft steps still skip fusion (``role != target`` in
    :func:`quant_spec_attention.llama_attention._try_fused_verifier_attention`) and use matmul + draft Q·K overlay.
    Previously this returned ``AUTO``, which resolved verify to ``TRITON_TARGET_VERIFY`` only — never invoking fusion.
    """
    from quant_spec_attention.attention_execution_context import AttentionKernelDispatch

    if kv == KVKernelBackend.REFERENCE:
        return AttentionKernelDispatch.HF_REFERENCE
    return AttentionKernelDispatch.TRITON_FUSED_VERIFIER


def _infer_num_layers(model: PreTrainedModel) -> int:
    c = model.config
    n = getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", None)
    if n is None:
        raise ValueError("Cannot infer num_hidden_layers from model.config")
    return int(n)


def _infer_heads_and_head_dim(model: PreTrainedModel) -> tuple[int, int]:
    c = model.config
    n_head = getattr(c, "num_attention_heads", None) or getattr(c, "n_head", None)
    n_embd = getattr(c, "hidden_size", None) or getattr(c, "n_embd", None) or getattr(c, "n_embed", None)
    if n_head is None or n_embd is None:
        raise ValueError("Cannot infer attention heads / hidden size from model.config")
    n_head = int(n_head)
    n_embd = int(n_embd)
    if n_embd % n_head != 0:
        raise ValueError(f"hidden_size {n_embd} not divisible by num_heads {n_head}")
    return n_head, n_embd // n_head


def sync_hierarchical_store_from_hf_past(
    mgr: RecentBufferManager,
    past_key_values: Any,
    *,
    recent_tokens_cap: int | None = None,
) -> None:
    """Rebuild INT4 history + CF1 (+ empty CF2) from authoritative verifier ``past_key_values``."""
    layers_k, layers_v = hf_past_to_layer_lists(past_key_values)
    cap = recent_tokens_cap if recent_tokens_cap is not None else mgr.store.cf1_max_tokens
    mgr.prefill_initialize(layers_k, layers_v, recent_tokens_cap=cap)


def incremental_commit_from_verifier(
    mgr: RecentBufferManager,
    new_past: Any,
    *,
    n_committed: int,
    cf1_len_at_round_start: int,
) -> None:
    """Incrementally commit verified tokens into the hierarchical store.

    Instead of rebuilding the entire store from scratch, this extracts only the newly
    committed K/V slices from ``new_past`` and appends them to CF1, with rollover
    to INT4 history when CF1 exceeds capacity.

    Args:
        new_past: Verifier ``past_key_values`` after the round. With CF1-only past, this has
            ``cf1_len + n_committed`` tokens (no dequantized history prefix).
        n_committed: Number of newly committed tokens this round.
        cf1_len_at_round_start: The CF1 length before the round started. The new tokens
            in ``new_past`` start at this offset.
    """
    store = mgr.store

    mgr.clear_speculative()

    layers_k, layers_v = hf_past_to_layer_lists(new_past)
    start = cf1_len_at_round_start
    new_k = [k[:, :, start:start + n_committed, :] for k in layers_k]
    new_v = [v[:, :, start:start + n_committed, :] for v in layers_v]

    mgr.append_draft(new_k, new_v)
    mgr.accept_verified_prefix(n_committed)

    if store.cf1_len > store.cf1_max_tokens:
        mgr.rollover()


def _build_cf1_only_dynamic_cache(
    store: HierarchicalKVStore,
    *,
    config: Any = None,
) -> DynamicCache:
    """Build a DynamicCache containing only CF1 FP16 KV (no dequantized history)."""
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    cf1_len = store.cf1_len
    for i in range(store.num_layers):
        if cf1_len > 0 and store._cf1_k[i] is not None:
            k = store._cf1_k[i][:, :, :cf1_len, :].contiguous()
            v = store._cf1_v[i][:, :, :cf1_len, :].contiguous()
        else:
            k = torch.empty(
                store.batch_size, store.num_heads, 0, store.head_dim,
                device=store.device, dtype=store.dtype,
            )
            v = torch.empty(
                store.batch_size, store.num_heads, 0, store.head_dim,
                device=store.device, dtype=store.dtype,
            )
        pairs.append((k, v))
    return DynamicCache(ddp_cache_data=tuple(pairs), config=config)


def draft_greedy_proposals_hierarchical(
    model: PreTrainedModel,
    mgr: RecentBufferManager,
    *,
    start_logits: torch.Tensor,
    gamma: int,
) -> list[torch.Tensor]:
    """γ sequential greedy drafts using fused kernel on INT4 history + CF1-only DynamicCache.

    The DynamicCache holds only CF1 FP16 KV (no dequantized history). The patched
    ``QuantSpecLlamaAttention`` dispatches to the fused draft decode kernel which reads
    INT4 history directly from the hierarchical store and FP16 recent from the DynamicCache.
    Explicit ``position_ids`` ensure correct RoPE despite the shortened cache.
    """
    proposals: list[torch.Tensor] = []
    logits = start_logits
    store = mgr.store
    cf1_cache = _build_cf1_only_dynamic_cache(store, config=model.config)
    logical_offset = store.hist_len + store.cf1_len
    device = start_logits.device

    with torch.inference_mode():
        for step in range(gamma):
            next_tok = logits.argmax(dim=-1, keepdim=True)
            proposals.append(next_tok)
            pos_ids = torch.tensor([[logical_offset + step]], device=device, dtype=torch.long)
            out = model(
                input_ids=next_tok,
                past_key_values=cf1_cache,
                position_ids=pos_ids,
                use_cache=True,
            )
            cf1_cache = out.past_key_values
            logits = out.logits[:, -1, :]

    cf1_len = store.cf1_len
    draft_k_layers: list[torch.Tensor] = []
    draft_v_layers: list[torch.Tensor] = []
    pairs = list(zip(cf1_cache.key_cache, cf1_cache.value_cache))
    for k, v in pairs:
        draft_k_layers.append(k[:, :, cf1_len:, :].contiguous())
        draft_v_layers.append(v[:, :, cf1_len:, :].contiguous())
    mgr.append_draft(draft_k_layers, draft_v_layers)

    return proposals


def _draft_greedy_proposals_dequant_fallback(
    model: PreTrainedModel,
    mgr: RecentBufferManager,
    *,
    start_logits: torch.Tensor,
    gamma: int,
) -> list[torch.Tensor]:
    """Fallback draft path: dequantizes history into past_key_values (for non-patched models)."""
    from cache.hf_past_adapters import extract_last_token_kv_per_layer

    proposals: list[torch.Tensor] = []
    logits = start_logits
    cfg = model.config
    with torch.inference_mode():
        for _ in range(gamma):
            next_tok = logits.argmax(dim=-1, keepdim=True)
            proposals.append(next_tok)
            past_draft = hierarchical_view_to_past_key_values(mgr.draft_view(), config=cfg)
            out = model(input_ids=next_tok, past_key_values=past_draft, use_cache=True)
            nk, nv = extract_last_token_kv_per_layer(out.past_key_values)
            mgr.append_draft(nk, nv)
            logits = out.logits[:, -1, :]
    return proposals


@dataclass
class HierarchicalRoundDebug:
    """Inspectable state for one speculative round."""

    hist_len: int
    cf1_len: int
    cf2_len_after_draft: int
    accepted_prefix_len: int
    rejected: bool
    gamma: int


@dataclass(frozen=True)
class SpeculativeHierarchicalDecodeResult:
    """Output of :class:`SpeculativeDecoderDenseHierarchical`."""

    text: str
    prompt_token_ids: torch.Tensor
    new_token_ids: torch.Tensor
    full_token_ids: torch.Tensor
    metrics: DenseSpeculativeMetrics
    mgr: RecentBufferManager
    last_round: HierarchicalRoundDebug | None
    round_history: tuple[HierarchicalRoundDebug, ...]
    """Per-round debug snapshots (small runs)."""


class SpeculativeDecoderDenseHierarchical:
    """Self-speculative greedy decoding with hierarchical store + CF1/CF2 (algorithmic prototype).

    **KV kernel hook (Phase F / I)**

    * ``kv_kernel_backend`` — ``reference`` vs ``triton`` microbench validation (:func:`kv_kernels.integration.validate_qk_kernels_cuda`).
    * ``attention_kernel_dispatch`` — Phase I dispatch inside Llama attention (``hf_reference``, ``auto``,
      ``triton_target_verify``, ``triton_fused_verifier``, …).
      When ``apply_quant_spec_attention_patch=True`` and the model is ``LlamaForCausalLM``, layers use
      :class:`quant_spec_attention.llama_attention.QuantSpecLlamaAttention` so packed-history Q·K can run
      on the Triton path during draft vs target forwards (see ``attention_context_scope`` in :meth:`decode`).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        gamma: int,
        *,
        G: int = 64,
        quant_group_size: int | None = None,
        verbose: bool = False,
        debug: bool = False,
        verify_match_autoregressive: bool = True,
        recent_tokens_cap: int | None = None,
        kv_kernel_backend: str = "reference",
        validate_triton_kernels_at_start: bool = False,
        apply_quant_spec_attention_patch: bool = False,
        attention_kernel_dispatch: Any = None,
        serving_mode: bool = False,
        legacy_double_clone_verifier: bool = False,
        block_pool_config: BlockPoolConfig | None = None,
    ) -> None:
        if gamma < 1:
            raise ValueError("gamma must be >= 1")
        self.model = model
        self.tokenizer = tokenizer
        self.gamma = int(gamma)
        self.G = int(G)
        self.quant_group_size = quant_group_size
        self.verbose = verbose
        self.debug = debug
        self.verify_match_autoregressive = verify_match_autoregressive
        self.serving_mode = bool(serving_mode)
        self.legacy_double_clone_verifier = bool(legacy_double_clone_verifier)
        self._block_pool_config = block_pool_config
        self._recent_cap_override = recent_tokens_cap
        self.kv_kernel_backend: KVKernelBackend = normalize_backend(kv_kernel_backend)
        self._validate_triton_kernels_at_start = bool(validate_triton_kernels_at_start)

        from quant_spec_attention.attention_execution_context import AttentionKernelDispatch
        from quant_spec_attention.patch import is_llama_causal_lm, patch_llama_model_with_quant_spec_attention

        if attention_kernel_dispatch is None:
            self._attention_kernel_dispatch: Any = _attention_dispatch_from_kv_backend(self.kv_kernel_backend)
        else:
            self._attention_kernel_dispatch = (
                AttentionKernelDispatch(attention_kernel_dispatch)
                if isinstance(attention_kernel_dispatch, str)
                else attention_kernel_dispatch
            )

        self._model_patched = False
        if apply_quant_spec_attention_patch:
            if is_llama_causal_lm(model):
                patch_llama_model_with_quant_spec_attention(model)
                self._model_patched = True
            else:
                logger.warning(
                    "apply_quant_spec_attention_patch=True but model is not LlamaForCausalLM; skipping attention patch"
                )

        n_layer = _infer_num_layers(model)
        n_heads, head_dim = _infer_heads_and_head_dim(model)
        device = model_device(model)
        self._n_layer = n_layer
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._device = device

        store = HierarchicalKVStore(
            num_layers=n_layer,
            num_heads=n_heads,
            head_dim=head_dim,
            batch_size=1,
            G=self.G,
            quant_group_size=quant_group_size,
            device=device,
            dtype=torch.float16,
        )
        self.mgr = RecentBufferManager(store)

    def _recent_cap(self) -> int | None:
        return self._recent_cap_override

    def _attention_scope(self, *, role: Any, query_length: int) -> Any:
        """Phase I: bind :class:`~quant_spec_attention.attention_execution_context.AttentionExecutionContext` for Llama forwards."""
        from quant_spec_attention.attention_execution_context import (
            AttentionExecutionContext,
            AttentionKernelDispatch,
            KVLayout,
            attention_context_scope,
        )

        base_dispatch = self._attention_kernel_dispatch
        if base_dispatch not in (
            AttentionKernelDispatch.HF_REFERENCE,
        ):
            if role == AttentionRole.DRAFT:
                dispatch = AttentionKernelDispatch.TRITON_DRAFT_DECODE
            else:
                dispatch = AttentionKernelDispatch.TRITON_FUSED_VERIFIER
        else:
            dispatch = base_dispatch

        return attention_context_scope(
            AttentionExecutionContext(
                role=role,
                kernel_dispatch=dispatch,
                kv_layout=KVLayout.HIERARCHICAL,
                query_length=int(query_length),
                recent_buffer_manager=self.mgr,
            )
        )

    def decode(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        benchmark_profile: bool = False,
    ) -> SpeculativeHierarchicalDecodeResult:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        device = self._device
        t0 = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids: torch.Tensor = inputs["input_ids"].to(device)
        full_ids = prompt_ids

        total_draft_proposals = 0
        total_draft_accepted = 0
        rejection_events = 0
        rounds = 0
        new_generated = 0

        round_debug_list: list[HierarchicalRoundDebug] = []
        last_snap: HierarchicalRoundDebug | None = None

        prefill_s = 0.0
        draft_sum = 0.0
        verify_sum = 0.0
        resync_sum = 0.0

        req_state = None
        if self.serving_mode:
            from serving.request_state import reset_request_state

            req_state = reset_request_state(
                request_id="hier-0",
                prompt_len=int(prompt_ids.shape[1]),
                max_new_tokens=max_new_tokens,
                gamma=self.gamma,
                device=device,
            )

        with torch.inference_mode():
            if (
                self.kv_kernel_backend == KVKernelBackend.TRITON
                and self._validate_triton_kernels_at_start
            ):
                if torch.cuda.is_available() and triton_available():
                    validate_qk_kernels_cuda()
                else:
                    logger.warning(
                        "kv_kernel_backend=triton but CUDA or Triton unavailable; "
                        "skipping kernel validation (decoder still uses HF attention)."
                    )

            if benchmark_profile:
                with timed_cuda_interval(device) as pre_slot:
                    prefill_out = self.model(input_ids=prompt_ids, use_cache=True, past_key_values=None)
                    last_logits = prefill_out.logits[:, -1, :].clone()
                    sync_hierarchical_store_from_hf_past(
                        self.mgr,
                        prefill_out.past_key_values,
                        recent_tokens_cap=self._recent_cap(),
                    )
                prefill_s = float(pre_slot[0])
            else:
                prefill_out = self.model(input_ids=prompt_ids, use_cache=True, past_key_values=None)
                last_logits = prefill_out.logits[:, -1, :].clone()
                sync_hierarchical_store_from_hf_past(
                    self.mgr,
                    prefill_out.past_key_values,
                    recent_tokens_cap=self._recent_cap(),
                )

            while new_generated < max_new_tokens:
                remaining = max_new_tokens - new_generated
                gamma_eff = min(self.gamma, remaining)
                rounds += 1

                store = self.mgr.store
                cf1_len_at_start = store.cf1_len
                hist_len_for_offset = store.hist_len
                use_fused = self._model_patched

                logits_snap = last_logits.clone()

                if use_fused:
                    past_for_verify = _build_cf1_only_dynamic_cache(
                        store, config=self.model.config,
                    )
                    draft_fn = draft_greedy_proposals_hierarchical
                else:
                    view_past = hierarchical_view_to_past_key_values(
                        self.mgr.target_view_without_cf2(),
                        config=self.model.config,
                    )
                    past_for_verify = view_past
                    draft_fn = _draft_greedy_proposals_dequant_fallback

                if benchmark_profile:
                    with self._attention_scope(role=AttentionRole.DRAFT, query_length=1):
                        with timed_cuda_interval(device) as d_slot:
                            proposals = draft_fn(
                                self.model,
                                self.mgr,
                                start_logits=logits_snap,
                                gamma=gamma_eff,
                            )
                    draft_sum += float(d_slot[0])
                else:
                    with self._attention_scope(role=AttentionRole.DRAFT, query_length=1):
                        proposals = draft_fn(
                            self.model,
                            self.mgr,
                            start_logits=logits_snap,
                            gamma=gamma_eff,
                        )
                total_draft_proposals += len(proposals)

                cf2_after_draft = self.mgr.store.cf2_len

                verify_offset = hist_len_for_offset if use_fused else 0
                if benchmark_profile:
                    with self._attention_scope(role=AttentionRole.TARGET, query_length=gamma_eff):
                        with timed_cuda_interval(device) as v_slot:
                            committed, new_past, new_logits, n_accepted_draft, rejected = verify_block_and_commit(
                                self.model,
                                verifier_past_at_round_start=past_for_verify,
                                start_logits=logits_snap,
                                proposals=proposals,
                                debug=self.debug,
                                logical_seq_offset=verify_offset,
                            )
                    verify_sum += float(v_slot[0])
                else:
                    with self._attention_scope(role=AttentionRole.TARGET, query_length=gamma_eff):
                        committed, new_past, new_logits, n_accepted_draft, rejected = verify_block_and_commit(
                            self.model,
                            verifier_past_at_round_start=past_for_verify,
                            start_logits=logits_snap,
                            proposals=proposals,
                            debug=self.debug,
                            logical_seq_offset=verify_offset,
                        )
                total_draft_accepted += n_accepted_draft
                if rejected:
                    rejection_events += 1

                if self.verbose:
                    prop_ids = [int(p[0, 0].item()) for p in proposals]
                    com_ids = [int(c[0, 0].item()) for c in committed]
                    print(
                        f"[spec_dense_hier] round={rounds} gamma={gamma_eff} proposals={prop_ids} "
                        f"accepted_draft={n_accepted_draft} rejected={rejected} committed={com_ids}"
                    )

                n_committed = len(committed)
                if use_fused:
                    if benchmark_profile:
                        with timed_cuda_interval(device) as rs_slot:
                            incremental_commit_from_verifier(
                                self.mgr,
                                new_past,
                                n_committed=n_committed,
                                cf1_len_at_round_start=cf1_len_at_start,
                            )
                        resync_sum += float(rs_slot[0])
                    else:
                        incremental_commit_from_verifier(
                            self.mgr,
                            new_past,
                            n_committed=n_committed,
                            cf1_len_at_round_start=cf1_len_at_start,
                        )
                else:
                    if benchmark_profile:
                        with timed_cuda_interval(device) as rs_slot:
                            self.mgr.clear_speculative()
                            sync_hierarchical_store_from_hf_past(
                                self.mgr,
                                new_past,
                                recent_tokens_cap=self._recent_cap(),
                            )
                        resync_sum += float(rs_slot[0])
                    else:
                        self.mgr.clear_speculative()
                        sync_hierarchical_store_from_hf_past(
                            self.mgr,
                            new_past,
                            recent_tokens_cap=self._recent_cap(),
                        )

                s = self.mgr.store
                snap = HierarchicalRoundDebug(
                    hist_len=s.hist_len,
                    cf1_len=s.cf1_len,
                    cf2_len_after_draft=cf2_after_draft,
                    accepted_prefix_len=int(n_accepted_draft),
                    rejected=bool(rejected),
                    gamma=int(gamma_eff),
                )
                round_debug_list.append(snap)
                last_snap = snap

                full_ids = concat_committed(full_ids, committed)
                if self.debug and not past_contains_sliding_window_layer(new_past):
                    from cache.hf_kv_trim import verifier_cache_seq_len_hf

                    tok_len = int(full_ids.shape[1])
                    offset = hist_len_for_offset if use_fused else 0
                    kv_len = verifier_cache_seq_len_hf(new_past) + offset
                    if kv_len != tok_len:
                        raise AssertionError(
                            f"[spec_dense_hier debug] seq len {tok_len} != kv len {kv_len} "
                            f"(hist_offset={offset})"
                        )
                last_logits = new_logits.clone()
                new_generated += len(committed)

                if req_state is not None:
                    from serving.request_state import update_request_state_after_round

                    update_request_state_after_round(
                        req_state,
                        num_committed=len(committed),
                        new_kv_len=int(full_ids.shape[1]),
                    )
                    if req_state.graph_hints is not None:
                        req_state.graph_hints.static_block_forward = gamma_eff == self.gamma
                    if self._block_pool_config is not None:
                        from serving.block_kv import logical_blocks_for_length

                        tbl = logical_blocks_for_length(
                            int(full_ids.shape[1]),
                            self._block_pool_config,
                        )
                        req_state.extra["logical_num_blocks"] = tbl.num_blocks

            if new_generated != max_new_tokens:
                raise RuntimeError("internal error: token count mismatch")

        total_runtime = time.perf_counter() - t0
        gen_chunk = full_ids[:, prompt_ids.shape[1] :].detach().cpu()
        text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

        acc_rate = (total_draft_accepted / total_draft_proposals) if total_draft_proposals > 0 else 0.0
        decode_phase = (
            float(draft_sum + verify_sum + resync_sum) if benchmark_profile else 0.0
        )
        metrics = DenseSpeculativeMetrics(
            acceptance_rate=float(acc_rate),
            total_accepted_draft_tokens=int(total_draft_accepted),
            total_draft_proposals=int(total_draft_proposals),
            rejection_events=int(rejection_events),
            total_rounds=int(rounds),
            total_runtime_s=float(total_runtime),
            total_new_tokens=int(max_new_tokens),
            prefill_time_s=float(prefill_s) if benchmark_profile else 0.0,
            decode_phase_time_s=decode_phase if benchmark_profile else 0.0,
            draft_phase_time_s_total=float(draft_sum) if benchmark_profile else 0.0,
            verify_phase_time_s_total=float(verify_sum) if benchmark_profile else 0.0,
            quant_resync_time_s_total=float(resync_sum) if benchmark_profile else 0.0,
        )

        if self.verify_match_autoregressive:
            ref = decode_greedy_autoregressive(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                warmup=False,
                trial_index=0,
            )
            if not torch.equal(ref.full_token_ids, full_ids.detach().cpu()):
                raise RuntimeError(
                    "SpeculativeDenseHierarchical decode mismatch vs autoregressive baseline: "
                    f"spec={full_ids.detach().cpu().tolist()} ar={ref.full_token_ids.tolist()}"
                )

        return SpeculativeHierarchicalDecodeResult(
            text=text,
            prompt_token_ids=prompt_ids.detach().cpu(),
            new_token_ids=gen_chunk,
            full_token_ids=full_ids.detach().cpu(),
            metrics=metrics,
            mgr=self.mgr,
            last_round=last_snap,
            round_history=tuple(round_debug_list),
        )


def format_hierarchical_debug_line(snap: HierarchicalRoundDebug | None) -> str:
    """Single-line summary for logging."""
    if snap is None:
        return "hierarchical_debug: <no round yet>"
    return (
        f"hist={snap.hist_len} cf1={snap.cf1_len} cf2_after_draft={snap.cf2_len_after_draft} "
        f"accepted_prefix={snap.accepted_prefix_len} rejected={snap.rejected} gamma={snap.gamma}"
    )


def dump_hierarchical_debug_state(
    mgr: RecentBufferManager,
    last_round: HierarchicalRoundDebug | None = None,
) -> dict[str, int | float | bool | None]:
    """Structured debug payload: hist / CF1 / CF2 occupancy, instrumentation, last accepted prefix."""
    base = mgr.instrumentation_dict()
    if last_round is not None:
        base["accepted_prefix_len"] = last_round.accepted_prefix_len
        base["cf2_len_after_draft"] = last_round.cf2_len_after_draft
        base["last_round_rejected"] = last_round.rejected
        base["last_round_gamma"] = last_round.gamma
    return base
