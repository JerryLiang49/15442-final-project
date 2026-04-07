"""Uncompressed FP16 self-speculative decoding (draft and verifier share weights).

Algorithm (greedy, one round):

1. Snapshot verifier KV and ``last_logits`` (distribution for the next token).
2. **Draft** runs up to ``K`` greedy steps from a **draft** :class:`~mlsys_kv.cache.kv_cache_base.KVCacheBase`
   loaded from a cloned verifier snapshot (mode selects backend; Phase 4 implements ``fp16`` only).
3. **Verifier (Phase 9–10)** checks all ``K`` proposals with **one** block forward (``[1, K]``,
   batch **1**). On mismatch, **crop** block KV to length ``L0+j`` then **one** forward on the
   **correction** token (see :mod:`mlsys_kv.cache.hf_kv_trim`). If crop is unsafe (e.g. HF
   sliding-window limits), **repair** from round-start ``past`` over ``[accepted draft | correction]``.

**Greedy alignment (token ↔ logit)**

Let ``start_logits`` be verifier logits for the first new position (shape ``[1, V]``). Let
``block_logits`` be ``model(...).logits`` from the block forward (shape ``[1, K, V]``).

* Draft ``p_0`` is compared to ``argmax(start_logits)``.
* Draft ``p_i`` for ``i >= 1`` is compared to ``argmax(block_logits[:, i - 1, :])``.

**Accepted prefix**

Let ``j`` be the smallest index where ``p_j`` differs from the verifier prediction, or ``j = K``
if all match. **Accepted draft tokens** are ``p_0..p_{j-1}``. If ``j < K``, the **correction** is
``argmax(start_logits)`` when ``j == 0``, else ``argmax(block_logits[:, j - 1, :])``. If ``j == K``,
all ``K`` proposals are committed; next logits are ``block_logits[:, -1, :]`` with KV from that
block forward.

**Acceptance rate** (metrics): ``total_accepted_tokens / total_draft_proposals``, where accepted
tokens are draft proposals that matched the verifier prefix (verifier fallback tokens are not
counted as accepted draft tokens).

**``K``:** maximum draft proposals attempted per round (clamped to remaining new tokens).

**Phase 12 (sparse draft integration):** Retention policy runs inside
:class:`~mlsys_kv.cache.sparse_hf_integration.SparseHFCacheIntegrator`. If you **reuse** a
:class:`~mlsys_kv.cache.kv_cache_sparse.KVCacheSparse` (or quantized variant) across prompts,
call :meth:`~mlsys_kv.cache.kv_cache_base.KVCacheBase.reset` before the new prompt so integrator
state cannot leak.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from mlsys_kv.benchmarks.timer import cuda_synchronize
from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.draft_factory import draft_cache_from_verifier_snapshot
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.kv_quant_semantics import IMPLEMENTED_DRAFT_KV_QUANTIZATION_SEMANTICS
from mlsys_kv.cache.hf_kv_clone import clone_past_key_values
from mlsys_kv.cache.hf_kv_trim import (
    crop_verifier_past_to_seq_len,
    past_contains_sliding_window_layer,
    verifier_cache_seq_len_hf,
)
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.kv_cache_fp16 import KVCacheFP16
from mlsys_kv.cache.kv_cache_int4 import KVCacheInt4Packed
from mlsys_kv.cache.kv_cache_quantized import KVCacheQuantized
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse
from mlsys_kv.cache.kv_cache_sparse_quantized import KVCacheSparseQuantized
from mlsys_kv.decoding.autoregressive import decode_greedy_autoregressive, model_device

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeMetrics:
    """Aggregated statistics for a speculative generation run."""

    acceptance_rate: float
    total_accepted_tokens: int
    total_draft_proposals: int
    avg_accepted_tokens_per_round: float
    rejection_events: int
    total_rounds: int
    total_runtime_s: float
    total_new_tokens: int
    draft_cache_mode: str
    draft_dequant_time_s_total: float = 0.0
    draft_refresh_time_s_total: float = 0.0
    draft_mean_sparsity_ratio: float = 0.0
    draft_quantization_kv_bits: int | None = None
    draft_cache_end_stats: dict[str, Any] | None = None
    draft_phase_time_s_total: float = 0.0
    verify_phase_time_s_total: float = 0.0
    # Phase 13: benchmark honesty — quantized draft KV is memory-only unless native path exists.
    draft_kv_quantization_semantics: str | None = None
    draft_runtime_accelerated_quant_attention: bool = False

    def to_jsonable(self) -> dict[str, float | int | str | dict[str, Any] | bool | None]:
        d: dict[str, Any] = {
            "acceptance_rate": float(self.acceptance_rate),
            "total_accepted_tokens": int(self.total_accepted_tokens),
            "total_draft_proposals": int(self.total_draft_proposals),
            "avg_accepted_tokens_per_round": float(self.avg_accepted_tokens_per_round),
            "rejection_events": int(self.rejection_events),
            "total_rounds": int(self.total_rounds),
            "total_runtime_s": float(self.total_runtime_s),
            "total_new_tokens": int(self.total_new_tokens),
            "draft_cache_mode": str(self.draft_cache_mode),
            "draft_dequant_time_s_total": float(self.draft_dequant_time_s_total),
            "draft_refresh_time_s_total": float(self.draft_refresh_time_s_total),
            "draft_mean_sparsity_ratio": float(self.draft_mean_sparsity_ratio),
            "draft_quantization_kv_bits": (
                int(self.draft_quantization_kv_bits)
                if self.draft_quantization_kv_bits is not None
                else None
            ),
            "draft_cache_end_stats": dict(self.draft_cache_end_stats) if self.draft_cache_end_stats else None,
            "draft_phase_time_s_total": float(self.draft_phase_time_s_total),
            "verify_phase_time_s_total": float(self.verify_phase_time_s_total),
            "draft_kv_quantization_semantics": self.draft_kv_quantization_semantics,
            "draft_runtime_accelerated_quant_attention": bool(
                self.draft_runtime_accelerated_quant_attention
            ),
        }
        return d


@dataclass(frozen=True)
class SpeculativeDecodeResult:
    """Decoded output plus metrics and optional draft/ver cache introspection."""

    text: str
    prompt_token_ids: torch.Tensor
    new_token_ids: torch.Tensor
    full_token_ids: torch.Tensor
    metrics: SpeculativeMetrics
    verifier_kv: KVCacheFP16
    draft_kv: KVCacheBase


def propose_draft_tokens(
    model: PreTrainedModel,
    *,
    draft_cache: KVCacheBase,
    start_logits: torch.Tensor,
    k: int,
) -> list[torch.Tensor]:
    """Run **K** greedy draft steps mutating ``draft_cache`` (HF adapter per step)."""
    proposals: list[torch.Tensor] = []
    logits = start_logits
    with torch.inference_mode():
        for _ in range(k):
            next_tok = logits.argmax(dim=-1, keepdim=True)
            proposals.append(next_tok)
            note = getattr(draft_cache, "note_forward_token", None)
            if callable(note):
                note(next_tok)
            past = draft_cache.get_attention_kv()
            pos = draft_cache.position_ids_for_next_queries(
                1, batch_size=int(next_tok.shape[0]), device=next_tok.device
            )
            fwd_kw: dict[str, Any] = {
                "input_ids": next_tok,
                "past_key_values": past,
                "use_cache": True,
            }
            if pos is not None:
                # Phase 11: sparse caches pass absolute positions L, L+1, …; HF default uses
                # get_seq_length() == physical R, which breaks wpe/RoPE for the next query.
                fwd_kw["position_ids"] = pos
            out = model(**fwd_kw)
            draft_cache.append_from_forward_output(out.past_key_values)
            logits = out.logits[:, -1, :]
    return proposals


def first_greedy_speculative_mismatch(
    start_logits: torch.Tensor,
    block_logits: torch.Tensor,
    proposal_block: torch.Tensor,
) -> int:
    """First draft index that fails greedy verification, or ``K`` if all match.

    **Batch size 1 only:** ``start_logits`` is ``[1, vocab]``, ``block_logits`` is ``[1, K, vocab]``,
    ``proposal_block`` is ``[1, K]`` (token ids).

    See module docstring for token ↔ logit alignment.
    """
    if proposal_block.dim() != 2 or proposal_block.shape[0] != 1:
        raise ValueError("first_greedy_speculative_mismatch supports batch size 1 only")
    _, k = proposal_block.shape
    if block_logits.shape[:2] != (1, k):
        raise ValueError(
            f"block_logits must be [1, K, V] with K={k}, got {tuple(block_logits.shape)}"
        )
    for j in range(k):
        if j == 0:
            pred = start_logits.argmax(dim=-1, keepdim=True)
        else:
            pred = block_logits[:, j - 1, :].argmax(dim=-1, keepdim=True)
        if not torch.equal(pred, proposal_block[:, j : j + 1]):
            return j
    return k


def greedy_speculative_correction_token(
    start_logits: torch.Tensor,
    block_logits: torch.Tensor,
    mismatch_j: int,
) -> torch.Tensor:
    """Greedy correction token ``[1, 1]`` long when mismatch occurs at ``mismatch_j``."""
    if mismatch_j == 0:
        return start_logits.argmax(dim=-1, keepdim=True)
    return block_logits[:, mismatch_j - 1, :].argmax(dim=-1, keepdim=True)


def verify_greedy_proposals(
    model: PreTrainedModel,
    *,
    verifier_past: Any,
    start_logits: torch.Tensor,
    proposals: list[torch.Tensor],
    debug: bool = False,
) -> tuple[list[torch.Tensor], Any, torch.Tensor, int, bool]:
    """Greedy speculative verification with **one** block forward over proposed tokens (batch 1).

    On rejection: **crop** block KV to ``seq_before_block+j``, then **one** forward on the correction token
    (see :mod:`mlsys_kv.cache.hf_kv_trim`). **Full accept:** keep block ``past_key_values``.

    Args:
        model: Causal LM in eval mode.
        verifier_past: HF ``past_key_values`` at round start.
        start_logits: Verifier logits for the first new position, ``[1, vocab]``.
        proposals: Length-``K`` list of ``[1, 1]`` long draft tokens.
        debug: If ``True``, assert cache-length invariants (skipped for sliding-window caches).

    Returns:
        ``committed_tokens``, ``new_verifier_past``, ``new_last_logits``, ``num_accepted_draft``,
        ``rejected`` (``True`` iff ``j < K``).
    """
    if not proposals:
        raise ValueError("verify_greedy_proposals requires at least one proposal")
    proposal_block = torch.cat(proposals, dim=1)
    if proposal_block.shape[0] != 1:
        raise ValueError("verify_greedy_proposals supports batch size 1 only")

    # HF mutates ``past_key_values`` in place — capture length **before** the block forward.
    seq_before_block = verifier_cache_seq_len_hf(verifier_past)

    with torch.inference_mode():
        block_out = model(
            input_ids=proposal_block,
            past_key_values=verifier_past,
            use_cache=True,
        )
        bl = block_out.logits
        j = first_greedy_speculative_mismatch(start_logits, bl, proposal_block)
        k = proposal_block.shape[1]

        if j == k:
            committed = [proposal_block[:, i : i + 1] for i in range(k)]
            new_past = block_out.past_key_values
            if debug and not past_contains_sliding_window_layer(new_past):
                exp = seq_before_block + k
                if verifier_cache_seq_len_hf(new_past) != exp:
                    raise AssertionError(
                        "[spec debug] full accept: KV length should be seq_before+K="
                        f"{exp}, got {verifier_cache_seq_len_hf(new_past)}"
                    )
            return (
                committed,
                new_past,
                bl[:, -1, :].clone(),
                k,
                False,
            )

        correction = greedy_speculative_correction_token(start_logits, bl, j)
        cur_block = verifier_cache_seq_len_hf(block_out.past_key_values)
        if debug and not past_contains_sliding_window_layer(
            verifier_past
        ) and not past_contains_sliding_window_layer(block_out.past_key_values):
            if cur_block != seq_before_block + k:
                raise AssertionError(
                    f"[spec debug] after block expected KV len {seq_before_block + k}, got {cur_block}"
                )

        # Invariant: keep prefix ``0 .. seq_before_block + j - 1`` (accepted draft only); drop rejected tail.
        target_trim = seq_before_block + j
        fix_out = None
        try:
            trimmed = crop_verifier_past_to_seq_len(block_out.past_key_values, target_trim)
            if debug and not past_contains_sliding_window_layer(trimmed):
                tlen = verifier_cache_seq_len_hf(trimmed)
                if tlen != target_trim:
                    raise AssertionError(
                        f"[spec debug] after trim expected {target_trim}, got {tlen}"
                    )
            fix_out = model(
                input_ids=correction,
                past_key_values=trimmed,
                use_cache=True,
            )
            if debug:
                logger.debug(
                    "verify: trim+correction seq_before_block=%s j=%s K=%s target_trim=%s",
                    seq_before_block,
                    j,
                    k,
                    target_trim,
                )
        except ValueError as exc:
            logger.warning(
                "verify_greedy_proposals: trim+correction failed (%s); repair forward from round start.",
                exc,
            )
            if j == 0:
                commit_block = correction
            else:
                commit_block = torch.cat([proposal_block[:, :j], correction], dim=1)
            fix_out = model(
                input_ids=commit_block,
                past_key_values=verifier_past,
                use_cache=True,
            )

        assert fix_out is not None  # for type checker
        committed = [proposal_block[:, i : i + 1] for i in range(j)] + [correction]
        # After one token of correction: committed sequence length is ``seq_before_block + j + 1``.
        expected_final_len = target_trim + 1
        if debug and not past_contains_sliding_window_layer(fix_out.past_key_values):
            fin = verifier_cache_seq_len_hf(fix_out.past_key_values)
            if fin != expected_final_len:
                raise AssertionError(
                    f"[spec debug] after correction expected KV len {expected_final_len}, got {fin}"
                )

        return (
            committed,
            fix_out.past_key_values,
            fix_out.logits[:, -1, :].clone(),
            j,
            True,
        )


def commit_tokens_to_sequence(
    full_ids: torch.Tensor,
    committed: list[torch.Tensor],
) -> torch.Tensor:
    """Concatenate committed ``[1,1]`` tokens onto ``full_ids`` (batch 1)."""
    if not committed:
        return full_ids
    chunk = torch.cat(committed, dim=1)
    return torch.cat([full_ids, chunk], dim=1)


class SpeculativeDecoder:
    """Self-speculative greedy decoder: verifier uses :class:`~mlsys_kv.cache.kv_cache_fp16.KVCacheFP16`; draft uses ``draft_mode``."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        k: int,
        *,
        draft_mode: DraftCacheMode = DraftCacheMode.FP16,
        verbose: bool = False,
        verify_match: bool = True,
        sparse_config: SparseRetentionConfig | None = None,
        kv_quant_bits: int = 8,
        debug_speculative: bool = False,
    ) -> None:
        if k < 1:
            raise ValueError("K must be >= 1")
        self.model = model
        self.tokenizer = tokenizer
        self.k = int(k)
        self.draft_mode = draft_mode
        self.verbose = verbose
        self.verify_match = verify_match
        self.sparse_config = sparse_config
        self.kv_quant_bits = int(kv_quant_bits)
        self.debug_speculative = bool(debug_speculative)

    def decode(self, prompt: str, max_new_tokens: int) -> SpeculativeDecodeResult:
        """Generate ``max_new_tokens`` new tokens after ``prompt``."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        device = model_device(self.model)
        cuda_synchronize(device)
        t0 = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids: torch.Tensor = inputs["input_ids"].to(device)
        full_ids = prompt_ids

        ver_kv = KVCacheFP16()
        ver_kv.reset()
        draft_kv: KVCacheBase

        total_draft_proposals = 0
        total_draft_accepted = 0
        rejection_events = 0
        rounds = 0
        new_generated = 0
        total_draft_dequant_s = 0.0
        total_draft_refresh_s = 0.0
        sparsity_sum_rounds = 0.0
        n_sparse_round_stats = 0
        total_draft_phase_s = 0.0
        total_verify_phase_s = 0.0

        with torch.inference_mode():
            prefill_out = self.model(
                input_ids=prompt_ids,
                use_cache=True,
                past_key_values=None,
            )
            vp = prefill_out.past_key_values
            ver_kv.append_from_forward_output(vp)
            draft_kv = draft_cache_from_verifier_snapshot(
                self.draft_mode,
                vp,
                model=self.model,
                sparse_config=self.sparse_config,
                kv_quant_bits=self.kv_quant_bits,
            )
            last_logits = prefill_out.logits[:, -1, :].clone()

            if self.verbose:
                print(f"[spec] prefill seq_len={prompt_ids.shape[1]} draft_mode={self.draft_mode.value}")

            while new_generated < max_new_tokens:
                remaining = max_new_tokens - new_generated
                k_eff = min(self.k, remaining)
                rounds += 1

                logits_snap = last_logits.clone()

                draft_round = draft_cache_from_verifier_snapshot(
                    self.draft_mode,
                    ver_kv.get_attention_kv(),
                    model=self.model,
                    sparse_config=self.sparse_config,
                    kv_quant_bits=self.kv_quant_bits,
                )
                cuda_synchronize(device)
                t_draft_0 = time.perf_counter()
                proposals = propose_draft_tokens(
                    self.model,
                    draft_cache=draft_round,
                    start_logits=logits_snap,
                    k=k_eff,
                )
                cuda_synchronize(device)
                total_draft_phase_s += time.perf_counter() - t_draft_0

                total_draft_proposals += len(proposals)
                if isinstance(draft_round, KVCacheSparseQuantized):
                    st_dr = draft_round.stats()
                    total_draft_dequant_s += float(st_dr.get("cumulative_dequant_time_s", 0.0))
                    total_draft_refresh_s += float(st_dr.get("cumulative_refresh_time_s", 0.0))
                    sparsity_sum_rounds += float(st_dr.get("mean_sparsity_ratio_over_appends", 0.0))
                    n_sparse_round_stats += 1
                elif isinstance(draft_round, KVCacheQuantized):
                    total_draft_dequant_s += float(
                        draft_round.stats().get("cumulative_dequant_time_s", 0.0)
                    )
                elif isinstance(draft_round, KVCacheInt4Packed):
                    total_draft_dequant_s += float(
                        draft_round.stats().get("cumulative_dequant_time_s", 0.0)
                    )
                elif isinstance(draft_round, KVCacheSparse):
                    st_dr = draft_round.stats()
                    total_draft_refresh_s += float(st_dr.get("cumulative_refresh_time_s", 0.0))
                    sparsity_sum_rounds += float(st_dr.get("mean_sparsity_ratio_over_appends", 0.0))
                    n_sparse_round_stats += 1

                cuda_synchronize(device)
                t_ver_0 = time.perf_counter()
                ver_past = clone_past_key_values(ver_kv.get_attention_kv())
                committed, new_past, new_logits, n_matched, rejected = verify_greedy_proposals(
                    self.model,
                    verifier_past=ver_past,
                    start_logits=logits_snap,
                    proposals=proposals,
                    debug=self.debug_speculative,
                )
                cuda_synchronize(device)
                total_verify_phase_s += time.perf_counter() - t_ver_0
                total_draft_accepted += n_matched
                if rejected:
                    rejection_events += 1

                if self.verbose:
                    prop_ids = [int(p[0, 0].item()) for p in proposals]
                    com_ids = [int(c[0, 0].item()) for c in committed]
                    print(
                        f"[spec] round={rounds} K={k_eff} proposals={prop_ids} "
                        f"matched_prefix={n_matched} rejected={rejected} committed={com_ids}"
                    )

                ver_kv.append_from_forward_output(new_past)
                if self.debug_speculative and not past_contains_sliding_window_layer(
                    ver_kv.get_attention_kv()
                ):
                    kv_len = verifier_cache_seq_len_hf(ver_kv.get_attention_kv())
                    tok_len = int(full_ids.shape[1])
                    # Invariant: KV length after this round's forwards == prior tokens + this round's commits.
                    if kv_len != tok_len + len(committed):
                        raise AssertionError(
                            f"[spec debug] KV vs commits: seq_before={tok_len}, "
                            f"committed_n={len(committed)}, kv_len={kv_len}"
                        )
                full_ids = commit_tokens_to_sequence(full_ids, committed)
                if self.debug_speculative and not past_contains_sliding_window_layer(
                    ver_kv.get_attention_kv()
                ):
                    kv_len = verifier_cache_seq_len_hf(ver_kv.get_attention_kv())
                    if kv_len != int(full_ids.shape[1]):
                        raise AssertionError(
                            f"[spec debug] verifier KV len {kv_len} != token seq {full_ids.shape[1]}"
                        )

                draft_kv = draft_cache_from_verifier_snapshot(
                    self.draft_mode,
                    new_past,
                    model=self.model,
                    sparse_config=self.sparse_config,
                    kv_quant_bits=self.kv_quant_bits,
                )
                last_logits = new_logits.clone()

                new_generated += len(committed)

            if new_generated != max_new_tokens:
                raise RuntimeError("internal error: did not reach target token count")

        cuda_synchronize(device)
        total_runtime = time.perf_counter() - t0

        gen_chunk = full_ids[:, prompt_ids.shape[1] :].detach().cpu()
        text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

        acc_rate = (total_draft_accepted / total_draft_proposals) if total_draft_proposals > 0 else 0.0
        avg_acc = (total_draft_accepted / rounds) if rounds > 0 else 0.0

        end_st = draft_kv.stats()
        mean_sparsity = (
            (sparsity_sum_rounds / n_sparse_round_stats) if n_sparse_round_stats > 0 else 0.0
        )
        qbits: int | None = None
        qb = end_st.get("quantization_kv_bits")
        if isinstance(qb, int):
            qbits = qb

        if self.draft_mode in (DraftCacheMode.QUANT_ONLY, DraftCacheMode.SPARSE_QUANT):
            q_sem = IMPLEMENTED_DRAFT_KV_QUANTIZATION_SEMANTICS
            runtime_q = False
            logger.debug(
                "Phase 13 draft KV: quantization_semantics=%s runtime_accelerated_quant_attention=%s "
                "(HF attention consumes dequantized K/V; do not infer speedup from narrow-bit storage alone).",
                q_sem,
                runtime_q,
            )
        else:
            q_sem = None
            runtime_q = False

        metrics = SpeculativeMetrics(
            acceptance_rate=float(acc_rate),
            total_accepted_tokens=int(total_draft_accepted),
            total_draft_proposals=int(total_draft_proposals),
            avg_accepted_tokens_per_round=float(avg_acc),
            rejection_events=int(rejection_events),
            total_rounds=int(rounds),
            total_runtime_s=float(total_runtime),
            total_new_tokens=int(max_new_tokens),
            draft_cache_mode=self.draft_mode.value,
            draft_dequant_time_s_total=float(total_draft_dequant_s),
            draft_refresh_time_s_total=float(total_draft_refresh_s),
            draft_mean_sparsity_ratio=mean_sparsity,
            draft_quantization_kv_bits=qbits,
            draft_cache_end_stats=dict(end_st),
            draft_phase_time_s_total=float(total_draft_phase_s),
            verify_phase_time_s_total=float(total_verify_phase_s),
            draft_kv_quantization_semantics=q_sem,
            draft_runtime_accelerated_quant_attention=runtime_q,
        )

        if self.verify_match:
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
                    "Speculative decode mismatch vs autoregressive baseline "
                    f"spec={full_ids.detach().cpu().tolist()} ar={ref.full_token_ids.tolist()}"
                )

        return SpeculativeDecodeResult(
            text=text,
            prompt_token_ids=prompt_ids.detach().cpu(),
            new_token_ids=gen_chunk,
            full_token_ids=full_ids.detach().cpu(),
            metrics=metrics,
            verifier_kv=ver_kv,
            draft_kv=draft_kv,
        )
