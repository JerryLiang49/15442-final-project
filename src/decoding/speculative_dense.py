"""Dense greedy self-speculative decoding (same model for draft and verifier).

**Algorithm (one round)**

1. **Draft:** From a clone of the verifier KV, greedily propose ``γ`` tokens (``γ`` sequential
   single-token forwards).
2. **Verify:** One batched forward on the verifier with ``input_ids`` of shape ``[1, γ]`` holding
   the proposals.
3. **Accept:** Find the smallest index ``j`` where draft ``p_j`` differs from the verifier greedy
   prediction at that position; accept ``p_0 … p_{j-1}``. If ``j < γ``, append the **correction**
   token from verifier logits (``start_logits`` if ``j==0``, else ``block_logits[:, j-1]``).
4. **Commit:** Verifier KV is updated to match the committed sequence exactly (trim + correction
   forward on partial reject; full block KV on full accept).

**Greedy alignment (token ↔ logit)**

* Draft ``p_0`` is compared to ``argmax(start_logits)`` (verifier distribution for the first new
  position).
* Draft ``p_i`` for ``i ≥ 1`` is compared to ``argmax(block_logits[:, i - 1, :])`` (prediction at
  the position after ``p_{i-1}`` in the block forward).

This matches standard parallel speculative decoding for greedy verification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from benchmarks.timer import timed_cuda_interval
from cache import KVCacheFP16
from cache.hf_kv_clone import clone_past_key_values
from cache.hf_kv_trim import (
    crop_verifier_past_to_seq_len,
    past_contains_sliding_window_layer,
    verifier_cache_seq_len_hf,
)

from .autoregressive import decode_greedy_autoregressive, model_device

if TYPE_CHECKING:
    from serving.block_kv import BlockPoolConfig

logger = logging.getLogger(__name__)


# --- Draft ---


def draft_cache_from_verifier_snapshot(
    verifier_past: Any,
    *,
    reuse: KVCacheFP16 | None = None,
) -> KVCacheFP16:
    """Draft cache whose KV equals a **clone** of ``verifier_past``.

    ``reuse`` avoids allocating a new :class:`~cache.kv_cache_fp16.KVCacheFP16` wrapper each round
    (orchestration only; tensors are still cloned from the verifier for correctness).
    """
    d = reuse if reuse is not None else KVCacheFP16()
    d.append_from_forward_output(clone_past_key_values(verifier_past))
    return d


def draft_greedy_proposals(
    model: PreTrainedModel,
    *,
    draft_cache: KVCacheFP16,
    start_logits: torch.Tensor,
    gamma: int,
) -> list[torch.Tensor]:
    """Sequentially run ``gamma`` greedy draft steps; each proposal is ``[1, 1]`` long."""
    proposals: list[torch.Tensor] = []
    logits = start_logits
    with torch.inference_mode():
        for _ in range(gamma):
            next_tok = logits.argmax(dim=-1, keepdim=True)
            proposals.append(next_tok)
            past = draft_cache.get_attention_kv()
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            draft_cache.append_from_forward_output(out.past_key_values)
            logits = out.logits[:, -1, :]
    return proposals


# --- Verify / accept ---


def first_mismatch_index_greedy(
    start_logits: torch.Tensor,
    block_logits: torch.Tensor,
    proposal_block: torch.Tensor,
) -> int:
    """Index of first draft token that fails greedy verification, or ``gamma`` if all match.

    ``start_logits``: ``[1, vocab]`` — verifier distribution before the block.
    ``block_logits``: ``[1, gamma, vocab]`` — logits from one block forward on proposals.
    ``proposal_block``: ``[1, gamma]`` — draft token ids.
    """
    if proposal_block.dim() != 2 or proposal_block.shape[0] != 1:
        raise ValueError("first_mismatch_index_greedy supports batch size 1 only")
    _, g = proposal_block.shape
    if block_logits.shape[:2] != (1, g):
        raise ValueError(
            f"block_logits must be [1, gamma, V] with gamma={g}, got {tuple(block_logits.shape)}"
        )
    for j in range(g):
        if j == 0:
            pred = start_logits.argmax(dim=-1, keepdim=True)
        else:
            pred = block_logits[:, j - 1, :].argmax(dim=-1, keepdim=True)
        if not torch.equal(pred, proposal_block[:, j : j + 1]):
            return j
    return g


def greedy_correction_token(
    start_logits: torch.Tensor,
    block_logits: torch.Tensor,
    mismatch_j: int,
) -> torch.Tensor:
    """Greedy verifier token when mismatch is at ``mismatch_j`` (shape ``[1, 1]``)."""
    if mismatch_j == 0:
        return start_logits.argmax(dim=-1, keepdim=True)
    return block_logits[:, mismatch_j - 1, :].argmax(dim=-1, keepdim=True)


def verify_block_and_commit(
    model: PreTrainedModel,
    *,
    verifier_past_at_round_start: Any,
    start_logits: torch.Tensor,
    proposals: list[torch.Tensor],
    debug: bool = False,
) -> tuple[list[torch.Tensor], Any, torch.Tensor, int, bool]:
    """One block forward on proposals; return committed tokens and updated verifier KV.

    **Clone semantics:** this function performs exactly **one** ``clone_past_key_values`` on
    ``verifier_past_at_round_start`` for the block forward. Callers must **not** pre-clone the
    verifier past (that was redundant in older code and doubled host/GPU work per round).

    Returns:
        ``committed_tokens`` (each ``[1,1]``), ``new_verifier_past``, ``new_last_logits`` for the
        next round, ``num_accepted_draft`` (count of accepted **draft** tokens only), ``rejected``
        (True iff not all draft tokens accepted).
    """
    if not proposals:
        raise ValueError("verify_block_and_commit requires at least one proposal")
    proposal_block = torch.cat(proposals, dim=1)
    if proposal_block.shape[0] != 1:
        raise ValueError("verify_block_and_commit supports batch size 1 only")

    ver_past = clone_past_key_values(verifier_past_at_round_start)
    seq_before_block = verifier_cache_seq_len_hf(ver_past)
    g = proposal_block.shape[1]

    with torch.inference_mode():
        block_out = model(
            input_ids=proposal_block,
            past_key_values=ver_past,
            use_cache=True,
        )
        bl = block_out.logits
        j = first_mismatch_index_greedy(start_logits, bl, proposal_block)

        if j == g:
            committed = [proposal_block[:, i : i + 1] for i in range(g)]
            new_past = block_out.past_key_values
            if debug and not past_contains_sliding_window_layer(new_past):
                exp = seq_before_block + g
                if verifier_cache_seq_len_hf(new_past) != exp:
                    raise AssertionError(
                        f"[spec_dense debug] full accept: expected KV len {exp}, "
                        f"got {verifier_cache_seq_len_hf(new_past)}"
                    )
            return committed, new_past, bl[:, -1, :].clone(), g, False

        correction = greedy_correction_token(start_logits, bl, j)
        target_trim = seq_before_block + j

        fix_out = None
        try:
            trimmed = crop_verifier_past_to_seq_len(block_out.past_key_values, target_trim)
            if debug and not past_contains_sliding_window_layer(trimmed):
                if verifier_cache_seq_len_hf(trimmed) != target_trim:
                    raise AssertionError(
                        f"[spec_dense debug] after trim expected {target_trim}, "
                        f"got {verifier_cache_seq_len_hf(trimmed)}"
                    )
            fix_out = model(
                input_ids=correction,
                past_key_values=trimmed,
                use_cache=True,
            )
        except ValueError as exc:
            logger.warning(
                "verify_block_and_commit: trim+correction failed (%s); repair from round-start past.",
                exc,
            )
            if j == 0:
                commit_block = correction
            else:
                commit_block = torch.cat([proposal_block[:, :j], correction], dim=1)
            fix_out = model(
                input_ids=commit_block,
                past_key_values=verifier_past_at_round_start,
                use_cache=True,
            )

        assert fix_out is not None
        committed = [proposal_block[:, i : i + 1] for i in range(j)] + [correction]
        expected_final_len = target_trim + 1
        if debug and not past_contains_sliding_window_layer(fix_out.past_key_values):
            fin = verifier_cache_seq_len_hf(fix_out.past_key_values)
            if fin != expected_final_len:
                raise AssertionError(
                    f"[spec_dense debug] after correction expected KV len {expected_final_len}, got {fin}"
                )

        return committed, fix_out.past_key_values, fix_out.logits[:, -1, :].clone(), j, True


def concat_committed(full_ids: torch.Tensor, committed: list[torch.Tensor]) -> torch.Tensor:
    """Append committed ``[1,1]`` tokens to ``full_ids`` (batch 1)."""
    if not committed:
        return full_ids
    chunk = torch.cat(committed, dim=1)
    return torch.cat([full_ids, chunk], dim=1)


# --- Metrics + decoder ---


@dataclass
class DenseSpeculativeMetrics:
    """Lightweight metrics for dense speculative runs.

    **Phase H (``benchmark_profile=True``)** — CUDA-synchronized intervals via
    :func:`benchmarks.timer.timed_cuda_interval`:

    * ``prefill_time_s``: prompt prefill forward(s).
    * ``decode_phase_time_s``: sum of speculative rounds (draft + verify); excludes prefill.
    * ``draft_phase_time_s_total`` / ``verify_phase_time_s_total``: split inside the decode phase.
    * ``quant_resync_time_s_total``: hierarchical QuantSpec path only (HF KV → store sync).
    """

    acceptance_rate: float
    total_accepted_draft_tokens: int
    total_draft_proposals: int
    rejection_events: int
    total_rounds: int
    total_runtime_s: float
    total_new_tokens: int
    prefill_time_s: float = 0.0
    decode_phase_time_s: float = 0.0
    draft_phase_time_s_total: float = 0.0
    verify_phase_time_s_total: float = 0.0
    quant_resync_time_s_total: float = 0.0

    def to_jsonable(self) -> dict[str, float | int]:
        return {
            "acceptance_rate": float(self.acceptance_rate),
            "total_accepted_draft_tokens": int(self.total_accepted_draft_tokens),
            "total_draft_proposals": int(self.total_draft_proposals),
            "rejection_events": int(self.rejection_events),
            "total_rounds": int(self.total_rounds),
            "total_runtime_s": float(self.total_runtime_s),
            "total_new_tokens": int(self.total_new_tokens),
            "prefill_time_s": float(self.prefill_time_s),
            "decode_phase_time_s": float(self.decode_phase_time_s),
            "draft_phase_time_s_total": float(self.draft_phase_time_s_total),
            "verify_phase_time_s_total": float(self.verify_phase_time_s_total),
            "quant_resync_time_s_total": float(self.quant_resync_time_s_total),
        }


@dataclass(frozen=True)
class DenseSpeculativeDecodeResult:
    """Output of :meth:`SpeculativeDecoderDense.decode`."""

    text: str
    prompt_token_ids: torch.Tensor
    new_token_ids: torch.Tensor
    full_token_ids: torch.Tensor
    metrics: DenseSpeculativeMetrics
    verifier_kv: KVCacheFP16


class SpeculativeDecoderDense:
    """Lossless dense self-speculative decoder (FP16 KV; draft == verifier weights).

    **Serving mode (Phase G)**

    * ``serving_mode=True``: reuse one :class:`~cache.kv_cache_fp16.KVCacheFP16` for draft rounds,
      track :class:`serving.request_state.SpeculativeRequestState`, optional logical block accounting.
    * ``legacy_double_clone_verifier=True``: restore the old **double** verifier clone per round
      (pre-clone at call site plus clone inside :func:`verify_block_and_commit`) — for A/B benchmarks
      only; default ``False``.
    * **Graph capture:** fixed ``gamma`` and ``max_new_tokens`` divisible by ``gamma`` yields
      constant ``gamma_eff`` on all full rounds (tail may shorten the last round).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        gamma: int,
        *,
        verbose: bool = False,
        debug: bool = False,
        verify_match_autoregressive: bool = True,
        serving_mode: bool = False,
        legacy_double_clone_verifier: bool = False,
        block_pool_config: BlockPoolConfig | None = None,
    ) -> None:
        if gamma < 1:
            raise ValueError("gamma must be >= 1")
        self.model = model
        self.tokenizer = tokenizer
        self.gamma = int(gamma)
        self.verbose = verbose
        self.debug = debug
        self.verify_match_autoregressive = verify_match_autoregressive
        self.serving_mode = bool(serving_mode)
        self.legacy_double_clone_verifier = bool(legacy_double_clone_verifier)
        self._block_pool_config = block_pool_config

    def decode(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        benchmark_profile: bool = False,
    ) -> DenseSpeculativeDecodeResult:
        """Generate exactly ``max_new_tokens`` new tokens after ``prompt`` (greedy).

        ``benchmark_profile=True`` records CUDA-synchronized prefill vs decode-phase timings
        (draft vs verify) for Phase H throughput tables; adds sync overhead on CUDA.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        device = model_device(self.model)
        t0 = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids: torch.Tensor = inputs["input_ids"].to(device)
        full_ids = prompt_ids

        ver_kv = KVCacheFP16()
        draft_buf: KVCacheFP16 | None = KVCacheFP16() if self.serving_mode else None
        req_state = None
        if self.serving_mode:
            from serving.request_state import reset_request_state

            req_state = reset_request_state(
                request_id="dense-0",
                prompt_len=int(prompt_ids.shape[1]),
                max_new_tokens=max_new_tokens,
                gamma=self.gamma,
                device=device,
            )

        total_draft_proposals = 0
        total_draft_accepted = 0
        rejection_events = 0
        rounds = 0
        new_generated = 0

        prefill_s = 0.0
        draft_sum = 0.0
        verify_sum = 0.0

        with torch.inference_mode():
            if benchmark_profile:
                with timed_cuda_interval(device) as pre_slot:
                    prefill_out = self.model(input_ids=prompt_ids, use_cache=True, past_key_values=None)
                prefill_s = float(pre_slot[0])
                ver_kv.append_from_forward_output(prefill_out.past_key_values)
            else:
                prefill_out = self.model(input_ids=prompt_ids, use_cache=True, past_key_values=None)
                ver_kv.append_from_forward_output(prefill_out.past_key_values)
            last_logits = prefill_out.logits[:, -1, :].clone()

            while new_generated < max_new_tokens:
                remaining = max_new_tokens - new_generated
                gamma_eff = min(self.gamma, remaining)
                rounds += 1

                logits_snap = last_logits.clone()
                if benchmark_profile:
                    with timed_cuda_interval(device) as d_slot:
                        if draft_buf is not None:
                            draft_round = draft_cache_from_verifier_snapshot(
                                ver_kv.get_attention_kv(),
                                reuse=draft_buf,
                            )
                        else:
                            draft_round = draft_cache_from_verifier_snapshot(ver_kv.get_attention_kv())
                        proposals = draft_greedy_proposals(
                            self.model,
                            draft_cache=draft_round,
                            start_logits=logits_snap,
                            gamma=gamma_eff,
                        )
                    draft_sum += float(d_slot[0])
                else:
                    if draft_buf is not None:
                        draft_round = draft_cache_from_verifier_snapshot(
                            ver_kv.get_attention_kv(),
                            reuse=draft_buf,
                        )
                    else:
                        draft_round = draft_cache_from_verifier_snapshot(ver_kv.get_attention_kv())
                    proposals = draft_greedy_proposals(
                        self.model,
                        draft_cache=draft_round,
                        start_logits=logits_snap,
                        gamma=gamma_eff,
                    )
                total_draft_proposals += len(proposals)

                past_for_verify = (
                    clone_past_key_values(ver_kv.get_attention_kv())
                    if self.legacy_double_clone_verifier
                    else ver_kv.get_attention_kv()
                )
                if benchmark_profile:
                    with timed_cuda_interval(device) as v_slot:
                        committed, new_past, new_logits, n_accepted_draft, rejected = verify_block_and_commit(
                            self.model,
                            verifier_past_at_round_start=past_for_verify,
                            start_logits=logits_snap,
                            proposals=proposals,
                            debug=self.debug,
                        )
                    verify_sum += float(v_slot[0])
                else:
                    committed, new_past, new_logits, n_accepted_draft, rejected = verify_block_and_commit(
                        self.model,
                        verifier_past_at_round_start=past_for_verify,
                        start_logits=logits_snap,
                        proposals=proposals,
                        debug=self.debug,
                    )
                total_draft_accepted += n_accepted_draft
                if rejected:
                    rejection_events += 1

                if self.verbose:
                    prop_ids = [int(p[0, 0].item()) for p in proposals]
                    com_ids = [int(c[0, 0].item()) for c in committed]
                    print(
                        f"[spec_dense] round={rounds} gamma={gamma_eff} proposals={prop_ids} "
                        f"accepted_draft={n_accepted_draft} rejected={rejected} committed={com_ids}"
                    )

                ver_kv.append_from_forward_output(new_past)
                if self.debug and not past_contains_sliding_window_layer(ver_kv.get_attention_kv()):
                    tok_len = int(full_ids.shape[1])
                    kv_len = verifier_cache_seq_len_hf(ver_kv.get_attention_kv())
                    if kv_len != tok_len + len(committed):
                        raise AssertionError(
                            f"[spec_dense debug] pre-commit invariant: seq={tok_len}, "
                            f"n_committed={len(committed)}, kv_len={kv_len}"
                        )

                full_ids = concat_committed(full_ids, committed)
                if self.debug and not past_contains_sliding_window_layer(ver_kv.get_attention_kv()):
                    kv_len = verifier_cache_seq_len_hf(ver_kv.get_attention_kv())
                    if kv_len != int(full_ids.shape[1]):
                        raise AssertionError(
                            f"[spec_dense debug] verifier KV len {kv_len} != token seq {full_ids.shape[1]}"
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
        decode_phase = float(draft_sum + verify_sum) if benchmark_profile else 0.0
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
            quant_resync_time_s_total=0.0,
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
                    "SpeculativeDense decode mismatch vs autoregressive baseline: "
                    f"spec={full_ids.detach().cpu().tolist()} ar={ref.full_token_ids.tolist()}"
                )

        return DenseSpeculativeDecodeResult(
            text=text,
            prompt_token_ids=prompt_ids.detach().cpu(),
            new_token_ids=gen_chunk,
            full_token_ids=full_ids.detach().cpu(),
            metrics=metrics,
            verifier_kv=ver_kv,
        )
