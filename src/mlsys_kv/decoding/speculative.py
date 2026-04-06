"""Uncompressed FP16 self-speculative decoding (draft and verifier share weights).

Algorithm (greedy, one round):

1. Snapshot verifier KV and ``last_logits`` (distribution for the next token).
2. **Draft** runs up to ``K`` greedy steps from a **draft** :class:`~mlsys_kv.cache.kv_cache_base.KVCacheBase`
   loaded from a cloned verifier snapshot (mode selects backend; Phase 4 implements ``fp16`` only).
3. **Verifier** replays proposals using a **fresh FP16** clone of the verifier KV (unchanged from Phase 3).
4. After acceptance, the **verifier** cache is updated from HF ``past_key_values``; the **draft** cache
   is resynced from a clone of the new verifier state for inspection/metrics.

**Acceptance rate** (metrics): ``total_accepted_tokens / total_draft_proposals``, where accepted
tokens are draft proposals that matched the verifier prefix (verifier fallback tokens are not
counted as accepted draft tokens).

**``K``:** maximum draft proposals attempted per round (clamped to remaining new tokens).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from mlsys_kv.benchmarks.timer import cuda_synchronize
from mlsys_kv.cache.draft_cache_mode import DraftCacheMode
from mlsys_kv.cache.draft_factory import draft_cache_from_verifier_snapshot
from mlsys_kv.cache.heavy_hitter_selector import SparseRetentionConfig
from mlsys_kv.cache.hf_kv_clone import clone_past_key_values
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.kv_cache_fp16 import KVCacheFP16
from mlsys_kv.cache.kv_cache_quantized import KVCacheQuantized
from mlsys_kv.cache.kv_cache_sparse import KVCacheSparse
from mlsys_kv.decoding.autoregressive import decode_greedy_autoregressive, model_device


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
    draft_cache_end_stats: dict[str, Any] | None = None

    def to_jsonable(self) -> dict[str, float | int | str | dict[str, Any] | None]:
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
            "draft_cache_end_stats": dict(self.draft_cache_end_stats) if self.draft_cache_end_stats else None,
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
            out = model(
                input_ids=next_tok,
                past_key_values=draft_cache.get_attention_kv(),
                use_cache=True,
            )
            draft_cache.append_from_forward_output(out.past_key_values)
            logits = out.logits[:, -1, :]
    return proposals


def verify_greedy_proposals(
    model: PreTrainedModel,
    *,
    verifier_past: Any,
    start_logits: torch.Tensor,
    proposals: list[torch.Tensor],
) -> tuple[list[torch.Tensor], Any, torch.Tensor, int, bool]:
    """Verifier replay; greedy acceptance with longest-prefix match.

    ``verifier_past`` is native HF ``past_key_values`` (**clone** at round start), not a wrapper,
    so the loop matches the Phase 3 implementation bit-for-bit.

    Returns:
        ``committed_tokens``, ``new_verifier_past``, ``new_last_logits``, ``num_matched_draft``, ``rejected``
    """
    v_past = verifier_past
    logits = start_logits
    with torch.inference_mode():
        for i, d_i in enumerate(proposals):
            pred = logits.argmax(dim=-1, keepdim=True)
            if not torch.equal(pred, d_i):
                out = model(input_ids=pred, past_key_values=v_past, use_cache=True)
                new_past = out.past_key_values
                new_logits = out.logits[:, -1, :]
                committed = proposals[:i] + [pred]
                return committed, new_past, new_logits, i, True
            out = model(input_ids=d_i, past_key_values=v_past, use_cache=True)
            v_past = out.past_key_values
            logits = out.logits[:, -1, :]
    committed = list(proposals)
    return committed, v_past, logits, len(proposals), False


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
                )
                proposals = propose_draft_tokens(
                    self.model,
                    draft_cache=draft_round,
                    start_logits=logits_snap,
                    k=k_eff,
                )
                total_draft_proposals += len(proposals)
                if isinstance(draft_round, KVCacheQuantized):
                    total_draft_dequant_s += float(
                        draft_round.stats().get("cumulative_dequant_time_s", 0.0)
                    )
                if isinstance(draft_round, KVCacheSparse):
                    st_dr = draft_round.stats()
                    total_draft_refresh_s += float(st_dr.get("cumulative_refresh_time_s", 0.0))
                    sparsity_sum_rounds += float(st_dr.get("mean_sparsity_ratio_over_appends", 0.0))
                    n_sparse_round_stats += 1

                ver_past = clone_past_key_values(ver_kv.get_attention_kv())
                committed, new_past, new_logits, n_matched, rejected = verify_greedy_proposals(
                    self.model,
                    verifier_past=ver_past,
                    start_logits=logits_snap,
                    proposals=proposals,
                )
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
                draft_kv = draft_cache_from_verifier_snapshot(
                    self.draft_mode,
                    new_past,
                    model=self.model,
                    sparse_config=self.sparse_config,
                )
                last_logits = new_logits.clone()
                full_ids = commit_tokens_to_sequence(full_ids, committed)

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
            draft_cache_end_stats=dict(end_st),
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
