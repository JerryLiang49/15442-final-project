"""Heavy-hitter token selection for sparse draft KV (SnapKV-style, simplified).

**Retention rule (token-level, layer-shared)**

For sequence length ``L``:

* **Recent window** — always keep the last ``W`` tokens: indices
  ``{L-W, ..., L-1}`` (clamped when ``L < W``).
* **Heavy hitters** — from the **pool** of earlier tokens ``{0, ..., L-W-1}``,
  keep the top ``H`` tokens by an importance **score** (same index set for every
  layer; no per-head masks in this phase).

The retained set is the sorted union of these two subsets.

**Scoring**

* ``scoring="attention"``: one forward with ``output_attentions=True`` (forces
  ``attn_implementation="eager"`` on the config), **last layer** weights averaged
  over heads for the single new query vs prefix keys (length ``L-1``), then
  padded so index ``L-1`` is ``+inf``. **Note:** some stacks (e.g. GPT-2 on recent
  ``transformers`` with ``past_key_values``) return an empty ``attentions`` tuple;
  draft code falls back to key-norm in that case.
* ``scoring="key_norm"`` (default in :class:`~mlsys_kv.infra.config.RunConfig`):
  mean ``||K_{ℓ,h,i,:}||`` over layers ``ℓ`` and heads ``h`` — cheap, stable, and
  works with any KV layout; used for verifier snapshots without a paired decode
  token or when attention weights are unavailable.

**Refresh**

Every ``refresh_interval`` :meth:`KVCacheSparse.append_from_forward_output` calls,
scores are recomputed (see :mod:`mlsys_kv.cache.kv_cache_sparse`). Between
refreshes, the same score vector is reused so the heavy-hitter subset tracks a
stale ranking until the next refresh — a deliberate trade-off for low selector
overhead, consistent with periodic SnapKV-style schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import PreTrainedModel

from mlsys_kv.cache.hf_kv_clone import past_sequence_length
from mlsys_kv.cache.kv_cache_fp16 import _iter_kv_tensors

ScoringMode = Literal["attention", "key_norm"]


@dataclass
class SparseRetentionConfig:
    """Hyperparameters for draft-only sparse retention (see module docstring)."""

    recent_window: int = 32
    heavy_hitter_budget: int = 32
    refresh_interval: int = 4
    scoring: ScoringMode = "key_norm"

    def __post_init__(self) -> None:
        if self.recent_window < 0:
            raise ValueError("recent_window must be non-negative")
        if self.heavy_hitter_budget < 0:
            raise ValueError("heavy_hitter_budget must be non-negative")
        if self.refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")
        if self.scoring not in ("attention", "key_norm"):
            raise ValueError("scoring must be 'attention' or 'key_norm'")


def select_retained_token_indices(
    seq_len: int,
    scores: torch.Tensor,
    *,
    recent_window: int,
    heavy_hitter_budget: int,
    eligible_positions: frozenset[int] | None = None,
) -> list[int]:
    """Return sorted unique indices to retain (recent ∪ top heavy hitters in pool).

    **Sparse draft (Phase 11):** When the KV tensor only stores a subset of past
    tokens, pass ``eligible_positions`` so recent / heavy pools only consider
    globals that have a physical K/V row; the result is still **global** indices.
    """
    if seq_len <= 0:
        return []
    scores = scores.flatten()[:seq_len]
    recent = list(range(max(0, seq_len - recent_window), seq_len))
    if eligible_positions is not None:
        recent = [i for i in recent if i in eligible_positions]
    pool_end = max(0, seq_len - recent_window)
    pool = list(range(0, pool_end))
    if eligible_positions is not None:
        pool = [i for i in pool if i in eligible_positions]

    chosen = sorted(set(recent))
    if heavy_hitter_budget <= 0 or not pool:
        if not chosen and eligible_positions:
            fb = sorted(eligible_positions)
            w = min(max(recent_window, 1), len(fb))
            return fb[-w:]
        return chosen

    device = scores.device
    pool_t = torch.tensor(pool, device=device, dtype=torch.long)
    pool_scores = scores[pool_t]
    k = min(int(heavy_hitter_budget), len(pool))
    _, topi = torch.topk(pool_scores, k=k)
    heavy = [pool[int(j)] for j in topi.tolist()]
    out = sorted(set(recent) | set(heavy))
    if not out and eligible_positions:
        fb = sorted(eligible_positions)
        w = min(max(recent_window, 1), len(fb))
        return fb[-w:]
    return out


def key_norm_token_scores(past_key_values: Any) -> torch.Tensor:
    """Aggregate key-vector L2 norms per token across layers and heads (fallback scorer)."""
    pairs = _iter_kv_tensors(past_key_values)
    if not pairs:
        return torch.zeros(0)
    L = int(pairs[0][0].shape[-2])
    total = torch.zeros(L, device=pairs[0][0].device, dtype=torch.float32)
    for k, _ in pairs:
        kn = k.float().norm(dim=-1).mean(dim=0).mean(dim=0)  # [L]
        if kn.shape[0] != L:
            raise RuntimeError("inconsistent KV lengths across layers in key_norm_token_scores")
        total = total + kn
    return total


def attention_mass_from_last_token(
    model: PreTrainedModel,
    past_prefix: Any,
    last_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Attention mass over prefix key positions for the **last** layer (single query step).

    Args:
        model: Causal LM (same device / dtype as ``past_prefix``).
        past_prefix: HF ``past_key_values`` with length ``L-1`` (no KV for ``last_token_ids`` yet).
        last_token_ids: ``[batch, 1]`` token id(s) fed on this step.

    Returns:
        Float tensor of shape ``[L-1]`` (unnormalized attention weights averaged over heads).
    """
    if past_prefix is None or past_sequence_length(past_prefix) == 0:
        return torch.zeros(0, device=last_token_ids.device)
    cfg = model.config
    prev_attn_impl = getattr(cfg, "attn_implementation", None)
    try:
        # SDPA / flash backends often omit attentions; SnapKV scoring needs weights.
        cfg.attn_implementation = "eager"
        with torch.inference_mode():
            out = model(
                input_ids=last_token_ids,
                past_key_values=past_prefix,
                use_cache=True,
                output_attentions=True,
            )
    finally:
        cfg.attn_implementation = prev_attn_impl
    attn = getattr(out, "attentions", None)
    if not attn or len(attn) == 0:
        # Several HF fast paths return an empty tuple when ``past_key_values`` is set
        # (even with ``attn_implementation='eager'``); callers should fall back to key-norm.
        raise RuntimeError("Model returned no attentions with output_attentions=True")
    last = attn[-1]
    # [B, H, Q, S] typical — one query position for AR step
    w = last.float().mean(dim=1)  # [B, Q, S]
    w = w.mean(dim=0).squeeze(0)  # [S]
    return w


def build_full_length_scores_from_attention_prefix(
    prefix_scores: torch.Tensor,
    *,
    total_len: int,
) -> torch.Tensor:
    """Place ``prefix_scores`` (length ``total_len-1``) into a length-``total_len`` vector; last index → inf."""
    if total_len <= 0:
        return torch.zeros(0)
    device = prefix_scores.device
    dtype = prefix_scores.dtype
    out = torch.zeros(total_len, device=device, dtype=dtype)
    if total_len == 1:
        out[0] = torch.tensor(float("inf"), device=device, dtype=dtype)
        return out
    out[: total_len - 1] = prefix_scores
    out[total_len - 1] = torch.tensor(float("inf"), device=device, dtype=dtype)
    return out
