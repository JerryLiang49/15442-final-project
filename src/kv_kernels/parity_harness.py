"""Phase M — numerical parity helpers for fused Triton vs high-precision reference.

**Official tolerances (fused verifier attention output, FP32 accumulation)**

* Default ``rtol=1e-3``, ``atol=1e-3`` — matches historical :mod:`tests.test_fused_verifier_block`
  expectations for CUDA Triton vs PyTorch reference on random inputs.
* INT4/FP16 dequantization plus softmax weighting can accumulate small drift; these bounds are
  **validation gates for benchmarking**, not bitwise equality.

**Deterministic expectations**

* **Greedy token IDs** in end-to-end runs match between reference and fused paths when attention
  outputs are close enough that argmax logits agree. If logits differ near decision boundaries,
  sequences may diverge — treat as a failure to investigate, not a loosened tolerance.

**Supported conditions (fused verifier kernel)**

* Batch size ``B=1``, CUDA + Triton available, head dimension ``D`` and group sizes matching
  :func:`fused_verifier_block_attention` (see that module for layout).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class ParityTolerances:
    """Configurable ``torch.allclose`` thresholds."""

    rtol: float = 1e-3
    atol: float = 1e-3


DEFAULT_LAYERWISE_TOLERANCES = ParityTolerances(rtol=1e-3, atol=1e-3)


@dataclass
class TensorParityReport:
    """Metrics comparing ``candidate`` to ``reference`` (same shape / dtype semantics)."""

    allclose: bool
    max_abs: float
    max_rel: float
    rtol: float
    atol: float
    passed: bool
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reference_shape"] = list(self.reference_shape)
        d["candidate_shape"] = list(self.candidate_shape)
        return d


def max_relative_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """Largest ``|a-b| / max(|a|, |b|, eps)`` over elements (stable for small values)."""
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    a = reference.detach().float()
    b = candidate.detach().float()
    diff = (a - b).abs()
    scale = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.tensor(eps, device=a.device, dtype=a.dtype))
    return float((diff / scale).max().item())


def tensor_parity_report(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    tolerances: ParityTolerances | None = None,
) -> TensorParityReport:
    """Compute ``allclose``, max absolute error, and max relative error."""
    tol = tolerances or DEFAULT_LAYERWISE_TOLERANCES
    if reference.shape != candidate.shape:
        return TensorParityReport(
            allclose=False,
            max_abs=float("nan"),
            max_rel=float("nan"),
            rtol=tol.rtol,
            atol=tol.atol,
            passed=False,
            reference_shape=tuple(reference.shape),
            candidate_shape=tuple(candidate.shape),
            extra={"error": "shape_mismatch"},
        )

    ref_f = reference.detach().float().cpu()
    cand_f = candidate.detach().float().cpu()
    diff = (ref_f - cand_f).abs()
    max_abs = float(diff.max().item())
    max_rel = max_relative_error(ref_f, cand_f)

    close = torch.allclose(cand_f, ref_f, rtol=tol.rtol, atol=tol.atol)
    passed = bool(close)

    return TensorParityReport(
        allclose=bool(close),
        max_abs=max_abs,
        max_rel=max_rel,
        rtol=tol.rtol,
        atol=tol.atol,
        passed=passed,
        reference_shape=tuple(reference.shape),
        candidate_shape=tuple(candidate.shape),
    )


def assert_tensor_parity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    tolerances: ParityTolerances | None = None,
    context: str = "",
) -> TensorParityReport:
    """Raise ``AssertionError`` with a short report if parity fails."""
    rep = tensor_parity_report(reference, candidate, tolerances=tolerances)
    if rep.passed:
        return rep
    msg = (
        f"Parity failed{f' ({context})' if context else ''}: "
        f"max_abs={rep.max_abs:.6g} max_rel={rep.max_rel:.6g} "
        f"(rtol={rep.rtol} atol={rep.atol})"
    )
    raise AssertionError(msg)


def _tensor_summary(t: torch.Tensor, *, max_elems: int = 8) -> dict[str, Any]:
    t_cpu = t.detach().cpu()
    flat = t_cpu.flatten()
    n = min(int(flat.numel()), max_elems)
    return {
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "device": str(t_cpu.device),
        "sample_flat": flat[:n].tolist(),
    }


def triage_fused_verifier_mismatch(
    *,
    backend_triton: str,
    layer_idx: int | None = None,
    head_idx: int | None = None,
    query_row_t: int | None = None,
    gamma: int | None = None,
    s_hist: int | None = None,
    s_rec: int | None = None,
    group_size_k: int | None = None,
    group_size_v: int | None = None,
    # optional tensors for layout debugging
    k_uq_hist: torch.Tensor | None = None,
    k_lq_hist: torch.Tensor | None = None,
    k_scale_u: torch.Tensor | None = None,
    k_zp_u: torch.Tensor | None = None,
    k_block: torch.Tensor | None = None,
    q: torch.Tensor | None = None,
    report: TensorParityReport | None = None,
    tile_d_start: int = 0,
    tile_d_width: int = 16,
) -> dict[str, Any]:
    """Structured payload for CI logs when fused verifier parity fails (repro / debug).

    Includes small **byte-level** slices of packed INT4 codes (as integer lists) around ``tile_d_start``.
    """
    out: dict[str, Any] = {
        "kernel": "fused_verifier_block_attention",
        "backend_triton": backend_triton,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "query_row_t": query_row_t,
        "gamma": gamma,
        "s_hist": s_hist,
        "s_rec": s_rec,
        "group_size_k": group_size_k,
        "group_size_v": group_size_v,
    }
    if report is not None:
        out["parity"] = report.to_dict()

    def _slice_codes(t: torch.Tensor | None, head: int | None) -> dict[str, Any] | None:
        if t is None or head is None:
            return None
        th = t
        if th.dim() >= 3:
            th = th[head]
        if th.dim() != 2:
            return {"note": "unexpected_rank", "summary": _tensor_summary(t)}
        d0, d1 = th.shape
        d_end = min(d0, tile_d_start + tile_d_width)
        sl = th[tile_d_start:d_end, :]
        return {
            "tile_d_range": [tile_d_start, d_end],
            "nibbles_flat_sample": sl.flatten()[: 4 * tile_d_width].tolist(),
        }

    out["packed_k_upper_tile"] = _slice_codes(k_uq_hist, head_idx)
    out["packed_k_lower_tile"] = _slice_codes(k_lq_hist, head_idx)
    if k_scale_u is not None and head_idx is not None:
        out["k_scale_u_row0"] = _tensor_summary(k_scale_u[head_idx, 0])
    if k_zp_u is not None and head_idx is not None:
        out["k_zp_u_row0"] = _tensor_summary(k_zp_u[head_idx, 0])
    if k_block is not None and head_idx is not None:
        out["k_block_row0"] = _tensor_summary(k_block[head_idx : head_idx + 1, 0, :])
    if q is not None:
        out["q_summary"] = _tensor_summary(q)

    return out


def triage_json_dumps(obj: dict[str, Any]) -> str:
    """JSON for logging (no non-finite floats in tensor paths — already lists)."""
    return json.dumps(obj, indent=2, default=str)


def locate_worst_head_token(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[int, int, float]:
    """For ``[B,H,T,D]`` tensors, return ``(head, t, max_abs)`` at worst (mean over D) row."""
    if reference.shape != candidate.shape or reference.dim() != 4:
        raise ValueError("expected [B,H,T,D] with matching shapes")
    diff = (reference.detach().float() - candidate.detach().float()).abs().mean(dim=-1)
    flat = diff[0].reshape(-1)
    j = int(flat.argmax().item())
    h, t = reference.shape[1], reference.shape[2]
    head = j // t
    tt = j % t
    return head, tt, float(flat[j].item())


__all__ = [
    "DEFAULT_LAYERWISE_TOLERANCES",
    "ParityTolerances",
    "TensorParityReport",
    "assert_tensor_parity",
    "locate_worst_head_token",
    "max_relative_error",
    "tensor_parity_report",
    "triage_fused_verifier_mismatch",
    "triage_json_dumps",
]
