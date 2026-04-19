"""Practical roofline helpers: byte proxies for KV traffic (not hardware counters).

These are **analytical estimates** for comparing runs in CSV — not Nsight/Vendor HW metrics.
"""

from __future__ import annotations

from typing import Any


def estimate_attention_hist_kv_read_bytes_proxy(
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    hist_seq_len: int,
    kv_kernel_backend: str,
    bytes_per_fp16: int = 2,
) -> float:
    """Rough proxy for historical K/V **read** volume touched in one score pass over history.

    * **reference** — assumes FP16 K and V are read for matmul-style attention over ``hist`` positions.
    * **triton** — INT4-packed history: ~half the FP16 bytes for the same layout (upper+lower pathways
      still read packed storage; constant factors omitted).

    This is meant for **ratio** analysis (fused vs reference), not absolute GB/s roofline ceilings.
    """
    h = max(hist_seq_len, 0)
    if h <= 0:
        return 0.0
    # Per layer, per head: K[h,d] and V[h,d] visits across seq — O(h * d) reads for QK and softmax path.
    per_layer_fp16_reads = float(num_heads) * float(h) * float(head_dim) * 2.0 * float(bytes_per_fp16)
    total_fp16 = float(num_layers) * per_layer_fp16_reads
    kb = (kv_kernel_backend or "").strip().lower()
    if kb in ("triton", "cuda"):
        # Packed INT4 payload ~4 bits/weight for stored codes; dequant still touches scales/zp.
        return total_fp16 * 0.35
    return total_fp16


def traffic_proxy_bytes_per_output_token_from_row(
    *,
    baseline: str,
    logical_quant_store_bytes: Any,
    logical_verifier_kv_bytes: Any,
    full_seq_len: int,
) -> float | None:
    """Map logical footprint columns to a **per-output-token** proxy for plotting."""
    if full_seq_len <= 0:
        return None
    if baseline == "quant_spec":
        if logical_quant_store_bytes in ("", None):
            return None
        try:
            return float(logical_quant_store_bytes) / float(full_seq_len)
        except (TypeError, ValueError):
            return None
    if logical_verifier_kv_bytes in ("", None):
        return None
    try:
        return float(logical_verifier_kv_bytes) / float(full_seq_len)
    except (TypeError, ValueError):
        return None


def implied_bandwidth_gbps(
    tokens_per_sec_decode: float | None,
    bytes_per_token: float | None,
) -> float | None:
    """``tok/s × bytes/tok`` → GB/s (useful vs peak HBM on datasheets)."""
    if tokens_per_sec_decode is None or bytes_per_token is None:
        return None
    if tokens_per_sec_decode <= 0 or bytes_per_token <= 0:
        return None
    return (tokens_per_sec_decode * bytes_per_token) / 1e9
