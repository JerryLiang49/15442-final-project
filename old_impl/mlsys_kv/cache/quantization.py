"""Symmetric INT8 quantization helpers for draft KV caches.

Scheme (per tensor, symmetric, zero-point free)
===============================================

For a floating tensor ``x`` (typically FP16/BF16 on GPU):

1. Convert to FP32 for stable reduction: ``xf = x.float()``.
2. ``amax = max(|xf|)`` over all elements (vanishes only for all-zero tensors).
3. **Scale** ``s = amax / 127``, lower-bounded by ``1e-8`` to avoid division by zero.
4. **Quantize** ``q = round(xf / s)`` clipped to ``[-128, 127]``, dtype ``int8``.
5. **Dequantize** ``x_hat = q.to(fp32) * s``, then cast to ``out_dtype`` on ``out_device``.

Each **key** tensor and **value** tensor gets its **own** scale (two scales per layer).

This is intentionally simple for course reports; INT4 or grouped scales can extend the same API later.
"""

from __future__ import annotations

import torch


def symmetric_quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric INT8 quantization.

    Args:
        x: Any floating dtype; not modified in-place.

    Returns:
        ``(q, scale)`` where ``q`` is ``int8`` on the **same device** as ``x``,
        and ``scale`` is a **0-dimensional float32 tensor** on **CPU** (storage accounting).
    """
    xf = x.detach().float()
    amax = xf.abs().max()
    scale_val = (amax / 127.0).clamp(min=1e-8)
    q = torch.round(xf / scale_val).clamp(-128, 127).to(torch.int8)
    scale_cpu = scale_val.cpu().to(torch.float32)
    return q, scale_cpu


def symmetric_dequantize_int8(
    q: torch.Tensor,
    scale_cpu: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    out_device: torch.device,
) -> torch.Tensor:
    """Map INT8 codes back to floating K/V for attention.

    Args:
        q: Quantized tensor (any device).
        scale_cpu: 0-dim float32 scale from :func:`symmetric_quantize_int8`.
        out_dtype: Target dtype (e.g. model activations FP16).
        out_device: Target device for the dequantized tensor.

    Returns:
        Dequantized tensor, same shape as ``q``.
    """
    s = scale_cpu.to(device=out_device, dtype=torch.float32)
    return (q.to(dtype=torch.float32) * s).to(dtype=out_dtype, device=out_device)
