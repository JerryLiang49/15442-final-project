"""Device resolution for single-GPU runs."""

from __future__ import annotations

import os

import torch


def resolve_device(device: str) -> torch.device:
    """Return a :class:`torch.device` from a user string.

    Args:
        device: ``"auto"`` (CUDA if available else CPU), ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
    """
    d = (device or "auto").strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def device_metadata(device: torch.device) -> dict[str, str | None]:
    """Return lightweight device metadata for JSON logs (best effort)."""
    meta: dict[str, str | None] = {
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            meta["gpu_name"] = torch.cuda.get_device_name(device)
        except Exception:
            meta["gpu_name"] = None
    return meta
