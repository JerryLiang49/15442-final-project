"""Environment and library metadata for structured logs."""

from __future__ import annotations

import platform
from typing import Any


def collect_env_metadata() -> dict[str, Any]:
    """Return best-effort metadata about the runtime (versions, OS)."""
    meta: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except Exception:
        meta["torch"] = None
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except Exception:
        meta["transformers"] = None
    return meta
