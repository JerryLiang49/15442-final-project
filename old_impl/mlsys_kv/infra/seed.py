"""Reproducibility helpers for RNG state."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds where available.

    Also configures deterministic algorithms when ``TORCH_DETERMINISTIC=1`` (optional).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if os.environ.get("TORCH_DETERMINISTIC") == "1":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
