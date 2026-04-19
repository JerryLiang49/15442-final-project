"""Load prompts from simple text files (one prompt per line)."""

from __future__ import annotations

from pathlib import Path


def load_prompts_file(path: str | Path) -> list[str]:
    """Return non-empty stripped lines from ``path``."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]
