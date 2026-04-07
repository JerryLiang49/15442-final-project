"""MT-Bench-style prompt subset loading (fixed JSON for reproducible Phase 8 sweeps)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MBPrompt:
    """Single turn from the curated subset."""

    id: str
    category: str
    text: str


def load_mt_bench_subset(path: str | Path) -> list[MBPrompt]:
    """Load prompts from JSON array of ``{"id","category","text"}`` objects."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list in {p}")
    out: list[MBPrompt] = []
    for i, row in enumerate(raw):
        if isinstance(row, str):
            out.append(MBPrompt(id=f"line-{i}", category="unknown", text=row))
            continue
        if not isinstance(row, dict):
            raise ValueError(f"Invalid row {i} in {p}")
        pid = str(row.get("id", f"row-{i}"))
        cat = str(row.get("category", "unknown"))
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        out.append(MBPrompt(id=pid, category=cat, text=text))
    return out
