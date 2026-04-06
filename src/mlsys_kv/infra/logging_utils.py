"""Console logging plus JSONL run logs for experiments."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    """Return a unique run id (time-ordered prefix + uuid4)."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"


@dataclass
class RunLogger:
    """Structured JSONL logging alongside standard console logs.

    Each call to :meth:`log_event` appends one JSON object to the run's JSONL file.
    """

    run_id: str
    jsonl_path: Path
    console: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "console", logging.getLogger(f"mlsys_kv.run.{self.run_id}"))
        self.console.handlers.clear()
        self.console.setLevel(logging.INFO)
        self.console.propagate = False
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        self.console.addHandler(h)

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: str, **fields: Any) -> None:
        """Append one JSON record and echo a short line to the console."""
        record: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.console.info("[%s] %s", event, {k: fields[k] for k in sorted(fields)})

    def log_exception(self, exc: BaseException, context: Mapping[str, Any] | None = None) -> None:
        """Log an exception with optional context fields."""
        payload: dict[str, Any] = {
            "exc_type": type(exc).__name__,
            "exc_str": str(exc),
        }
        if context:
            payload.update(context)
        self.log_event("exception", **payload)


@dataclass
class RunLogContext:
    """Context manager that initializes ``RunLogger`` and writes ``run_start`` / ``run_end`` markers."""

    output_dir: Path
    run_id: str | None = None
    environment: Mapping[str, Any] | None = None

    logger: RunLogger | None = field(default=None, init=False)

    def __enter__(self) -> RunLogger:
        rid = self.run_id or new_run_id()
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        jsonl_path = out / f"run_{rid}.jsonl"
        self.logger = RunLogger(run_id=rid, jsonl_path=jsonl_path)
        self.logger.log_event("run_start", output_dir=str(out), environment=dict(self.environment or {}))
        return self.logger

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self.logger is not None
        if exc is not None:
            self.logger.log_exception(exc)
        self.logger.log_event("run_end", ok=exc is None)
