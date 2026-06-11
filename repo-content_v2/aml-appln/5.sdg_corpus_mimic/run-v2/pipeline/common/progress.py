"""Progress + structured logging helpers.

Logs go to stdout (human-readable) and to per-stage log files under
`data/logs/` as JSON-lines (machine-readable for post-run analysis).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pipeline.config import LOGS_DIR


# ============================================================================
# Logger setup
# ============================================================================
class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_LOGGERS_CONFIGURED: set[str] = set()


def configure_logging(stage_id: str, *, level: int = logging.INFO) -> logging.Logger:
    """One log file per stage, plus mirroring to stdout. Idempotent per stage_id."""
    logger = logging.getLogger(stage_id)
    if stage_id in _LOGGERS_CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.handlers.clear()

    # File handler (JSON lines)
    log_path = LOGS_DIR / f"{stage_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(_JsonLineFormatter())
    logger.addHandler(fh)

    # Stdout handler (human-readable)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(
        f"[%(asctime)s] [{stage_id}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(sh)

    logger.propagate = False
    _LOGGERS_CONFIGURED.add(stage_id)
    return logger


# ============================================================================
# Stage timer (for manifest)
# ============================================================================
class StageTimer:
    """Context-manager timer; exposes start_iso / end_iso / elapsed."""

    def __init__(self) -> None:
        self.start_ts: float = 0.0
        self.end_ts: float = 0.0
        self.start_iso: str = ""
        self.end_iso: str = ""

    def __enter__(self) -> "StageTimer":
        self.start_ts = time.time()
        self.start_iso = datetime.now(timezone.utc).isoformat()
        return self

    def __exit__(self, *_exc) -> None:
        self.end_ts = time.time()
        self.end_iso = datetime.now(timezone.utc).isoformat()

    @property
    def elapsed(self) -> float:
        return (self.end_ts or time.time()) - self.start_ts


# ============================================================================
# Banner helpers
# ============================================================================
@contextmanager
def stage_banner(stage_id: str, total_records: int) -> Iterator[None]:
    """Print a clear visual delimiter at the start and end of a stage."""
    bar = "=" * 78
    print(f"\n{bar}\n[{stage_id}] start — N={total_records}\n{bar}", flush=True)
    t0 = time.time()
    try:
        yield
    finally:
        print(f"[{stage_id}] end — elapsed {time.time() - t0:.1f}s", flush=True)
