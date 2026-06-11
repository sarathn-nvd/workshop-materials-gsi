"""Shared helpers for all source downloaders.

Owns the canonical output directory layout, per-source stats tracking,
structured logging, and a small set of filesystem utilities.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "DATA_DIR",
    "RAW_DIR",
    "FINAL_DIR",
    "LOG_FILE",
    "STATS_FILE",
    "REPORT_FILE",
    "source_dir",
    "ensure_dir",
    "setup_logging",
    "StatsTracker",
    "SourceStats",
    "write_report",
    "human_bytes",
    "dir_size",
]


SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FINAL_DIR = DATA_DIR / "final"
LOG_FILE = SCRIPT_DIR / "output.log"
STATS_FILE = DATA_DIR / "phase_stats.json"
REPORT_FILE = SCRIPT_DIR / "data_download_output.md"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_dir(phase: str, source: str, layer: str | None = None) -> Path:
    """Return the canonical raw/<phase>/[<layer>/]<source>/ directory.

    phase: "cpt" | "sft"
    layer: "level_1" | "level_2" (only for CPT)
    """
    if phase == "cpt":
        if layer not in {"level_1", "level_2"}:
            raise ValueError(f"CPT requires layer='level_1' or 'level_2', got {layer!r}")
        return ensure_dir(RAW_DIR / phase / layer / source)
    if phase == "sft":
        return ensure_dir(RAW_DIR / phase / source)
    raise ValueError(f"Unknown phase: {phase!r}")


@dataclass
class SourceStats:
    source: str
    phase: str
    layer: str | None = None
    status: str = "pending"  # pending | ok | partial | failed | skipped
    files_written: int = 0
    bytes_written: int = 0
    records_kept: int = 0
    records_filtered_out: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float | None:
        if self.started_at and self.finished_at:
            return round(self.finished_at - self.started_at, 2)
        return None


class StatsTracker:
    """Thread-safe per-source statistics collector.

    Persist to data/phase_stats.json at the end of the run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, SourceStats] = {}

    def _key(self, source: str, phase: str, layer: str | None) -> str:
        return f"{phase}/{layer}/{source}" if layer else f"{phase}/{source}"

    def create(self, source: str, phase: str, layer: str | None = None) -> SourceStats:
        with self._lock:
            key = self._key(source, phase, layer)
            if key not in self._stats:
                self._stats[key] = SourceStats(source=source, phase=phase, layer=layer)
            return self._stats[key]

    @contextmanager
    def track(self, source: str, phase: str, layer: str | None = None) -> Iterator[SourceStats]:
        stats = self.create(source, phase, layer)
        stats.status = "running"
        stats.started_at = time.time()
        logger = logging.getLogger(source)
        try:
            yield stats
            if stats.status == "running":
                # Auto-downgrade ok -> partial when the source produced no
                # meaningful data. This guards against silent "filter
                # rejected everything", "all candidate URLs 404'd", or
                # "auth gate not approved" cases that otherwise hide as
                # status=ok. We use a small byte threshold (1 KB) instead
                # of strict zero so the rule still catches sources that
                # ended up with only a stub/index file or directory
                # metadata (e.g. an empty parquet directory weighing
                # 1 byte from filesystem inode overhead).
                if stats.files_written == 0 or stats.bytes_written < 1024:
                    stats.status = "partial"
                    stats.error = stats.error or (
                        f"completed without producing meaningful output "
                        f"(files={stats.files_written}, "
                        f"bytes={stats.bytes_written}) — source likely "
                        f"needs investigation (filter mismatch, auth "
                        f"failure, or empty upstream listing)"
                    )
                else:
                    stats.status = "ok"
        except Exception as e:
            stats.status = "failed"
            stats.error = f"{type(e).__name__}: {e}"
            logger.exception("Source %s failed: %s", source, e)
        finally:
            stats.finished_at = time.time()
            logger.info(
                "Source %s finished: status=%s files=%d bytes=%s duration=%ss",
                source, stats.status, stats.files_written,
                human_bytes(stats.bytes_written), stats.duration_sec,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {k: asdict(v) for k, v in self._stats.items()}

    def save(self) -> None:
        """Persist stats, merging with any pre-existing ``phase_stats.json``.

        Targeted re-runs (e.g. ``--tasks courtlistener``) only know about
        their own tasks. Without merging, each such run would clobber the
        full stats file. We load the previous payload first and overwrite
        only the source keys touched in this run.
        """
        ensure_dir(DATA_DIR)
        prior_sources: dict[str, Any] = {}
        if STATS_FILE.exists():
            try:
                prior = json.loads(STATS_FILE.read_text())
                prior_sources = prior.get("sources", {}) or {}
            except (OSError, json.JSONDecodeError):
                prior_sources = {}
        merged = {**prior_sources, **self.snapshot()}
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sources": merged,
        }
        STATS_FILE.write_text(json.dumps(payload, indent=2, default=str))


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with console + file handlers."""
    ensure_dir(SCRIPT_DIR)
    level = logging.DEBUG if verbose else logging.INFO

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet overly verbose libraries. ``fsspec`` in particular emits one DEBUG
    # line per HTTP byte-range read — for a single edgar-corpus shard this can
    # produce 30k+ lines and tens of MB in output.log.
    for noisy in (
        "urllib3", "httpx", "huggingface_hub", "datasets", "filelock",
        "fsspec", "fsspec.spec", "fsspec.http",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def human_bytes(n: int) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def write_report(tracker: StatsTracker) -> None:
    """Generate a human-readable markdown summary of the run."""
    rows: list[str] = []
    rows.append("# Step 1 — Data Download Report")
    rows.append("")
    rows.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_")
    rows.append("")

    snapshot = tracker.snapshot()
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for key, s in snapshot.items():
        phase = s["phase"] + (f"/{s['layer']}" if s.get("layer") else "")
        by_phase.setdefault(phase, []).append(s)

    for phase in sorted(by_phase):
        rows.append(f"## {phase}")
        rows.append("")
        rows.append("| Source | Status | Files | Size | Records Kept | Records Filtered | Duration (s) | Error |")
        rows.append("|---|---|---:|---:|---:|---:|---:|---|")
        for s in sorted(by_phase[phase], key=lambda x: x["source"]):
            err = (s.get("error") or "").replace("|", "\\|")[:80]
            rows.append(
                f"| `{s['source']}` | {s['status']} | {s['files_written']} | "
                f"{human_bytes(s['bytes_written'])} | {s['records_kept']} | "
                f"{s['records_filtered_out']} | {s.get('duration_sec') or ''} | {err} |"
            )
        rows.append("")

    REPORT_FILE.write_text("\n".join(rows))
