"""IO helpers: parquet/jsonl readers + writers + manifest writer.

Interim parquet for stage-to-stage handoff; jsonl for the final corpus and for
chat-SFT records. Manifests are pretty-printed JSON.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from pipeline.schemas import ChatSFTRecord, StageManifest

logger = logging.getLogger(__name__)


# ============================================================================
# Parquet (interim stage handoff)
# ============================================================================
def read_parquet(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Interim parquet missing: {path}")
    df = pd.read_parquet(path)
    # Normalize: parquet stores list-of-dict columns as numpy ndarrays; convert
    # back to plain Python lists so `... or []` and `if x:` patterns work
    # naturally downstream.
    df = _normalize_object_columns(df)
    logger.info("read_parquet(%s) → %d rows", path, len(df))
    return df


def _normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numpy-array values in object columns back to Python lists.

    pyarrow round-trips list[struct] columns as numpy.ndarray of dicts. This
    breaks `value or []` (numpy ambiguous-truth) and `bool(value)`. Walking
    object columns once at IO time is far cheaper than guarding every use.
    """
    import numpy as np  # local import keeps this file optional
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        sample = next((v for v in df[col].head(50) if v is not None), None)
        if isinstance(sample, np.ndarray):
            df[col] = df[col].apply(
                lambda v: v.tolist() if isinstance(v, np.ndarray) else (v if v is not None else None)
            )
    return df


def coerce_list(v) -> list:
    """Defensive coercion: anything iterable-and-not-None → list; else []."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        if hasattr(v, "tolist"):
            return v.tolist()
        return list(v)
    except (TypeError, ValueError):
        return []


def write_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("write_parquet(%s) ← %d rows", path, len(df))
    return path


# ============================================================================
# JSONL (final corpus + chat-SFT records)
# ============================================================================
def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Interim jsonl missing: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    logger.info("read_jsonl(%s) → %d records", path, len(rows))
    return rows


def iter_jsonl(path: Path | str) -> Iterator[dict]:
    """Streaming reader for very large jsonl files."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[dict | ChatSFTRecord], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, ChatSFTRecord):
                payload = rec.model_dump(mode="json")
            else:
                payload = rec
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    logger.info("write_jsonl(%s) ← %d records", path, n)
    return n


# ============================================================================
# Manifest writer
# ============================================================================
def write_manifest(manifest: StageManifest, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("write_manifest(%s) — status=%s, runtime=%.1fs",
                path, manifest.status, manifest.runtime_seconds)
    return path


def read_manifest(path: Path | str) -> StageManifest:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return StageManifest(**data)


# ============================================================================
# Generic JSON helpers
# ============================================================================
def write_json(obj: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


def read_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
