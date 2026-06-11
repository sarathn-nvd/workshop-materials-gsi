"""Stage 5A artifact builder — `policy_chunks.parquet`.

Walks the 9 policy jsonl files, splits each record's text into ≈800-token
chunks (≈3200 chars) with 10% overlap, tags each chunk with all matching
typologies via a keyword dictionary, writes one parquet.

Run once:
    python -m scripts.tools_prep.build_policy_chunks
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from pipeline.common.io import write_parquet
from pipeline.common.parallel import parmap, safe_workers
from pipeline.common.progress import configure_logging
from pipeline.config import CONCURRENCY, TOOLS_POLICY_CHUNKS
from pipeline.pools import policy_corpus

logger = configure_logging("tools_prep.build_policy_chunks")

CHUNK_TARGET_CHARS = 3200       # ≈ 800 tokens
CHUNK_OVERLAP_CHARS = 320       # 10% overlap


# Per-typology keyword dictionary (Stage 5A Step 1.3)
TYPOLOGY_KEYWORDS: dict[str, list[str]] = {
    "structuring": [
        "structuring", "structure", "sub-CTR", "sub-threshold", "currency transaction report",
        "$10,000", "5324", "FIN-2014-A005", "split", "deposits below",
    ],
    "smurfing": [
        "smurfing", "smurf", "multiple actors", "scatter", "fan-in",
        "sub-threshold deposits", "5324",
    ],
    "layering": [
        "layering", "peeling", "peel chain", "nested", "intermediary accounts",
        "FATF Recommendation 10", "FATF Recommendation 11", "5318(g)", "cycle",
    ],
    "trade_based_ml": [
        "trade-based", "TBML", "over-invoicing", "under-invoicing", "phantom shipments",
        "trade misinvoicing", "import-export", "5318(g)",
    ],
    "shell_company": [
        "shell company", "shell entity", "beneficial owner", "beneficial ownership",
        "1010.230", "BVI", "Cayman", "Panama", "nominee director",
    ],
    "human_trafficking": [
        "human trafficking", "trafficking", "FIN-2014-A008", "5318(g)",
        "labor exploitation", "victims",
    ],
    "terrorist_financing": [
        "terrorist financing", "terrorism", "2339B", "IEEPA", "1705",
        "designated", "material support", "OFAC", "SDN",
    ],
    "elder_exploitation": [
        "elder financial exploitation", "elder abuse", "FIN-2022-A002",
        "senior", "retiree", "fiduciary", "5318(g)",
    ],
}


def _split_text(text: str, target: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    text = re.sub(r"\s+\n", "\n", text or "").strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + target // 2, end)
            if nl > start:
                end = nl
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def _tags_for(text: str) -> list[str]:
    lower = (text or "").lower()
    tags = []
    for typology, kws in TYPOLOGY_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in lower:
                tags.append(typology)
                break
    return tags


def _process_file(path: Path) -> list[dict]:
    """Process one policy jsonl into a list of chunk rows."""
    src_label = policy_corpus.source_for(path)
    rows: list[dict] = []
    for i, rec in enumerate(policy_corpus.iter_records(path)):
        text = rec.get("content") or rec.get("text") or ""
        if not isinstance(text, str):
            continue
        meta = rec.get("metadata") or {}
        section = (meta.get("title") or rec.get("doc_id") or f"{path.stem}_{i}")[:200]
        url = meta.get("url") or ""
        for c_idx, chunk in enumerate(_split_text(text)):
            rows.append({
                "chunk_id": f"{path.stem}_{i:06d}_{c_idx:03d}",
                "source": src_label,
                "section": section,
                "url": url,
                "text": chunk,
                "typology_tags": _tags_for(chunk),
                "src_file": path.name,
            })
    logger.info("%s → %d chunks", path.name, len(rows))
    return rows


def build() -> Path:
    files = policy_corpus.list_files()
    if not files:
        raise FileNotFoundError("No policy corpus files found on disk.")
    workers = safe_workers(CONCURRENCY.max_cpu_workers, files)
    nested = parmap(_process_file, files, workers=workers, desc="policy_chunks")
    flat = [row for sublist in nested for row in sublist]
    df = pd.DataFrame(flat)
    write_parquet(df, TOOLS_POLICY_CHUNKS)
    logger.info("Wrote %d chunks → %s", len(df), TOOLS_POLICY_CHUNKS)
    # Per-typology depth sanity check
    if "typology_tags" in df.columns:
        depth: dict[str, int] = {}
        for _, row in df.iterrows():
            for t in row["typology_tags"]:
                depth[t] = depth.get(t, 0) + 1
        for t in TYPOLOGY_KEYWORDS:
            n = depth.get(t, 0)
            level = logging.INFO if n >= 10 else logging.WARNING
            logger.log(level, "  typology=%s tagged-chunks=%d %s", t, n, "OK" if n >= 10 else "BELOW-FLOOR")
    return TOOLS_POLICY_CHUNKS


if __name__ == "__main__":
    build()
