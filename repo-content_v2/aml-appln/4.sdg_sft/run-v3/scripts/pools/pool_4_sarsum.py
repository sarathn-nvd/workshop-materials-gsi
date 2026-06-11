"""Pool_4 — SARSum: prose SAR narratives + decision labels.

Each SARSum row has a free-text `notes` array + a `decision` label. No
explicit typology — Stage 1 calls an LLM classifier to assign one.
"""
from __future__ import annotations

import json
import logging
from typing import Iterator

import pandas as pd

from scripts.config import POOLS

logger = logging.getLogger(__name__)


def _iter_jsonl() -> Iterator[dict]:
    path = POOLS.pool_4_sarsum
    if not path.exists():
        logger.warning("Pool_4 file missing: %s", path)
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


import re

# Decision label is embedded inside the notes text on a "Decision: ..." line,
# not provided as a separate field. Match permissively.
_DECISION_RE = re.compile(r"Decision\s*:\s*([A-Za-z][A-Za-z\s]*?)\.", re.IGNORECASE)


def _extract_decision(notes_text: str) -> str:
    """Extract the decision verdict from inside the notes prose."""
    if not notes_text:
        return ""
    m = _DECISION_RE.search(notes_text)
    return m.group(1).strip().lower() if m else ""


def load(max_rows: int | None = None) -> pd.DataFrame:
    """Load Pool_4 → DataFrame of prose SAR records.

    Columns:
      record_id, source, typology (None until LLM-classified), label,
      severity (None — Stage 3 LLM-extract assigns), notes (str),
      summaries (list[str]), kyc_seed (dict)

    SARSum schema notes:
      - `notes` is a LIST of strings — joined into a single prose blob here.
      - `decision` is NOT a separate field; it is embedded inside the notes
        as "Decision: Suspicious." or "Decision: Not suspicious.".
      - `summaries` is a list of pre-written SAR summaries (used by Stage 7
        as the canonical narrative for Record_4 if available; falls back to
        notes if absent).
    """
    out: list[dict] = []
    for i, payload in enumerate(_iter_jsonl()):
        md = payload.get("metadata") or {}
        row = (md.get("row_fields") or {}) if isinstance(md, dict) else {}

        # `notes` is a list of strings on disk
        notes_raw = row.get("notes") or payload.get("notes") or ""
        if isinstance(notes_raw, list):
            notes = "\n\n".join(str(n) for n in notes_raw if n)
        else:
            notes = str(notes_raw)

        # `summaries` may be a list of strings OR a list of dicts; project to list[str]
        summaries_raw = row.get("summaries") or payload.get("summaries") or []
        summaries: list[str] = []
        if isinstance(summaries_raw, list):
            for s in summaries_raw:
                if isinstance(s, dict):
                    txt = s.get("text") or s.get("summary") or ""
                    if txt:
                        summaries.append(str(txt))
                elif s:
                    summaries.append(str(s))

        # Decision is embedded inside notes prose, not in a separate field
        decision = _extract_decision(notes)
        # Per strategy §2A, Pool_4 (SARSum) is a positive-only path (Record_4).
        # We label True for "Suspicious" decisions and False otherwise; downstream
        # filters will keep only positives for Record_1/4 and only negatives for R5.
        label = decision.startswith("susp")

        out.append({
            "record_id": f"sarsum_{i:06d}",
            "source": "sarsum",
            "typology": None,                # Stage 1 LLM-classifies if needed
            "label": label,
            "severity": None,                # Stage 3 LLM-extract assigns from prose
            "notes": notes,
            "summaries": summaries,
            "decision_native": decision,     # for traceability
            "kyc_seed": {},                  # Stage 1/2 LLM-extract from prose
        })
        if max_rows and len(out) >= max_rows:
            break
    df = pd.DataFrame(out)
    if not df.empty:
        n_pos = int(df["label"].sum())
        n_with_summary = int(df["summaries"].apply(lambda s: bool(s)).sum())
        logger.info(
            "pool_4_sarsum.load → %d records (%d positive 'Suspicious', %d with "
            "non-empty summaries[])", len(df), n_pos, n_with_summary,
        )
    else:
        logger.info("pool_4_sarsum.load → 0 records")
    return df
