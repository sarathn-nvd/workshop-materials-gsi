"""In-process tool clients (Parquet-backed).

For the workshop demo we skip Postgres + Docker and run all five tools as
direct Python function calls against the on-disk artifacts. Same response
shape as the future FastAPI/Postgres deployment, so the runtime orchestrator
above this layer doesn't need to change when we promote to real services.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import threading
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from pipeline.common.typology_keywords import TYPOLOGY_KEYWORDS
from pipeline.config import (
    POOLS,
    SFT_TOOL_POLICY_CHUNKS,
    SFT_TOOL_SOPS_DIR,
    TOOL_1_DIR,
    TOOL_2_DIR,
)
from pipeline.schemas import (
    KYCProfile,
    PolicyExcerpt,
    SanctionsHit,
    SOPExcerpt,
    Transaction,
    Typology,
)

logger = logging.getLogger("pipeline.reference_agent.tool_clients")


# ============================================================================
# Tool 1 — transactions store
# ============================================================================
_tool1_lock = threading.Lock()
_tool1_df: pd.DataFrame | None = None


def _load_tool1() -> pd.DataFrame:
    global _tool1_df
    with _tool1_lock:
        if _tool1_df is None:
            path = TOOL_1_DIR / "transactions.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Tool 1 Parquet missing: {path}. Run step 4."
                )
            _tool1_df = pd.read_parquet(path)
            # Pre-parse dates for fast filtering
            _tool1_df["_date_parsed"] = pd.to_datetime(_tool1_df["date"], errors="coerce")
        return _tool1_df


def get_transactions(
    entity_id: str,
    window_start: str | date,
    window_end: str | date,
) -> list[Transaction]:
    """Returns transactions for `entity_id` within [window_start, window_end].

    Strips internal sidecars (source_pool, typology_tag) before returning —
    those are NEVER visible to the agent (MRULE-4-TYPOLOGY-TAG-INTERNAL).
    """
    df = _load_tool1()
    ws = pd.Timestamp(window_start)
    we = pd.Timestamp(window_end)
    mask = (df["entity_id"] == entity_id) & (df["_date_parsed"] >= ws) & (df["_date_parsed"] <= we)
    rows = df.loc[mask].to_dict("records")
    out: list[Transaction] = []
    for r in rows:
        out.append(Transaction(
            date=str(r["date"]),
            amount=float(r["amount"]),
            currency=str(r["currency"]),
            counterparty=str(r["counterparty"]),
            channel=str(r["channel"]),
            notes=str(r.get("notes", "") or ""),
        ))
    return out


# ============================================================================
# Tool 2 — KYC store
# ============================================================================
_tool2_lock = threading.Lock()
_tool2_index: dict[str, dict] | None = None


def _load_tool2() -> dict[str, dict]:
    global _tool2_index
    with _tool2_lock:
        if _tool2_index is None:
            path = TOOL_2_DIR / "entities.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Tool 2 Parquet missing: {path}. Run step 3.")
            df = pd.read_parquet(path)
            _tool2_index = {
                str(r["entity_id"]): r.to_dict() for _, r in df.iterrows()
            }
        return _tool2_index


class EntityNotFound(KeyError):
    pass


def get_kyc(entity_id: str) -> KYCProfile:
    idx = _load_tool2()
    row = idx.get(entity_id)
    if row is None:
        raise EntityNotFound(entity_id)
    # Drop sidecar fields (source_pool) before validating.
    d = {k: row[k] for k in KYCProfile.model_fields.keys() if k in row}
    return KYCProfile.model_validate(d)


# ============================================================================
# Tool 3 — Sanctions / PEP screen
# ============================================================================
_tool3_lock = threading.Lock()
_ofac_entries: list[dict] | None = None
_pep_entries: list[dict] | None = None


def _load_tool3() -> tuple[list[dict], list[dict]]:
    global _ofac_entries, _pep_entries
    with _tool3_lock:
        if _ofac_entries is None:
            _ofac_entries = _load_csv_simple(POOLS.ofac_targets)
        if _pep_entries is None:
            _pep_entries = _load_csv_simple(POOLS.pep_names)
        return _ofac_entries, _pep_entries


def _load_csv_simple(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("sanctions file missing: %s", path)
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(row)
    return out


def screen(name: str, country: str | None = None, *, min_score: float = 0.55) -> list[SanctionsHit]:
    """RapidFuzz token_set_ratio over OFAC + PEP entries.

    Returns hits with `match_score >= min_score`, sorted descending. Tags
    each hit with `list = "OFAC"` for OFAC pool, `"OpenSanctions"` for PEP.
    """
    if not name:
        return []
    ofac, peps = _load_tool3()
    hits: list[SanctionsHit] = []

    def _score(entry: dict, list_tag: str) -> SanctionsHit | None:
        # Source pool fields vary — accept several common name keys.
        cand = (entry.get("caption") or entry.get("name") or entry.get("Name") or "").strip()
        if not cand:
            return None
        score = fuzz.token_set_ratio(name, cand) / 100.0
        if country and entry.get("countries"):
            if country in entry["countries"]:
                score = min(1.0, score * 1.05)
        if score < min_score:
            return None
        return SanctionsHit(name=cand, list=list_tag, match_score=round(score, 3))

    for e in ofac:
        h = _score(e, "OFAC")
        if h is not None:
            hits.append(h)
    for e in peps:
        h = _score(e, "OpenSanctions")
        if h is not None:
            hits.append(h)
    hits.sort(key=lambda x: -x.match_score)
    return hits[:5]   # cap


# ============================================================================
# Tool 4 — Policy RAG
# ============================================================================
_tool4_lock = threading.Lock()
_chunks_df: pd.DataFrame | None = None


def _load_tool4() -> pd.DataFrame | None:
    global _chunks_df
    with _tool4_lock:
        if _chunks_df is None:
            if not SFT_TOOL_POLICY_CHUNKS.exists():
                logger.warning(
                    "Policy chunks Parquet missing: %s — Tool 4 will return empty",
                    SFT_TOOL_POLICY_CHUNKS,
                )
                return None
            _chunks_df = pd.read_parquet(SFT_TOOL_POLICY_CHUNKS)
        return _chunks_df


_TYPE_TO_SOURCES_PRIORITY = ["FinCEN", "FFIEC", "FATF", "OFAC"]


def retrieve(
    typology: Typology,
    k: int = 4,
    *,
    activity_descriptor: str | None = None,
) -> list[PolicyExcerpt]:
    """Stratified top-k retrieval across the four `source` enums."""
    if typology == "none":
        return []
    df = _load_tool4()
    if df is None or df.empty:
        return []
    # The chunks parquet has a `typology_tags` column (list of strings).
    if "typology_tags" not in df.columns:
        logger.warning("policy_chunks.parquet missing typology_tags; returning empty")
        return []
    # Filter rows matching this typology — typology_tags may be a numpy array
    # (from Parquet roundtrip) so we use explicit length check.
    def _has_typology(tags) -> bool:
        if tags is None:
            return False
        try:
            return typology in list(tags)
        except TypeError:
            return False
    mask = df["typology_tags"].apply(_has_typology)
    matching = df[mask]
    if matching.empty:
        return []

    # Stratify by source
    out: list[PolicyExcerpt] = []
    by_src = {s: matching[matching["source"] == s] for s in _TYPE_TO_SOURCES_PRIORITY}
    src_order = _TYPE_TO_SOURCES_PRIORITY * ((k // 4) + 1)
    for s in src_order:
        if len(out) >= k:
            break
        bucket = by_src.get(s)
        if bucket is None or bucket.empty:
            continue
        picked_idx = bucket.index[len(out) % len(bucket)]
        row = bucket.loc[picked_idx]
        out.append(PolicyExcerpt(
            source=str(row.get("source", "FinCEN")),
            section=str(row.get("section", "")[:200]),
            url=row.get("url") if pd.notna(row.get("url")) else None,
            text=str(row.get("text", ""))[:1500],
        ))
        by_src[s] = bucket.drop(picked_idx)
    return out[:k]


# ============================================================================
# Tool 5 — SOP service
# ============================================================================
_tool5_lock = threading.Lock()
_sop_cache: dict[str, list[tuple[str, str]]] | None = None


def _load_tool5() -> dict[str, list[tuple[str, str]]]:
    """Return mapping of (typology_upper, variant_n) -> list[(section_title, body)]."""
    global _sop_cache
    with _tool5_lock:
        if _sop_cache is None:
            _sop_cache = {}
            if not SFT_TOOL_SOPS_DIR.exists():
                logger.warning("SOPs directory missing: %s", SFT_TOOL_SOPS_DIR)
                return _sop_cache
            for md in SFT_TOOL_SOPS_DIR.glob("*.md"):
                # Filename pattern: <typology>_v<n>.md  e.g. structuring_v1.md
                m = re.match(r"^(?P<typ>[a-z_]+)_v(?P<n>\d+)\.md$", md.name)
                if not m:
                    continue
                key = (
                    f"SOP-{m.group('typ').upper().replace('_', '-')}-"
                    f"{int(m.group('n')):02d}"
                )
                txt = md.read_text(encoding="utf-8")
                sections = _split_sections(txt)
                _sop_cache[key] = sections
        return _sop_cache


def _split_sections(txt: str) -> list[tuple[str, str]]:
    parts = re.split(r"^##\s+(.+?)\s*$", txt, flags=re.MULTILINE)
    # parts[0] is preamble; pairs are (title, body)
    out: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        out.append((title, body))
    return out


_DEFAULT_SECTION_PRIORITY = [
    "Investigation Steps",
    "Escalation Criteria",
    "Documentation Requirements",
    "Filing Decision",
    "Tools and Systems",
    "References",
]


def sop(typology: Typology, *, variant: int = 1, section: str | None = None) -> list[SOPExcerpt]:
    """Return SOP excerpt(s) for a typology + variant."""
    if typology == "none":
        return []
    cache = _load_tool5()
    typ_token = typology.upper().replace("_", "-")
    sop_id = f"SOP-{typ_token}-{variant:02d}"
    sections = cache.get(sop_id, [])
    if not sections:
        return []
    if section:
        for title, body in sections:
            if section.lower() in title.lower():
                return [SOPExcerpt(sop_id=sop_id, section=title, text=body[:1500])]
        return []
    # Default: weighted-pick the highest-priority section that exists
    for pri in _DEFAULT_SECTION_PRIORITY:
        for title, body in sections:
            if title.lower() == pri.lower():
                return [SOPExcerpt(sop_id=sop_id, section=title, text=body[:1500])]
    title, body = sections[0]
    return [SOPExcerpt(sop_id=sop_id, section=title, text=body[:1500])]


# ============================================================================
# Convenience: invalidate caches (for tests / re-runs after data updates)
# ============================================================================
def reset_caches() -> None:
    global _tool1_df, _tool2_index, _ofac_entries, _pep_entries, _chunks_df, _sop_cache
    with _tool1_lock:
        _tool1_df = None
    with _tool2_lock:
        _tool2_index = None
    with _tool3_lock:
        _ofac_entries = None
        _pep_entries = None
    with _tool4_lock:
        _chunks_df = None
    with _tool5_lock:
        _sop_cache = None
