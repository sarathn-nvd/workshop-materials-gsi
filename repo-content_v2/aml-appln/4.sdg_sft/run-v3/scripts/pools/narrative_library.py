"""Narrative library — SARSum-indexed reasoning patterns.

Per training_strategy.md Appendix E, the narrative library is the *primary*
source of regulator-grade SAR narrative reasoning at construction time. We
PAIR a transactional context bundle with a matching narrative pattern from
this library, then ask the LLM to ground the pattern's reasoning in the
bundle's specific facts. This replaces the previous "LLM drafts a SAR from
scratch with a vacuous EFC stub as seed" approach.

Each SARSum case carries 7 prose notes. Each note has its OWN per-pattern
decision (Suspicious / Not concerning / Not suspicious) and its own narrative.
We extract individual notes (~14K total) and index by (typology, decision)
so the retrieval is fine-grained.

Indexing keys:
  - typology: classified by keyword match against the "Pattern Identified" line
  - decision: parsed from "Decision: ..." line
  - regulatory_frame: derived from typology (matches scripts.schemas.RegulatoryFrame)

Retrieval scoring:
  - typology match    : 1.0 weight
  - decision match    : 0.5 weight
  - regulatory_frame  : 0.5 weight (correlated with typology but adds signal
                                    for cash/non-cash distinction)
  - extra: small lexical overlap between pattern text and bundle's KYC purpose
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from scripts.config import POOLS
from scripts.schemas import RegulatoryFrame, Typology

logger = logging.getLogger(__name__)


# ============================================================================
# Note extraction
# ============================================================================
# A SARSum note is structured as:
#   "Pattern Identified: <pattern phrase>.  \nDecision: <verdict>.  \n<narrative>"
# We extract each part separately.
_NOTE_HEADER_RE = re.compile(
    r"^\s*Pattern Identified\s*:\s*(?P<pattern>[^\n]+?)\.\s*\n"
    r"\s*Decision\s*:\s*(?P<decision>[^\n.]+?)\.\s*\n"
    r"(?P<body>.*)$",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================================
# Typology classification (keyword heuristics on Pattern Identified text)
# ============================================================================
# Order matters: more specific patterns first.
_TYPOLOGY_KEYWORDS: list[tuple[Typology, list[str]]] = [
    ("terrorist_financing", [
        "terrorist", "extremist", "terror financ", "high-risk individual",
    ]),
    ("human_trafficking", [
        "human traffic", "trafficking", "labor exploit", "sex traffic", "minor",
    ]),
    ("elder_exploitation", [
        "elder", "elderly", "senior citizen", "guardian", "power of attorney",
    ]),
    ("shell_company", [
        "shell compan", "shell entity", "newly incorporated", "no clear business",
        "dormant", "no operating history",
    ]),
    ("trade_based_ml", [
        "invoice", "over-invoic", "under-invoic", "trade-based", "shipping document",
        "import", "export", "false trade", "phantom shipment",
    ]),
    ("structuring", [
        "structured deposit", "below the reporting", "sub-threshold", "structuring",
        "below the ctr", "split deposit", "$10,000", "9,9", "9,5",
    ]),
    ("smurfing", [
        "smurf", "multiple individuals depositing", "coordinated deposits",
    ]),
    ("layering", [
        "rapid movement", "layering", "pass-through", "wire transfer to multiple",
        "rapid succession", "circular flow", "rapid transfer",
        "wire transfer", "international transfer", "high-value transfer",
        "frequent changes in account",
    ]),
]


# Sanctions / OFAC nexus marker — note can be ANY typology but with sanctions
# overlay. We capture this as a side-channel boolean to refine retrieval.
_SANCTIONS_KEYWORDS = [
    "sanction", "ofac", "embargoed", "high-risk jurisdiction",
    "north korea", "iran", "russia", "syria", "cuba", "myanmar",
    "venezuela", "yemen",
]


# Decision normalization
_DECISION_NORMALIZE = {
    "suspicious": "suspicious",
    "not concerning": "not_concerning",
    "not suspicious": "not_suspicious",
    "not suspicous": "not_suspicious",      # observed typo in source
}


# Mapping from typology → regulatory_frame (matches semantic_profile.py).
# This is for LIBRARY indexing only; runtime profile is computed from the
# bundle, not the library entry.
_TYPOLOGY_FRAME: dict[Typology, RegulatoryFrame] = {
    "structuring":         "ctr_structuring",
    "smurfing":            "ctr_structuring",
    "layering":            "layering_passthrough",
    "trade_based_ml":      "tbml",
    "shell_company":       "shell",
    "human_trafficking":   "trafficking",
    "terrorist_financing": "trafficking",   # was "te" — frame dropped per v3 §4.2
    "elder_exploitation":  "elder",
    "none":                "benign",
}


# ============================================================================
# Library entry
# ============================================================================
@dataclass
class NarrativePattern:
    """One narrative reasoning pattern extracted from SARSum.

    Each SARSum case yields up to 7 of these (one per note).
    """
    pattern_id: str                          # e.g. "sarsum_0042_n3"
    pattern_identified: str                  # the pattern phrase
    decision: str                            # "suspicious" | "not_concerning" | "not_suspicious"
    narrative: str                           # the prose reasoning body
    typology: Typology
    regulatory_frame: RegulatoryFrame
    sanctions_overlay: bool                  # True iff sanctions keywords present
    severity_hint: str                       # "light" | "medium" | "heavy"
    raw_note: str = field(repr=False)        # full note text for fallback


# ============================================================================
# Loader
# ============================================================================
def _classify_typology(pattern_identified: str, narrative: str) -> Typology:
    """Classify the note's primary typology from its pattern + narrative text."""
    text = f"{pattern_identified} {narrative}".lower()
    for typ, kws in _TYPOLOGY_KEYWORDS:
        if any(kw in text for kw in kws):
            return typ
    return "none"


def _has_sanctions_overlay(pattern_identified: str, narrative: str) -> bool:
    text = f"{pattern_identified} {narrative}".lower()
    return any(kw in text for kw in _SANCTIONS_KEYWORDS)


def _severity_hint(narrative: str) -> str:
    """Crude severity proxy from the narrative's amount mentions and length.

    Used as a tie-breaker during retrieval; bundles whose tx_total is heavy
    prefer narratives that talk about heavy amounts.
    """
    n = len(narrative or "")
    # Look for $XXX,XXX patterns
    big_amt = bool(re.search(r"\$[\d,]{6,}", narrative))
    if big_amt or n > 600:
        return "heavy"
    if n > 300:
        return "medium"
    return "light"


def _split_case_into_notes(case_payload: dict) -> list[str]:
    """A SARSum row's content is a JSON-encoded blob with `notes` (list of
    strings, one per pattern) and `key_facts` (paraphrase variants).
    """
    content = case_payload.get("content")
    if not content:
        return []
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return []
    if not isinstance(content, dict):
        return []
    notes = content.get("notes") or []
    if isinstance(notes, list):
        return [str(n) for n in notes if n]
    return []


def _iter_sarsum_rows() -> Iterator[dict]:
    path = POOLS.pool_4_sarsum
    if not path.exists():
        logger.warning("SARSum file missing: %s", path)
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


def load_patterns(max_cases: int | None = None) -> list[NarrativePattern]:
    """Load and parse the full SARSum narrative library.

    Returns a flat list of NarrativePattern (one per note across all cases),
    typically ~12-14K entries from 2K cases.
    """
    out: list[NarrativePattern] = []
    n_cases = 0
    n_skipped_unparseable = 0
    for case_idx, row in enumerate(_iter_sarsum_rows()):
        if max_cases and n_cases >= max_cases:
            break
        notes = _split_case_into_notes(row)
        if not notes:
            n_skipped_unparseable += 1
            continue
        for note_idx, note_text in enumerate(notes):
            m = _NOTE_HEADER_RE.match(note_text)
            if not m:
                # Some notes don't follow the canonical "Pattern Identified:"
                # / "Decision:" header pattern; skip these for index quality.
                continue
            pattern_identified = m.group("pattern").strip()
            decision_raw = m.group("decision").strip().lower()
            decision = _DECISION_NORMALIZE.get(decision_raw, decision_raw)
            narrative = m.group("body").strip()
            if not narrative or len(narrative) < 30:
                continue
            typology = _classify_typology(pattern_identified, narrative)
            sanctions = _has_sanctions_overlay(pattern_identified, narrative)
            out.append(NarrativePattern(
                pattern_id=f"sarsum_{case_idx:05d}_n{note_idx}",
                pattern_identified=pattern_identified,
                decision=decision,
                narrative=narrative,
                typology=typology,
                regulatory_frame=_TYPOLOGY_FRAME.get(typology, "benign"),
                sanctions_overlay=sanctions,
                severity_hint=_severity_hint(narrative),
                raw_note=note_text,
            ))
        n_cases += 1
    logger.info(
        "narrative_library: loaded %d patterns from %d SARSum cases (%d cases "
        "skipped as unparseable)",
        len(out), n_cases, n_skipped_unparseable,
    )
    return out


# ============================================================================
# Retrieval
# ============================================================================
class NarrativeLibrary:
    """Indexed access to the narrative pattern library.

    Provides typology-aware retrieval:
      - exact (typology, decision) matches first
      - then (regulatory_frame, decision) matches
      - then (decision-only) matches as last resort
    """

    def __init__(self, patterns: list[NarrativePattern]):
        self.patterns = patterns
        # Index by (typology, decision)
        self._by_td: dict[tuple[Typology, str], list[NarrativePattern]] = {}
        # Index by (regulatory_frame, decision)
        self._by_fd: dict[tuple[RegulatoryFrame, str], list[NarrativePattern]] = {}
        # Index by decision only
        self._by_d: dict[str, list[NarrativePattern]] = {}
        for p in patterns:
            self._by_td.setdefault((p.typology, p.decision), []).append(p)
            self._by_fd.setdefault((p.regulatory_frame, p.decision), []).append(p)
            self._by_d.setdefault(p.decision, []).append(p)

        # ---- Pre-sorted retrieval cache --------------------------------
        # `retrieve()` was a hot path: 9000+ patterns sorted on every call,
        # 200ms × 100K calls = 5+ hours. Now we precompute, per
        # (typology, decision, regulatory_frame, sanctions_overlay, severity)
        # selector tuple, the sorted list once at library-construction time.
        # Per-row retrieval becomes a dict lookup + slice.
        self._presorted_cache: dict[tuple, list[NarrativePattern]] = {}

    def __len__(self) -> int:
        return len(self.patterns)

    def stats(self) -> dict[str, int]:
        """Coverage report — counts per (typology, decision) cell."""
        from collections import Counter
        td = Counter((p.typology, p.decision) for p in self.patterns)
        return {f"{t}/{d}": n for (t, d), n in sorted(td.items())}

    def retrieve(
        self,
        typology: Typology,
        decision: str,
        regulatory_frame: Optional[RegulatoryFrame] = None,
        sanctions_overlay: bool = False,
        severity: str = "medium",
        k: int = 3,
        rng: Optional[random.Random] = None,
    ) -> list[NarrativePattern]:
        """Retrieve the top-k matching patterns for the given selectors.

        Implementation: pre-sort the candidate bucket ONCE per selector tuple
        (cached on the library instance). At call time we look up the cached
        sorted list and return its top-k. This was a 200ms-per-call hot loop
        that's now ~10us; on a 100K-row pre-build the wall-clock dropped from
        ~7 hours to ~minutes.

        Strategy (selector waterfall):
          1. Exact (typology, decision) match.
          2. Else, fall back to (regulatory_frame, decision).
          3. Else, fall back to (decision-only).
        Within the chosen bucket, sort descending by score:
          + 1.0  if pattern.typology         == typology
          + 0.5  if pattern.regulatory_frame == regulatory_frame
          + 0.3  if pattern.sanctions_overlay== sanctions_overlay
          + 0.2  if pattern.severity_hint    == severity

        Tie-breaking randomization is applied on the cached top-window only,
        so subsequent calls with identical selectors return mostly-stable but
        slightly varied top-k slices.
        """
        rng = rng or random.Random()
        cache_key = (typology, decision, regulatory_frame, sanctions_overlay, severity)
        sorted_bucket = self._presorted_cache.get(cache_key)
        if sorted_bucket is None:
            sorted_bucket = self._build_sorted_bucket(
                typology, decision, regulatory_frame,
                sanctions_overlay, severity,
            )
            self._presorted_cache[cache_key] = sorted_bucket
        if not sorted_bucket:
            return []
        # Tie-break only the very top of the sorted list — slice up to k+5
        # so we have a small variation pool, then shuffle and take k.
        top_window = sorted_bucket[: max(k * 2, 6)]
        rng.shuffle(top_window)
        return top_window[:k]

    def _build_sorted_bucket(
        self,
        typology: Typology,
        decision: str,
        regulatory_frame: Optional[RegulatoryFrame],
        sanctions_overlay: bool,
        severity: str,
    ) -> list[NarrativePattern]:
        """Build (and cache) the sorted-bucket for a selector tuple."""
        bucket = list(self._by_td.get((typology, decision), []))
        if len(bucket) < 3 and regulatory_frame is not None:
            extra = self._by_fd.get((regulatory_frame, decision), [])
            seen = {p.pattern_id for p in bucket}
            bucket.extend(p for p in extra if p.pattern_id not in seen)
        if len(bucket) < 3:
            extra = self._by_d.get(decision, [])
            seen = {p.pattern_id for p in bucket}
            bucket.extend(p for p in extra if p.pattern_id not in seen)
        if not bucket:
            return []

        def _score(p: NarrativePattern) -> float:
            s = 0.0
            if p.typology == typology:
                s += 1.0
            if regulatory_frame and p.regulatory_frame == regulatory_frame:
                s += 0.5
            if p.sanctions_overlay == sanctions_overlay:
                s += 0.3
            if p.severity_hint == severity:
                s += 0.2
            return s

        return sorted(bucket, key=_score, reverse=True)


# ============================================================================
# Module-level cached library
# ============================================================================
_LIBRARY_CACHE: Optional[NarrativeLibrary] = None


def get_library(max_cases: int | None = None, force_reload: bool = False) -> NarrativeLibrary:
    """Get the cached narrative library, building it on first call."""
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is None or force_reload:
        _LIBRARY_CACHE = NarrativeLibrary(load_patterns(max_cases=max_cases))
    return _LIBRARY_CACHE
