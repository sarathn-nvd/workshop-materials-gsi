"""Pool_5 — CFPB Consumer Complaint Database (filtered subset).

Source: `complaints_filtered.parquet` produced by the data-acquisition stage.
All rows are negative-labelled (label=False). Per strategy §2A, Record_6 is
the "negative (near-miss)" path — meaning the surface activity superficially
resembles a typology but the customer context defuses the suspicion.

Typology assignment (this loader): we apply a deterministic keyword mapper
over the complaint Issue / Sub-issue / Product / Consumer narrative fields
to assign the closest near-miss typology. Rows with no keyword signal stay
typology="none" (truly benign — no surface resemblance to any typology).
This lets Stage 5 retrieve relevant policy excerpts, Stage 6 generate
auxiliary findings, and Stage 1 set surface_pattern="near_miss" — all
consistent with strategy §2A's "negative (near-miss)" intent.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from pipeline.config import POOLS

logger = logging.getLogger(__name__)


# ============================================================================
# CFPB complaint → near-miss typology keyword mapper
# ============================================================================
# Order matters: earlier patterns win. Each pattern is matched against the
# concatenation of (Product, Issue, Sub-issue, Sub-product, Consumer narrative).
# All resulting records have label=False, so these typologies are *near-miss*
# signals — the surface activity resembles the typology, the customer's
# explanation in the complaint defuses it.
_TYPOLOGY_KEYWORD_RULES: list[tuple[str, re.Pattern]] = [
    (
        "elder_exploitation",
        re.compile(
            r"\b(elder|senior|elderly|retir(e|ee|ement)|nursing\s+home|"
            r"power\s+of\s+attorney|guardian|caregiver|grandparent|"
            r"old(er)?\s+(mother|father|parent|relative)|65\+?|"
            r"social\s+security|reverse\s+mortgage)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "trade_based_ml",
        re.compile(
            r"\b(letter\s+of\s+credit|trade\s+finance|invoice|import|export|"
            r"customs|shipping|freight|cargo|bill\s+of\s+lading|"
            r"international\s+(trade|commerce)|foreign\s+invoice)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "shell_company",
        re.compile(
            r"\b(shell\s+(company|corp|llc)|anonymous\s+(llc|company)|"
            r"beneficial\s+owner|nominee|holding\s+company|offshore|"
            r"BVI|cayman|panama|opaque\s+ownership)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "structuring",
        re.compile(
            r"\b(cash\s+deposit|cash\s+withdraw|atm\s+(deposit|withdraw)|"
            r"\$?\s*9[,.]?[0-9]{3}|just\s+below\s+(\$)?10[,.]?000|"
            r"under\s+(\$)?10[,.]?000|reporting\s+threshold|"
            r"split(ting)?\s+(deposit|withdraw|payment|transaction)|"
            r"multiple\s+(small\s+)?deposit|smurf|currency\s+transaction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "layering",
        re.compile(
            r"\b(wire\s+transfer|international\s+transfer|cross[\-\s]?border|"
            r"foreign\s+(wire|transfer)|rapid\s+movement|fund\s+movement|"
            r"crypto(currency)?|bitcoin|virtual\s+(currency|asset)|"
            r"multiple\s+account|round[\-\s]?trip|pass[\-\s]?through|"
            r"intermediary\s+(account|bank)|correspondent\s+bank)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "human_trafficking",
        re.compile(
            r"\b(traffick(ing|er)?|forced\s+labor|sex\s+work|"
            r"prostitution|exploit(ation|ed)\s+(worker|person|minor))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "terrorist_financing",
        re.compile(
            r"\b(terror(ism|ist)?|extremis(t|m)|sanctioned\s+(country|entity)|"
            r"OFAC|specially\s+designated|state\s+sponsor)\b",
            re.IGNORECASE,
        ),
    ),
]


def _classify_typology(text: str) -> str:
    """Apply keyword rules in order; first match wins. Empty text → none."""
    if not text:
        return "none"
    for typology, pat in _TYPOLOGY_KEYWORD_RULES:
        if pat.search(text):
            return typology
    return "none"


# Change 8: complaint patterns that reveal genuine suspicion (not "near-miss"
# benign). We drop these from the negative pool — the LLM judge flagged 30
# such records on smoke as `evidence_actually_suspicious`.
#
# Heuristic: complaints describing ≥2 sub-CTR cash deposits / unexplained
# wires from offshore / sanctioned counterparties / fraud-investigation
# context. Anything that a junior analyst might reasonably flag in real
# review.
_SUSPICIOUS_CFPB = re.compile(
    r"\b("
    r"\$?\s?9[,.\s]?[0-9]{3}\s+(?:deposit|withdrawal)|"     # $9,xxx deposits
    r"multiple\s+(?:sub[\-\s]?\$?10[,.]?000|\$?9[,.]?[0-9]{3})\s+(?:cash|deposit)|"
    r"deposits?\s+just\s+(?:below|under)\s+\$?10[,.]?000|"
    r"structuring|"
    r"laundering|"
    r"sanctions?\s+(?:list|hit|match)|"
    r"OFAC|"
    r"(?:wire|transfer)\s+to\s+(?:Iran|Cuba|North\s+Korea|Syria|Russia)|"
    r"investigated\s+for\s+(?:fraud|money\s+laundering)"
    r")\b",
    re.IGNORECASE,
)


def _looks_actually_suspicious(text: str) -> bool:
    """True if the CFPB complaint shows red flags that contradict 'benign'.

    Used to drop records from Record_5 negatives where the LLM judge would
    later re-classify the activity as suspicious.
    """
    return bool(_SUSPICIOUS_CFPB.search(text or ""))


def load(max_rows: int | None = None) -> pd.DataFrame:
    """Load Pool_5 → DataFrame.

    Columns (best-effort projection from the parquet):
      record_id, source, typology (keyword-classified near-miss), label
      (always False), severity, notes (consumer narrative), product, issue,
      company, date_received, kyc_seed (dict; State → US-XX)
    """
    path = POOLS.pool_5_cfpb
    if not path.exists():
        logger.warning("Pool_5 parquet missing: %s", path)
        return pd.DataFrame()

    df_raw = pd.read_parquet(path)
    if max_rows:
        df_raw = df_raw.head(max_rows)

    # Field names vary between CFPB exports; project permissively.
    def _col(df: pd.DataFrame, *candidates: str) -> pd.Series:
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([None] * len(df))

    notes = _col(df_raw, "Consumer complaint narrative", "narrative", "complaint_narrative").fillna("").astype(str)
    product = _col(df_raw, "Product", "product").fillna("").astype(str)
    sub_product = _col(df_raw, "Sub-product", "sub_product").fillna("").astype(str)
    issue = _col(df_raw, "Issue", "issue").fillna("").astype(str)
    sub_issue = _col(df_raw, "Sub-issue", "sub_issue").fillna("").astype(str)

    # Concatenate text fields for keyword classification (Issue/Sub-issue first
    # since they're more deterministic; narrative last since it can be noisy).
    classify_text = (issue + " " + sub_issue + " " + product + " " + sub_product + " " + notes)
    typology = classify_text.apply(_classify_typology)

    out = pd.DataFrame({
        "record_id": [f"cfpb_{i:06d}" for i in range(len(df_raw))],
        "source": "cfpb",
        "typology": typology,
        "label": False,
        "severity": "light",
        "notes": notes,
        "product": product,
        "issue": issue,
        "sub_issue": sub_issue,
        "company": _col(df_raw, "Company", "company"),
        "date_received": _col(df_raw, "Date received", "date_received"),
        "state": _col(df_raw, "State", "state"),
    })
    out["kyc_seed"] = out["state"].apply(
        lambda s: {"incorporation_jurisdiction": f"US-{s}" if pd.notna(s) and len(str(s)) == 2 else None}
    )
    # Strategy §2A: Record_6 is "negative (near-miss)" — every row should
    # surface-resemble some typology. Filter to keyword-classified rows so
    # Stage 5/6 can produce meaningful policy excerpts and findings, and
    # Stage 1's surface_pattern="near_miss" assignment is semantically
    # grounded. The "truly benign" (typology=none) negative training signal
    # comes from Record_5 IBM RANDOM patterns instead.
    typed = out[out["typology"] != "none"].reset_index(drop=True)

    # Change 8: drop complaints that show genuine suspicion (sub-CTR cash
    # deposits, OFAC mentions, laundering investigations, etc.). Otherwise
    # they'd ship as label=False but the LLM judge would re-classify them as
    # suspicious — contaminating the negative training cell.
    is_suspicious = typed["notes"].apply(_looks_actually_suspicious)
    n_dropped_suspicious = int(is_suspicious.sum())
    typed = typed[~is_suspicious].reset_index(drop=True)
    typology_dist = typed["typology"].value_counts().to_dict()
    logger.info(
        "pool_5_cfpb.load → %d typed near-miss records (%d total CFPB rows; "
        "%.1f%% retained; %d additional dropped as actually-suspicious) | "
        "typology dist: %s",
        len(typed), len(out), len(typed) / len(out) * 100 if len(out) else 0,
        n_dropped_suspicious, typology_dist,
    )
    return typed
