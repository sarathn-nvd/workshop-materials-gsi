"""Semantic-profile computation — pure Python, no LLM.

Combines `classify_typology` + behavioral metrics + sanctions-override
rules into the `SemanticProfile` block. The profile is used INTERNALLY
by the orchestrator to:

  1. Route Tool 4 (policy excerpt retrieval) — keyed on `typology_inferred`.
  2. Route Tool 5 (SOP lookup) — keyed on `typology_inferred`.
  3. Decide which statute and which typology-specific numeric question to
     feed the four auxiliary skill calls (Phase 2 of the workflow).
  4. Persist in the per-case `CaseTrace` metadata for human review +
     post-hoc scoring (`MRULE-N-CLASSIFIER-COVERAGE`).

It is **NOT** injected into the SAR judgment user message. The v1 / v2
backend lifted `_regulatory_frame`, `_typology_inferred`, and
`_decision_target` into the user message — that was the deterministic
label leak v3.1 SFT explicitly removed (per
`4.sdg_sft/run-v3/SDG_STRATEGY_SFT.md` §2.1 Rule A +
`5.sdg_corpus_mimic/run-v2/AGENT_USAGE_GUIDE.md` §3.5). The v3.1
trained model derives typology + verdict from the bundle evidence alone.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from aml_app.common.behavioral_features import compute_behavioral_metrics, normalize_channel
from aml_app.common.schemas import (
    KYCProfile, RegulatoryFrame, SanctionsHit, SemanticProfile,
    Transaction, Typology,
)


# ---------------------------------------------------------------------------
# Typology → regulatory-frame mapping (one-way; profile holds both)
# ---------------------------------------------------------------------------
_TYPOLOGY_FRAME_HINT: "dict[Typology, RegulatoryFrame]" = {
    "structuring":         "ctr_structuring",
    "smurfing":            "ctr_structuring",
    "layering":            "layering_passthrough",
    "trade_based_ml":      "tbml",
    "shell_company":       "shell",
    "human_trafficking":   "trafficking",
    "terrorist_financing": "trafficking",   # FFIEC + FATF group these
    "elder_exploitation":  "elder",
    "none":                "benign",
}


# ---------------------------------------------------------------------------
# Common-name PEP noise filter — keeps OpenSanctions fuzzy collisions out
# of the override / classifier-trigger logic but still surfaces them in
# the bundle for analyst review.
# ---------------------------------------------------------------------------
_COMMON_FIRST_NAMES = frozenset({
    "Amy", "Adam", "Anna", "Eric", "Gary", "Jack", "John", "Jose",
    "Lisa", "Mark", "Mary", "Paul", "Ruth", "Ryan", "Aaron", "Betty",
    "Brian", "Carol", "David", "Debra", "Diane", "Donna", "Emily",
    "Frank", "Helen", "Henry", "Jacob", "James", "Janet", "Jason",
    "Jerry", "Karen", "Kevin", "Larry", "Laura", "Linda", "Nancy",
    "Peter", "Sarah", "Scott", "Susan", "Tyler", "Amanda", "Andrew",
    "Daniel", "Dennis", "Donald", "Edward", "George", "Joshua",
    "Justin", "Martha", "Nathan", "Olivia", "Pamela", "Robert",
    "Ronald", "Samuel", "Sandra", "Sharon", "Steven", "Anthony",
    "Barbara", "Brandon", "Cynthia", "Deborah", "Dorothy", "Douglas",
    "Gregory", "Jeffrey", "Kenneth", "Matthew", "Melissa", "Michael",
    "Patrick", "Raymond", "Rebecca", "Stephen", "Timothy", "William",
    "Zachary", "Benjamin", "Jennifer", "Jonathan", "Kimberly",
    "Margaret", "Michelle", "Nicholas", "Patricia", "Virginia",
    "Alexander", "Catherine", "Christine", "Elizabeth", "Stephanie",
    "Christopher",
})


def is_common_name_pep(hit: dict | None) -> bool:
    """A PEP-list hit on a generic two-token First-Last name pattern.

    Returns True if the hit looks like the OpenSanctions common-name noise
    that swamps the demo corpus. Such hits are kept in the bundle but
    excluded from override / classifier-trigger logic.
    """
    if not hit or hit.get("list") != "OpenSanctions":
        return False
    name = (hit.get("name") or "").strip()
    parts = name.split()
    if len(parts) != 2:
        return False
    return parts[0] in _COMMON_FIRST_NAMES


def filter_confident_hits(hits: Iterable[dict] | None,
                          min_score: float = 0.9) -> list[dict]:
    """Return only hits that are both high-score AND not common-name PEPs."""
    out: list[dict] = []
    for h in (hits or []):
        if isinstance(h, SanctionsHit):
            h = h.model_dump()
        if not isinstance(h, dict):
            continue
        if float(h.get("match_score", 0) or 0) < min_score:
            continue
        if is_common_name_pep(h):
            continue
        out.append(h)
    return out


def _channel_mix(txs: list[dict]) -> dict[str, float]:
    if not txs:
        return {}
    c = Counter(t.get("channel", "") for t in txs)
    n = len(txs)
    return {k: round(v / n, 4) for k, v in c.items()}


# ---------------------------------------------------------------------------
# Main entry — compose the SemanticProfile
# ---------------------------------------------------------------------------
def compute_semantic_profile(
    transactions: Iterable | None,
    kyc_profile: dict | KYCProfile | None,
    source_typology: Typology,
    sanctions_pep_hits: Iterable | None = None,
) -> SemanticProfile:
    """Compute the cross-stage semantic profile for a bundle."""
    txs: list[dict] = []
    for t in (transactions or []):
        if isinstance(t, Transaction):
            d = t.model_dump()
        elif isinstance(t, dict):
            d = dict(t)
        else:
            continue
        d["channel"] = normalize_channel(str(d.get("channel", "")))
        txs.append(d)

    if isinstance(kyc_profile, KYCProfile):
        kyc = kyc_profile.model_dump()
    elif isinstance(kyc_profile, dict):
        kyc = dict(kyc_profile)
    else:
        kyc = {}

    hits: list[dict] = []
    for h in (sanctions_pep_hits or []):
        if isinstance(h, SanctionsHit):
            hits.append(h.model_dump())
        elif isinstance(h, dict):
            hits.append(dict(h))

    channel_mix = _channel_mix(txs)
    cash_present = channel_mix.get("cash", 0) > 0

    typology_inferred = source_typology
    regulatory_frame = _TYPOLOGY_FRAME_HINT.get(source_typology, "benign")

    # Channel coherence: cash-present cases that aren't structuring/smurfing
    # are layering pass-throughs in the v3.1 frame taxonomy.
    if source_typology not in ("structuring", "smurfing") and cash_present:
        typology_inferred = "layering"
        regulatory_frame = "layering_passthrough"

    # Confident sanctions hit overrides everything else.
    if filter_confident_hits(hits, min_score=0.9):
        regulatory_frame = "sanctions"

    metrics = compute_behavioral_metrics(txs, kyc)

    ratio = metrics.vs_declared_volume_ratio
    if ratio == 0:
        declared_volume_band = "match"
    elif ratio < 0.5:
        declared_volume_band = "under"
    elif ratio <= 1.5:
        declared_volume_band = "match"
    else:
        declared_volume_band = "over"

    if metrics.country_risk_max >= 0.65:
        geo_risk = "high"
    elif metrics.country_risk_max >= 0.3:
        geo_risk = "medium"
    else:
        geo_risk = "low"

    return SemanticProfile(
        channel_mix=channel_mix,
        cash_present=cash_present,
        regulatory_frame=regulatory_frame,
        declared_volume_band=declared_volume_band,
        geo_risk=geo_risk,
        typology_inferred=typology_inferred,
    )


# ---------------------------------------------------------------------------
# Decision-target helpers (retained for back-compat; not in 7-key bundle)
# ---------------------------------------------------------------------------
def decision_target_from(typology_inferred: Typology) -> str:
    """Naive decision_target derivation (typology=none -> not_suspicious).

    Kept for back-compat / tests. Production callers should prefer
    :func:`derive_decision_target_calibrated`, which uses the full
    signal context and matches how the SFT teacher labelled the
    training corpus.
    """
    if typology_inferred == "none":
        return "not_suspicious"
    return "suspicious"


def _has_aux_red_flag(aux: dict | None) -> bool:
    """Return True if any of the four auxiliary findings raised a concern.

    The gate stores each aux task as either a single dict or a
    ``list[dict]`` (one entry per accepted finding); both shapes are
    supported here. Conservative: any non-empty ``red_flags`` /
    ``inconsistency`` / ``evidence_rows`` / ``anomalies`` / ``concerns``
    field counts as a flag, as does an explicit ``is_anomalous`` /
    ``flag`` / ``is_suspicious`` boolean True.
    """
    if not isinstance(aux, dict):
        return False
    RED_LIST_FIELDS = ("red_flags", "inconsistency", "evidence_rows",
                        "anomalies", "concerns")
    RED_BOOL_FIELDS = ("is_anomalous", "flag", "is_suspicious")
    for task in ("behavioral", "numeric", "citation", "statutory"):
        items = aux.get(task)
        if items is None:
            continue
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            for k in RED_LIST_FIELDS:
                v = it.get(k)
                if v and (isinstance(v, list) and len(v) > 0
                           or (isinstance(v, str) and v.strip())):
                    return True
            for k in RED_BOOL_FIELDS:
                if it.get(k) is True:
                    return True
    return False


def _has_structural_red_flag(
    semantic_profile: SemanticProfile,
    sanctions_pep_hits: list[dict] | None,
    behavioral_metrics: dict | None,
) -> bool:
    """High-confidence positive signal that should force `suspicious`."""
    if filter_confident_hits(sanctions_pep_hits or [], min_score=0.9):
        return True
    if not isinstance(behavioral_metrics, dict):
        return False
    if behavioral_metrics.get("peel_chain_detected") is True:
        return True
    if (float(behavioral_metrics.get("counterparty_concentration_top1", 0)) >= 0.8
            and int(behavioral_metrics.get("transaction_count", 0)) >= 8):
        return True
    if behavioral_metrics.get("escalation_pattern") is True:
        return True
    if (behavioral_metrics.get("round_amount_share", 0) >= 0.8
            and semantic_profile.cash_present):
        return True
    return False


def _has_clean_signals(
    semantic_profile: SemanticProfile,
    kyc: dict,
    sanctions_pep_hits: list[dict] | None,
    behavioral_metrics: dict | None,
) -> bool:
    """All of the conditions for a confident-clean verdict.

    Every clause is conservative: we only return True when the case
    looks unambiguously benign on every axis we have a signal for.
    """
    if kyc.get("risk_rating") == "high":
        return False
    if filter_confident_hits(sanctions_pep_hits or [], min_score=0.9):
        return False
    if semantic_profile.regulatory_frame == "sanctions":
        return False
    if semantic_profile.declared_volume_band == "over":
        return False
    if semantic_profile.geo_risk == "high":
        return False
    if not isinstance(behavioral_metrics, dict):
        return True
    if behavioral_metrics.get("peel_chain_detected") is True:
        return False
    if behavioral_metrics.get("escalation_pattern") is True:
        return False
    if float(behavioral_metrics.get("counterparty_concentration_top1", 0)) >= 0.8:
        return False
    return True


def derive_decision_target_calibrated(
    semantic_profile: SemanticProfile,
    kyc: dict,
    sanctions_pep_hits: list[dict] | None = None,
    behavioral_metrics: dict | None = None,
    auxiliary_findings: dict | None = None,
) -> str | None:
    """Calibrated derivation of `_decision_target` for the SAR bundle.

    Returns one of three values:

    * ``"not_suspicious"`` when independent clean signals align. The
      trained model treats this as a hard floor (SFT corpus: 6712/6712
      cases with this hint output ``is_suspicious=False``).
    * ``"suspicious"`` when a confirmed structural red flag is present
      (high-confidence sanctions hit, peel-chain, escalation pattern,
      etc.). The trained model honored this hint in ~84% of SFT cases
      and overruled it (down to ``False``) in the remaining ~16%, so
      this is a soft positive prior — not a hard ceiling.
    * ``None`` for the ambiguous middle, so the bundle omits the hint
      and the model judges on its own.

    The caller decides what to do with ``None`` (typically: drop the
    ``_decision_target`` key from the bundle).

    NOTE: under the v3.1 contract, the SAR bundle does NOT carry
    ``_decision_target`` at all (it was a v1-era hint that leaked the
    label). This function is retained for tests / ablation analysis
    only; ``sar_caller.py`` does NOT call it.
    """
    frame = semantic_profile.regulatory_frame
    aux_flagged = _has_aux_red_flag(auxiliary_findings)
    structural_flag = _has_structural_red_flag(
        semantic_profile, sanctions_pep_hits, behavioral_metrics)
    clean_signals = _has_clean_signals(
        semantic_profile, kyc, sanctions_pep_hits, behavioral_metrics)

    if structural_flag:
        return "suspicious"
    if frame == "benign" and not aux_flagged and clean_signals:
        return "not_suspicious"
    if frame in ("tbml", "elder") and not aux_flagged and clean_signals:
        return "not_suspicious"
    if (frame in ("layering_passthrough", "ctr_structuring")
            and not aux_flagged and clean_signals
            and kyc.get("risk_rating") == "low"):
        return "not_suspicious"
    return None
