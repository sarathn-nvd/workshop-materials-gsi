"""Cross-stage semantic profile (training_strategy.md Appendix F).

Computes a single `SemanticProfile` object from a (transactions[], kyc_profile,
source_typology) bundle. Every downstream stage that retrieves or generates
content reads this profile rather than the raw `typology` field, so a
typology change in one stage propagates to all others.

Pure function — no LLM, no I/O, no side effects.

The profile is an INTERNAL construction-pipeline contract; it is NOT serialized
into any SFT record (per Appendix F). Stages 4, 5, 6, 7, 8 read it; the
training data shows only the resulting bundle and its narrative/findings.
"""
from __future__ import annotations

from typing import Iterable

from pipeline.common.behavioral_features import (
    _COUNTRY_RISK,
    _country_risk,
    _normalize_channel,
    compute_behavioral_metrics,
)
from pipeline.schemas import (
    KYCProfile,
    RegulatoryFrame,
    SanctionsHit,
    SemanticProfile,
    Transaction,
    Typology,
)


# ============================================================================
# Source-typology → candidate regulatory frame mapping
# ============================================================================
# A source label of "structuring" *might* map to ctr_structuring (cash) OR
# layering_passthrough (non-cash) — channel mix decides. The other typologies
# are unambiguous w.r.t. cash, but we still keep the table for symmetry.
# Strategy doc §4.2 drops the `te` frame (60 v2 records, 100% one-sided →
# label proxy). The `terrorist_financing` typology survives and is reframed
# to `trafficking`. FFIEC and FATF group illicit value-transfer typologies
# (HT + TF) together under the same examination chapters, so the trafficking
# frame is the right structural fit.
_TYPOLOGY_FRAME_HINT: dict[Typology, RegulatoryFrame] = {
    "structuring":         "ctr_structuring",
    "smurfing":            "ctr_structuring",
    "layering":            "layering_passthrough",
    "trade_based_ml":      "tbml",
    "shell_company":       "shell",
    "human_trafficking":   "trafficking",
    "terrorist_financing": "trafficking",   # was "te" (frame dropped)
    "elder_exploitation":  "elder",
    "none":                "benign",
}


# ============================================================================
# Public API
# ============================================================================
def compute_semantic_profile(
    transactions: Iterable[dict | Transaction],
    kyc_profile: dict | KYCProfile,
    source_typology: Typology,
    sanctions_pep_hits: Iterable[dict | SanctionsHit] | None = None,
) -> SemanticProfile:
    """Compute the cross-stage semantic profile for a bundle.

    Channel-coherence remap is the most important behavior:
      - source_typology in {structuring, smurfing} + cash_present=False
        → typology_inferred='layering', regulatory_frame='layering_passthrough'

    Sanctions / OFAC nexus override:
      - any sanctions hit with match_score >= 0.5 promotes regulatory_frame
        to 'sanctions' regardless of typology, since the SAR narrative must
        center on the sanctions risk.
    """
    # ----- normalize ---------------------------------------------------------
    txs = []
    for t in transactions or []:
        if isinstance(t, Transaction):
            d = t.model_dump()
        elif isinstance(t, dict):
            d = dict(t)
        else:
            continue
        d["channel"] = _normalize_channel(str(d.get("channel", "")))
        txs.append(d)

    if isinstance(kyc_profile, KYCProfile):
        kyc = kyc_profile.model_dump()
    else:
        kyc = dict(kyc_profile or {})

    hits: list[dict] = []
    for h in sanctions_pep_hits or []:
        if isinstance(h, SanctionsHit):
            hits.append(h.model_dump())
        elif isinstance(h, dict):
            hits.append(dict(h))

    # ----- channel mix + cash flag ------------------------------------------
    channel_mix = _channel_mix(txs)
    cash_present = channel_mix.get("cash", 0.0) > 0.0

    # ----- start from typology hint -----------------------------------------
    typology_inferred: Typology = source_typology
    regulatory_frame: RegulatoryFrame = _TYPOLOGY_FRAME_HINT.get(
        source_typology, "benign"
    )

    # ----- Channel-coherence remap (the structural fix) ---------------------
    # Cash-specific typologies (structuring / smurfing) require cash channels.
    # If the source data labels a non-cash bundle as 'structuring', remap to
    # 'layering' rather than fight the data.
    if source_typology in ("structuring", "smurfing") and not cash_present:
        typology_inferred = "layering"
        regulatory_frame = "layering_passthrough"

    # ----- Sanctions override ----------------------------------------------
    if any(float(h.get("match_score", 0.0) or 0.0) >= 0.5 for h in hits):
        regulatory_frame = "sanctions"

    # ----- declared volume band ---------------------------------------------
    metrics = compute_behavioral_metrics(txs, kyc)
    ratio = metrics.vs_declared_volume_ratio
    if ratio == 0.0:
        declared_volume_band = "match"   # no declared volume to compare against
    elif ratio < 0.5:
        declared_volume_band = "under"
    elif ratio <= 1.5:
        declared_volume_band = "match"
    else:
        declared_volume_band = "over"

    # ----- geo risk ---------------------------------------------------------
    if metrics.country_risk_max >= 0.65:
        geo_risk = "high"
    elif metrics.country_risk_max >= 0.30:
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


# ============================================================================
# Channel-mix (also used by behavioral_features but recomputed here on raw
# dicts to keep this module self-contained for callers that only need the
# semantic profile)
# ============================================================================
def _channel_mix(txs: list[dict]) -> dict[str, float]:
    if not txs:
        return {}
    from collections import Counter
    c = Counter(t.get("channel", "wire") for t in txs)
    n = len(txs)
    return {k: round(v / n, 3) for k, v in c.most_common()}
