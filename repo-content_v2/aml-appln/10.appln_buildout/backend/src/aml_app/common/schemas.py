"""Canonical Pydantic schemas — single source of truth for every wire shape
in the AML investigation backend.

These models mirror what the SFT v3.1 pipeline produced
(`4.sdg_sft/run-v3/scripts/schemas.py`, byte-identical contract surface)
and what `5.sdg_corpus_mimic/run-v2/AGENT_USAGE_GUIDE.md` codifies as
the runtime contract. Any drift here silently degrades model behavior
at runtime — they are kept minimal and stable.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums (used as type literals throughout)
# ---------------------------------------------------------------------------
Typology = Literal[
    "structuring", "smurfing", "layering", "trade_based_ml",
    "shell_company", "human_trafficking", "terrorist_financing",
    "elder_exploitation", "none",
]

RegulatoryFrame = Literal[
    "ctr_structuring", "layering_passthrough", "tbml", "shell",
    "sanctions", "elder", "trafficking", "benign",
]

VolumeBand    = Literal["under", "match", "over"]
GeoRisk       = Literal["low", "medium", "high"]
EntityType    = Literal["individual", "business"]
RiskRating    = Literal["low", "medium", "high", "enhanced", "prohibited"]
SanctionsList = Literal["OFAC", "EU", "UN", "OpenSanctions"]
PolicySource  = Literal["FFIEC", "FATF", "FinCEN", "OFAC"]
ChannelEnum   = Literal["wire", "ach", "cash", "card", "cheque", "crypto", "transfer"]
StatutoryLabel = Literal["entailment", "contradiction", "neutral"]
DecisionTarget = Literal["suspicious", "not_suspicious"]


# ---------------------------------------------------------------------------
# Helper: tolerant evidence coercion (used by 3 finding models)
# ---------------------------------------------------------------------------
def _coerce_evidence_to_str(v):
    """Accept either a string or a list of strings for `evidence`. Lists are
    joined with '; ' so consumers always see a single string. Mirrors the
    SFT v3.1 contract where the trained model occasionally emits a list
    even though the SFT corpus uses string evidence consistently."""
    if isinstance(v, list):
        return "; ".join(str(x) for x in v if x)
    return v


# ---------------------------------------------------------------------------
# Input bundle pieces — what each tool returns
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    """Field order matches the SFT wire format exactly: alphabetical
    (amount, channel, counterparty, currency, date, notes). Pydantic's
    model_dump preserves declaration order, so this guarantees the JSON
    serialization matches the corpus the model was trained on."""
    model_config = ConfigDict(extra="ignore")

    amount: float
    channel: ChannelEnum
    counterparty: str
    currency: str
    date: str
    notes: str = ""


class KYCProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entity_id: str
    entity_type: EntityType
    expected_monthly_volume: float = Field(ge=0)
    business_purpose: str
    risk_rating: RiskRating
    incorporation_jurisdiction: str


class SanctionsHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    list: SanctionsList
    match_score: float = Field(ge=0.0, le=1.0)


class PolicyExcerpt(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: PolicySource
    section: str
    url: Optional[str] = None
    text: str


class SOPExcerpt(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sop_id: str
    section: str
    text: str


# ---------------------------------------------------------------------------
# Auxiliary finding shapes
# ---------------------------------------------------------------------------
class BehavioralMetrics(BaseModel):
    """The 10-field behavioral metrics block.

    Every field has a sensible default so a single-field omission in the
    trained model's output does not drop an otherwise-valid finding.
    Pure Pydantic semantics are preserved — fields the model DID emit
    are still validated against their type / range constraints.
    """
    model_config = ConfigDict(extra="ignore")

    tx_count: int = Field(default=0, ge=0)
    tx_total_usd: float = Field(default=0.0, ge=0.0)
    channel_mix: dict[str, float] = Field(default_factory=dict)
    velocity_24h_max: int = Field(default=0, ge=0)
    velocity_24h_avg_30d: float = Field(default=0.0, ge=0.0)
    unique_counterparties_7d: int = Field(default=0, ge=0)
    amount_z_score_max: float = 0.0
    country_risk_max: float = Field(default=0.0, ge=0.0, le=1.0)
    loop_detected: bool = False
    vs_declared_volume_ratio: float = Field(default=0.0, ge=0.0)


class BehavioralFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = ""
    summary: str
    metrics: BehavioralMetrics
    evidence: str = ""

    @field_validator("evidence", mode="before")
    @classmethod
    def _ev(cls, v):
        return _coerce_evidence_to_str(v)


class NumericFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = ""
    answer: str
    calculation: str = ""
    evidence: str = ""

    @field_validator("evidence", mode="before")
    @classmethod
    def _ev(cls, v):
        return _coerce_evidence_to_str(v)


class CitationFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = ""
    answer: str
    evidence_span: str = ""

    @field_validator("evidence_span", mode="before")
    @classmethod
    def _ev(cls, v):
        return _coerce_evidence_to_str(v)


class StatutoryFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = ""
    answer: str
    label: StatutoryLabel
    reasoning: str = ""


class AuxiliaryFindings(BaseModel):
    behavioral: list[BehavioralFinding] = Field(default_factory=list)
    numeric:    list[NumericFinding]    = Field(default_factory=list)
    citation:   list[CitationFinding]   = Field(default_factory=list)
    statutory:  list[StatutoryFinding]  = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SAR judgment — primary input + output schemas
# ---------------------------------------------------------------------------
class SARJudgmentInput(BaseModel):
    """Full bundle the orchestrator hands to sar_judgment_caller.

    sar_judgment_caller serialises this directly into the 7-key user
    message — `task_type, transactions, kyc_profile, sanctions_pep_hits,
    policy_excerpts, sop_excerpts, auxiliary_findings`. No hint fields
    (no `regulatory_frame`, `typology_inferred`, `decision_target`). The
    sar_caller's outer Pydantic input model uses `extra="forbid"` to
    reject any extra keys at request-construction time.
    """
    task_type: Literal["sar_judgment"] = "sar_judgment"
    transactions: list[Transaction]
    kyc_profile: KYCProfile
    sanctions_pep_hits: list[SanctionsHit] = Field(default_factory=list)
    policy_excerpts: list[PolicyExcerpt] = Field(default_factory=list)
    sop_excerpts: list[SOPExcerpt] = Field(default_factory=list)
    auxiliary_findings: Optional[AuxiliaryFindings] = None


class SARJudgmentOutput(BaseModel):
    is_suspicious: bool
    suspicious_activity_report: str


# ---------------------------------------------------------------------------
# Internal artifacts (audit-only; never serialised into user messages)
# ---------------------------------------------------------------------------
class SemanticProfile(BaseModel):
    channel_mix: dict[str, float] = Field(default_factory=dict)
    cash_present: bool
    regulatory_frame: RegulatoryFrame
    declared_volume_band: VolumeBand
    geo_risk: GeoRisk
    typology_inferred: Typology


class AlertCase(BaseModel):
    """Trigger payload for /api/investigation/run — case_id alone is enough
    when the demo manifest has the rest; the explicit fields are for
    callers that send a full alert."""
    case_id: str
    alert_id: str
    entity_id: str
    investigation_window_start: str
    investigation_window_end: str
    trigger_summary: str = ""
