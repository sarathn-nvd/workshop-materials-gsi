"""Single source of truth for every record/object shape in the pipeline.

The auxiliary record output shapes MUST match the per-entry shapes inlined
into Stage 6's `auxiliary_findings` — the strategy doc enforces this. Both
non-auxiliary Stage 6 and auxiliary Stage A2/A3 import the same Pydantic
models from this file, so byte-for-byte parity is structurally guaranteed.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# Driver tuple (Stage 1 output)
# ============================================================================
Typology = Literal[
    "structuring", "smurfing", "layering", "trade_based_ml",
    "shell_company", "human_trafficking", "terrorist_financing",
    "elder_exploitation", "none",
]

Severity = Literal["light", "medium", "heavy"]
SurfacePattern = Literal["direct", "near_miss"]
AuxVariant = Literal["augmented", "bare", "adversarial_aux"]


class DriverTuple(BaseModel):
    """The 6 driver values produced in Stage 1 + record-path tag."""
    typology: Typology
    label: bool
    severity: Severity
    surface_pattern: SurfacePattern
    aux_variant: AuxVariant
    entity_archetype: str               # one of 16 catalogued archetypes
    path: Literal[
        "Record_1", "Record_2", "Record_3", "Record_4",
        "Record_5", "Record_6", "Record_7",
    ]


# ============================================================================
# Bundle field schemas (parts of `input` for sar_judgment records)
# ============================================================================
class Transaction(BaseModel):
    date: str
    amount: float
    currency: str
    counterparty: str
    channel: Literal["wire", "ach", "cash", "card", "cheque", "crypto"]
    notes: str = ""


EntityType = Literal["individual", "business"]
RiskRating = Literal["low", "medium", "high", "enhanced", "prohibited"]


class KYCProfile(BaseModel):
    entity_id: str
    entity_type: EntityType
    expected_monthly_volume: float = Field(ge=0)
    business_purpose: str
    risk_rating: RiskRating
    incorporation_jurisdiction: str


SanctionsList = Literal["OFAC", "EU", "UN", "OpenSanctions"]


class SanctionsHit(BaseModel):
    name: str
    list: SanctionsList
    match_score: float = Field(ge=0.0, le=1.0)


PolicySource = Literal["FFIEC", "FATF", "FinCEN", "OFAC"]


class PolicyExcerpt(BaseModel):
    source: PolicySource
    section: str
    url: Optional[str] = None
    text: str


class SOPExcerpt(BaseModel):
    sop_id: str                          # e.g., "SOP-STRUCTURING-02"
    section: str
    text: str


# ============================================================================
# Auxiliary findings — THE shape contract
# These three classes are the same shape Stage 6 inlines AND the assistant
# content of standalone auxiliary records (Stage A3).
# ============================================================================
# Finding shape contract — used in two surfaces:
#   1) Standalone aux record assistant content (auxiliary_* tasks): the
#      `question` is in the user message; the assistant emits answer-only
#      fields. `question` is therefore OPTIONAL on the assistant side and
#      defaults to empty. This avoids the model wasting tokens echoing the
#      question that NAT (the agent) just sent.
#   2) Inlined finding under `sar_judgment.input.auxiliary_findings.{kind}[]`:
#      `question` is REQUIRED. Stage 6 builds these inlined findings with
#      the per-typology question populated; at runtime NAT does the same
#      programmatically when bundling auxiliary tool outputs into the next
#      sar_judgment call.
def _coerce_evidence_to_str(v):
    """Accept either a string or a list of strings for `evidence`. Lists are
    flattened with '; ' separators so the contract presented to downstream
    consumers (validators, narrative writers, runtime) is always a single
    string. This handles a real failure mode observed on the smoke run:
    gemma, when given a passage with multiple referenced table cells, emits
    `evidence: ["Table row X col Y", "Table row Z col W"]`. Strictly typing
    the field as `str` would drop those records on Pydantic validation.

    Pydantic v2 calls validators before type coercion, so this fires on the
    raw input value.
    """
    if isinstance(v, list):
        return "; ".join(str(x) for x in v if x)
    return v


class NumericFinding(BaseModel):
    question: str = ""
    answer: str
    calculation: str
    evidence: str = Field(default="")    # transactions[i,j,k] indices or evidence span

    @field_validator("evidence", mode="before")
    @classmethod
    def _ev(cls, v): return _coerce_evidence_to_str(v)


class CitationFinding(BaseModel):
    question: str = ""
    answer: str
    evidence_span: str                   # verbatim substring of policy_excerpts[].text

    @field_validator("evidence_span", mode="before")
    @classmethod
    def _ev(cls, v): return _coerce_evidence_to_str(v)


StatutoryLabel = Literal["entailment", "contradiction", "neutral"]


class StatutoryFinding(BaseModel):
    question: str = ""
    answer: str
    label: StatutoryLabel
    reasoning: str                       # must contain the cited statute token


# ----------------------------------------------------------------------------
# Behavioral finding (auxiliary_behavioral output)
#
# Two halves to the contract:
#   - `metrics` is gold-anchored: at construction time it comes from a
#     deterministic Pandas computation over the source rows (the model never
#     invents these numbers during training; it learns to *reproduce* them).
#   - `summary` is LLM-written prose that interprets the metrics and is
#     mechanically validated to cite metric values verbatim.
#
# Both halves are emitted at runtime by the model when invoked with
# task_type=auxiliary_behavioral; the assistant content is the JSON-serialized
# `BehavioralFinding`.
# ----------------------------------------------------------------------------
class BehavioralMetrics(BaseModel):
    """Deterministic aggregations over a (transactions[], kyc_profile) bundle.

    All fields are computable from the bundle by the deterministic feature
    computer (`common/behavioral_features.py`). Values are gold at construction
    time, model-produced at runtime; both must agree within tolerance for the
    record to be considered consistent.
    """
    tx_count: int = Field(ge=0)
    tx_total_usd: float = Field(ge=0.0)            # FX-normalized total
    channel_mix: dict[str, float] = Field(default_factory=dict)  # e.g. {"wire": 1.0}
    velocity_24h_max: int = Field(ge=0)
    velocity_24h_avg_30d: float = Field(ge=0.0)
    unique_counterparties_7d: int = Field(ge=0)
    amount_z_score_max: float                       # may be negative
    country_risk_max: float = Field(ge=0.0, le=1.0)
    loop_detected: bool
    vs_declared_volume_ratio: float = Field(ge=0.0)


class BehavioralFinding(BaseModel):
    """Output of `auxiliary_behavioral`. Mirrors the shape used in
    `sar_judgment.input.auxiliary_findings.behavioral[]`.

    `question` is OPTIONAL on the assistant side (NAT supplies it via the
    user message) — same convention as the other auxiliary findings.
    """
    question: str = ""
    summary: str                                     # prose interpretation
    metrics: BehavioralMetrics
    evidence: str = Field(default="")                # transactions[i,j..]; kyc_profile.X

    @field_validator("evidence", mode="before")
    @classmethod
    def _ev(cls, v): return _coerce_evidence_to_str(v)


class AuxiliaryFindings(BaseModel):
    behavioral: list[BehavioralFinding] = Field(default_factory=list)
    numeric: list[NumericFinding] = Field(default_factory=list)
    citation: list[CitationFinding] = Field(default_factory=list)
    statutory: list[StatutoryFinding] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.behavioral or self.numeric or self.citation or self.statutory)


# ============================================================================
# sar_judgment input bundle (the union of all input fields)
# ============================================================================
class SARJudgmentInput(BaseModel):
    task_type: Literal["sar_judgment"] = "sar_judgment"
    transactions: list[Transaction]
    kyc_profile: KYCProfile
    sanctions_pep_hits: list[SanctionsHit] = Field(default_factory=list)
    policy_excerpts: list[PolicyExcerpt] = Field(default_factory=list)
    sop_excerpts: list[SOPExcerpt] = Field(default_factory=list)
    auxiliary_findings: Optional[AuxiliaryFindings] = None


# ============================================================================
# sar_judgment output (the model's training target)
# ============================================================================
class SARJudgmentOutput(BaseModel):
    is_suspicious: bool
    suspicious_activity_report: str


# ============================================================================
# Auxiliary record input bundles
# ============================================================================
class AuxiliaryNumericInput(BaseModel):
    task_type: Literal["auxiliary_numeric"] = "auxiliary_numeric"
    passage: str
    question: str


class AuxiliaryCitationInput(BaseModel):
    task_type: Literal["auxiliary_citation"] = "auxiliary_citation"
    passage: str
    question: str


class AuxiliaryStatutoryInput(BaseModel):
    task_type: Literal["auxiliary_statutory"] = "auxiliary_statutory"
    statute: str
    fact_pattern: str
    question: str


class AuxiliaryBehavioralInput(BaseModel):
    """User-message content for the auxiliary_behavioral task.

    `passage` is the rendered (transactions[] + kyc_profile) bundle as a
    structured text block — same shape NAT will assemble at runtime. The
    question is optional; the system prompt tells the model what to produce.
    """
    task_type: Literal["auxiliary_behavioral"] = "auxiliary_behavioral"
    passage: str
    question: str = ""


# ============================================================================
# Semantic profile — cross-stage construction contract (per training_strategy
# Appendix F). NOT part of any SFT record; consumed by Stages 4/5/6/7/8 to
# keep retrieval and generation coherent with the bundle's actual channel /
# regulatory framing.
# ============================================================================
RegulatoryFrame = Literal[
    "ctr_structuring",       # cash-channel structuring; CTR / 31 USC 5324 framing applies
    "layering_passthrough",  # non-cash rapid movement / circular flow
    "tbml",                  # trade-based money laundering
    "shell",                 # shell-company pass-through
    "sanctions",             # sanctions evasion / OFAC nexus
    "elder",                 # elder financial exploitation
    "trafficking",           # human trafficking AND terrorist financing
                             # (FFIEC/FATF group them; `te` frame dropped per v3 §4.2)
    "benign",                # no laundering signal; benign / near-miss
]

VolumeBand = Literal["under", "match", "over"]
GeoRisk = Literal["low", "medium", "high"]


class SemanticProfile(BaseModel):
    """Computed once per bundle in Stage 1; read by every downstream stage.

    Channel mix and cash_present drive the 'is structuring framing valid'
    decision. regulatory_frame is the canonical retrieval / generation key.
    typology_inferred may differ from the source-data `typology` when channel
    coherence forces a remap (e.g., source says 'structuring' but channels
    are all wires → typology_inferred='layering').
    """
    channel_mix: dict[str, float] = Field(default_factory=dict)
    cash_present: bool
    regulatory_frame: RegulatoryFrame
    declared_volume_band: VolumeBand
    geo_risk: GeoRisk
    typology_inferred: Typology


# ============================================================================
# Metadata — sidecar; never seen by the trainer's model
# ============================================================================
class Metadata(BaseModel):
    model_config = ConfigDict(extra="allow")  # source pools may add custom fields

    record_id: str
    phase: Literal["sft"] = "sft"
    source: str
    typology: Typology
    sar_variant: Optional[AuxVariant] = None      # only set for sar_judgment records
    synthetic: bool = False
    task_type: Literal[
        "sar_judgment",
        "auxiliary_behavioral",
        "auxiliary_numeric", "auxiliary_citation", "auxiliary_statutory",
    ]
    adversarial_op: Optional[Literal[
        "NUM-FLIP", "CIT-SWAP", "STAT-INVERT", "BEHAV-CORRUPT",
    ]] = None


# ============================================================================
# Final chat-SFT envelope (the line on disk in the .jsonl)
# ============================================================================
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatSFTRecord(BaseModel):
    """Final on-disk shape for one training record (one JSONL line)."""
    messages: list[ChatMessage] = Field(min_length=3, max_length=3)
    metadata: Metadata


# ============================================================================
# Rule failure report
# ============================================================================
class RuleFailure(BaseModel):
    rule_id: str
    record_id: str
    reason: str


class StageManifest(BaseModel):
    """Per-stage status manifest, written at the end of every stage."""
    stage: str
    pipeline: Literal["nonaux", "aux", "tools_prep", "combine"]
    started_at: str
    ended_at: str
    runtime_seconds: float
    input_files: list[str]
    output_files: list[str]
    counts: dict[str, int]
    llm_calls: int = 0
    llm_calls_per_record_avg: float = 0.0
    rule_failures: dict[str, int] = Field(default_factory=dict)
    drift: dict[str, Any] = Field(default_factory=dict)
    sample_failures: list[RuleFailure] = Field(default_factory=list)
    status: Literal["ok", "warn", "fail"] = "ok"
    notes: str = ""
