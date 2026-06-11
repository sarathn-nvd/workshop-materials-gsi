"""Final SAR call to the SFT-trained Custom Task NIM.

Assembles the user-message JSON in the EXACT shape the shipped SFT corpus
uses (the pair-ground path:
`4.sdg_sft/scripts/non_auxiliary/stage_7_pair_ground/stage.py`), sends it
to the LLM with the verbatim SFT-time system prompt, parses the response
into `is_suspicious` + `suspicious_activity_report`.

User-message shape (10 keys, in this exact order, matching the shipped
SFT corpus):

    {
      "task_type":            "sar_judgment",
      "transactions":         [<tx>, ...],
      "kyc_profile":          {
        "entity_id":                  str,
        "entity_type":                str,
        "expected_monthly_volume":    int,        # int, not float
        "business_purpose":           str,
        "risk_rating":                str,
        "incorporation_jurisdiction": str,
      },
      "sanctions_pep_hits":   [<hit>, ...],
      "policy_excerpts":      [<excerpt>, ...],
      "sop_excerpts":         [<sop>, ...],
      "auxiliary_findings":   {behavioral|numeric|citation|statutory:[...]} | null,
      "_regulatory_frame":    "<regulatory frame from compute_semantic_profile>",
      "_typology_inferred":   "<typology from compute_semantic_profile>",
      "_decision_target":     "suspicious | not_suspicious"
    }

Alignment notes vs SFT:
- The 3 underscore-prefixed hint fields (`_regulatory_frame`, `_typology_inferred`,
  `_decision_target`) appear in EVERY one of the 24,487 shipped SFT records.
  They were computed at SFT-construction time from `compute_semantic_profile`
  (frame + inferred typology) and the gold label (decision_target). At
  runtime we don't have the label, so `_decision_target` is derived from
  our typology classifier's output: `none` → "not_suspicious"; else →
  "suspicious". The model is trained to use this as a prior and can
  override it via its own `is_suspicious` decision.
- `must_cite_verbatim` is NOT in the shipped corpus (despite being in some
  prompt variants). We don't include it either.
- `expected_monthly_volume` is cast to `int`.
- JSON serialization uses `ensure_ascii=False, default=str`.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from pydantic import ValidationError

from pipeline.reference_agent.llm_client import chat
from pipeline.reference_agent.prompts import SAR_JUDGMENT_SYSTEM_PROMPT
from pipeline.schemas import AuxiliaryFindings, SARJudgmentInput, SARJudgmentOutput

logger = logging.getLogger("pipeline.reference_agent.sar_caller")

# A/B test toggle: set OMIT_DECISION_TARGET=1 to drop _decision_target from
# the user message. Used to measure whether the SFT-trained model handles
# the field's absence gracefully (the field was in 100% of training records).
_OMIT_DECISION_TARGET = os.getenv("OMIT_DECISION_TARGET", "0") == "1"


@dataclass
class SARCallResult:
    output: SARJudgmentOutput | None
    raw_text: str
    parse_error: str | None
    latency_ms: float
    user_content: str = ""        # the exact user JSON sent (for traces)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
# User-message JSON assembly — mirrors SFT pair_ground exactly
# ============================================================================
def _build_user_content(
    input_bundle: SARJudgmentInput,
    *,
    regulatory_frame: str,
    typology_inferred: str,
    decision_target: str,
) -> str:
    """Build the user-message JSON exactly as the SFT pair_ground stage did.

    Mirrors the shipped SFT corpus shape — 10 keys in this exact order:
    task_type, transactions, kyc_profile, sanctions_pep_hits,
    policy_excerpts, sop_excerpts, auxiliary_findings, _regulatory_frame,
    _typology_inferred, _decision_target.
    """
    aux = input_bundle.auxiliary_findings
    aux_dict: dict | None
    if isinstance(aux, AuxiliaryFindings):
        aux_dict = aux.model_dump()
        # Treat fully-empty aux as null (matches SFT bare-variant convention)
        if not any(
            (aux_dict.get(k) or [])
            for k in ("behavioral", "numeric", "citation", "statutory")
        ):
            aux_dict = None
    else:
        aux_dict = None

    bundle: dict[str, object] = {
        "task_type": "sar_judgment",
        "transactions": [t.model_dump() for t in input_bundle.transactions],
    }
    kyc = input_bundle.kyc_profile
    bundle["kyc_profile"] = {
        "entity_id": kyc.entity_id,
        "entity_type": kyc.entity_type,
        "expected_monthly_volume": int(kyc.expected_monthly_volume),
        "business_purpose": kyc.business_purpose,
        "risk_rating": kyc.risk_rating,
        "incorporation_jurisdiction": kyc.incorporation_jurisdiction,
    }
    bundle["sanctions_pep_hits"] = [
        h.model_dump() for h in input_bundle.sanctions_pep_hits
    ]
    bundle["policy_excerpts"] = [p.model_dump() for p in input_bundle.policy_excerpts]
    bundle["sop_excerpts"] = [s.model_dump() for s in input_bundle.sop_excerpts]
    bundle["auxiliary_findings"] = aux_dict
    bundle["_regulatory_frame"] = regulatory_frame
    bundle["_typology_inferred"] = typology_inferred
    if not _OMIT_DECISION_TARGET:
        bundle["_decision_target"] = decision_target

    return json.dumps(bundle, ensure_ascii=False, default=str)


def call_sar(
    input_bundle: SARJudgmentInput,
    *,
    regulatory_frame: str,
    typology_inferred: str,
    decision_target: str,
    temperature: float = 0.3,
) -> SARCallResult:
    """One sar_judgment call. Returns the parsed output (or None on failure).

    User message is built to match the SHIPPED SFT corpus shape exactly;
    system prompt is the verbatim SFT-time pair_ground prompt.
    """
    user_msg = _build_user_content(
        input_bundle,
        regulatory_frame=regulatory_frame,
        typology_inferred=typology_inferred,
        decision_target=decision_target,
    )
    resp = chat(
        system=SAR_JUDGMENT_SYSTEM_PROMPT,
        user=user_msg,
        temperature=temperature,
        max_tokens=2500,
    )
    payload = _extract_json(resp.text)
    if payload is None:
        return SARCallResult(
            output=None, raw_text=resp.text,
            parse_error="no JSON found in model output",
            latency_ms=resp.latency_ms,
            user_content=user_msg,
        )
    try:
        out = SARJudgmentOutput.model_validate(payload)
        return SARCallResult(
            output=out, raw_text=resp.text,
            parse_error=None, latency_ms=resp.latency_ms,
            user_content=user_msg,
        )
    except ValidationError as e:
        return SARCallResult(
            output=None, raw_text=resp.text,
            parse_error=str(e)[:200],
            latency_ms=resp.latency_ms,
            user_content=user_msg,
        )
