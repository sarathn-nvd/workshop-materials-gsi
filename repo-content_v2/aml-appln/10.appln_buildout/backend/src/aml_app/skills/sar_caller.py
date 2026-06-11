"""sar_judgment_caller — assembles the **7-key** user message in the EXACT
order specified by `5.sdg_corpus_mimic/run-v2/AGENT_USAGE_GUIDE.md` §3.5
and `4.sdg_sft/run-v3/SDG_STRATEGY_SFT.md` §3.3, posts it to the trained
Custom Task NIM, and parses the response into SARJudgmentOutput.

**v2 update (aligned with SFT v3.1)**: the bundle has EXACTLY 7 keys.
The previous 10-key bundle injected `_regulatory_frame`,
`_typology_inferred`, and `_decision_target` hint fields — that was the
deterministic label leak v3.1 SFT explicitly removed
(`RULE-1-NO-LEAKY-FIELDS` + `RULE-7-NO-LEAKY-HINTS`). The v3.1-trained
model derives typology + frame + verdict from the bundle evidence
ALONE. Hint values are still computed at the orchestrator layer
(`tools/hints.py`) but are used only to ROUTE Tool 4 / Tool 5 retrieval
and to populate audit metadata in the `CaseTrace` — they MUST NOT
appear in the user message sent to the model.

The orchestrator passes typed Pydantic objects; this leaf is the single
place that knows the wire-format ordering. Pydantic validation rejects
malformed inputs before the LLM is ever invoked, so the orchestrator
cannot accidentally re-order or drop fields. `SARCallerInput` uses
``extra="forbid"`` so any caller passing a hint field is rejected at
request-construction time.
"""
import json
import logging
import os
from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from aml_app.common.schemas import (
    AuxiliaryFindings, KYCProfile, PolicyExcerpt, SanctionsHit,
    SARJudgmentOutput, SOPExcerpt, Transaction,
)
from aml_app.skills.aux_call import extract_json
from aml_app.skills.prompts import SAR_JUDGMENT_SYSTEM_PROMPT, _maybe_no_think


logger = logging.getLogger("aml_app.skills.sar_caller")


# ---------------------------------------------------------------------------
# I/O shapes
# ---------------------------------------------------------------------------
class SARCallerInput(BaseModel):
    """The 7 evidence fields the orchestrator passes in.

    ``extra="forbid"`` actively rejects any caller that tries to inject
    a v1-era hint field (`regulatory_frame`, `typology_inferred`,
    `decision_target`). This guards the SFT v3.1 contract at the input
    boundary — the trained model never saw those fields and they would
    re-introduce the label leak the v3.1 corpus was redesigned around.
    """
    model_config = ConfigDict(extra="forbid")

    transactions: list[dict] = Field(default_factory=list)
    kyc_profile: dict = Field(default_factory=dict)
    sanctions_pep_hits: list[dict] = Field(default_factory=list)
    policy_excerpts: list[dict] = Field(default_factory=list)
    sop_excerpts: list[dict] = Field(default_factory=list)
    auxiliary_findings: dict | None = None


class SARCallerOutput(BaseModel):
    is_suspicious: bool | None
    suspicious_activity_report: str
    raw_text: str
    parse_error: str | None
    user_message: str


class SARJudgmentCallerConfig(FunctionBaseConfig, name="sar_judgment_caller"):
    llm_name: LLMRef = Field(..., description="LLM binding for the trained Custom Task NIM.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2500, ge=64, le=8192)
    max_thinking_tokens: int = Field(
        default=0,
        description=(
            "Cap reasoning tokens for reasoning-capable Nemotron NIMs. "
            "0 = no cap. The trained Custom Task NIM ignores this; on a "
            "base model set to e.g. 512 to keep latency in check."
        ),
    )


# ---------------------------------------------------------------------------
# Contract constants — these MUST not change
# ---------------------------------------------------------------------------
SAR_BUNDLE_KEYS = (
    "task_type",
    "transactions",
    "kyc_profile",
    "sanctions_pep_hits",
    "policy_excerpts",
    "sop_excerpts",
    "auxiliary_findings",
)

_FORBIDDEN_HINT_KEYS = frozenset({
    "_decision_target",
    "_regulatory_frame",
    "_typology_inferred",
    "_reference_patterns",
})


# ---------------------------------------------------------------------------
# Build the 7-key user message
# ---------------------------------------------------------------------------
def build_user_message(args: SARCallerInput) -> str:
    """Build the **7-key** SAR user message — matches what the v3.1-trained
    model saw at every step of Stage 7 SFT.

    Key order is non-negotiable:

        task_type, transactions, kyc_profile, sanctions_pep_hits,
        policy_excerpts, sop_excerpts, auxiliary_findings

    No hint fields. No `_regulatory_frame`, `_typology_inferred`,
    `_decision_target`, `_reference_patterns`. The orchestrator can compute
    these for internal routing (see `tools/hints.py`) but MUST NOT inject
    them here.
    """
    # Validate each sub-shape through its Pydantic model. This both
    # normalizes (e.g. strips unknown keys) and rejects malformed values.
    tx_list = [Transaction.model_validate(t).model_dump() for t in args.transactions]

    kyc = args.kyc_profile or {}
    kyc_clean = {
        "entity_id": str(kyc.get("entity_id", "")),
        "entity_type": kyc.get("entity_type", "individual"),
        "expected_monthly_volume": float(kyc.get("expected_monthly_volume", 0) or 0),
        "business_purpose": kyc.get("business_purpose", ""),
        "risk_rating": kyc.get("risk_rating", "medium"),
        "incorporation_jurisdiction": kyc.get("incorporation_jurisdiction", ""),
    }
    # Re-validate via KYCProfile to ensure literal-types check.
    kyc_clean = KYCProfile.model_validate(kyc_clean).model_dump()

    hits = [SanctionsHit.model_validate(h).model_dump()
            for h in (args.sanctions_pep_hits or [])]
    policies = [PolicyExcerpt.model_validate(p).model_dump()
                for p in (args.policy_excerpts or [])]
    sops = [SOPExcerpt.model_validate(s).model_dump()
            for s in (args.sop_excerpts or [])]

    # Auxiliary findings: validate AND collapse to null when empty.
    aux_clean = None
    if args.auxiliary_findings:
        aux_obj = AuxiliaryFindings.model_validate(args.auxiliary_findings)
        if not (aux_obj.behavioral or aux_obj.numeric or aux_obj.citation or aux_obj.statutory):
            aux_clean = None
        else:
            aux_clean = aux_obj.model_dump()

    bundle = {
        "task_type":          "sar_judgment",
        "transactions":       tx_list,
        "kyc_profile":        kyc_clean,
        "sanctions_pep_hits": hits,
        "policy_excerpts":    policies,
        "sop_excerpts":       sops,
        "auxiliary_findings": aux_clean,
    }

    # Defence-in-depth: verify the bundle keys MATCH the contract order +
    # forbid any hint-field leak.
    actual_keys = tuple(bundle.keys())
    assert actual_keys == SAR_BUNDLE_KEYS, (
        f"SAR bundle keys must be {SAR_BUNDLE_KEYS} in exact order; "
        f"got {actual_keys}"
    )
    leaked = _FORBIDDEN_HINT_KEYS.intersection(bundle.keys())
    assert not leaked, (
        f"v3.1 leak — forbidden hint field(s) in SAR bundle: "
        f"{sorted(leaked)}. The trained model never saw these and they "
        f"re-introduce the v1 F1=0.262 label-leak failure mode."
    )

    return json.dumps(bundle)


# ---------------------------------------------------------------------------
# Registered NAT function
# ---------------------------------------------------------------------------
@register_function(config_type=SARJudgmentCallerConfig)
async def sar_judgment_caller(config: SARJudgmentCallerConfig, builder: Builder):
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = _maybe_no_think(SAR_JUDGMENT_SYSTEM_PROMPT)
    llm = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )

    async def _call(args: SARCallerInput) -> dict:
        if args is None:
            args = SARCallerInput()

        try:
            user_text = build_user_message(args)
        except (ValidationError, AssertionError) as e:
            return {
                "is_suspicious": None,
                "suspicious_activity_report": "",
                "raw_text": "",
                "parse_error": f"build_user_message: {type(e).__name__}: {str(e)[:300]}",
                "user_message": "",
            }

        # Generation params (same env-var overrides as aux_call).
        gen_kwargs = {
            "temperature": config.temperature,
            "max_tokens":  int(os.environ.get("NAT_AML_SAR_MAX_TOKENS",
                                                config.max_tokens) or config.max_tokens),
        }
        if os.environ.get("NAT_AML_NO_THINK", "").strip().lower() in ("1", "true", "yes", "y"):
            gen_kwargs.setdefault("extra_body", {})
            gen_kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        thinking_budget = int(os.environ.get("NAT_AML_THINKING_BUDGET",
                                                config.max_thinking_tokens) or 0)
        if thinking_budget > 0:
            gen_kwargs.setdefault("extra_body", {})["max_thinking_tokens"] = thinking_budget

        try:
            bound = llm.bind(**gen_kwargs)
        except Exception:
            bound = llm

        messages = [SystemMessage(content=system_prompt),
                    HumanMessage(content=user_text)]
        try:
            resp = await bound.ainvoke(messages)
            raw_text = getattr(resp, "content", None) or str(resp)
        except Exception as e:
            return {
                "is_suspicious": None,
                "suspicious_activity_report": "",
                "raw_text": "",
                "parse_error": f"llm.ainvoke: {type(e).__name__}: {str(e)[:300]}",
                "user_message": user_text,
            }

        parsed = extract_json(raw_text)
        if parsed is None:
            return {
                "is_suspicious": None,
                "suspicious_activity_report": "",
                "raw_text": raw_text or "",
                "parse_error": "no_json",
                "user_message": user_text,
            }

        try:
            output = SARJudgmentOutput.model_validate(parsed)
            return {
                "is_suspicious": output.is_suspicious,
                "suspicious_activity_report": output.suspicious_activity_report,
                "raw_text": raw_text,
                "parse_error": None,
                "user_message": user_text,
            }
        except ValidationError as e:
            return {
                "is_suspicious": parsed.get("is_suspicious") if isinstance(parsed, dict) else None,
                "suspicious_activity_report": (
                    parsed.get("suspicious_activity_report", "") if isinstance(parsed, dict) else ""
                ),
                "raw_text": raw_text,
                "parse_error": f"schema_validation: {str(e)[:300]}",
                "user_message": user_text,
            }

    yield FunctionInfo.from_fn(
        _call,
        description=(
            "Assemble the 7-key SAR bundle, post to the trained model, "
            "parse the response into SARJudgmentOutput."
        ),
        input_schema=SARCallerInput,
    )
