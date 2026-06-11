"""aux_gate \u2014 input-availability guard + schema validity + LLM-as-Judge.

Receives the 4 specialist findings from the orchestrator. Returns a
filtered AuxiliaryFindings object plus a per-finding decision log
(USE / DROP + reason). Findings that fail any of the three stages are
dropped before they reach sar_judgment_caller.
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from aml_app.common.schemas import (
    AuxiliaryFindings,
    BehavioralFinding,
    CitationFinding,
    NumericFinding,
    StatutoryFinding,
)
from aml_app.skills.aux_call import extract_json
from aml_app.skills.prompts import STATUTE_BY_TYPOLOGY, reviewer_prompt

logger = logging.getLogger("aml_app.gating.aux_gate")


# ---------------------------------------------------------------------------
# Input / output schemas
# ---------------------------------------------------------------------------
class AuxGateInput(BaseModel):
    behavioral: Optional[dict] = None
    numeric: Optional[dict] = None
    citation: Optional[dict] = None
    statutory: Optional[dict] = None
    transactions: list[dict] = Field(default_factory=list)
    kyc_profile: dict = Field(default_factory=dict)
    policy_excerpts: list[dict] = Field(default_factory=list)
    typology: str = "none"


class GateDecisionEntry(BaseModel):
    task: str
    used: bool
    reason: str
    reviewer_verdict: Optional[str] = None
    reviewer_explain: Optional[str] = None


class AuxGateOutput(BaseModel):
    auxiliary_findings: dict
    decisions: list[GateDecisionEntry]
    incomplete: bool
    used_count: int


class AuxGateConfig(FunctionBaseConfig, name="aux_gate"):
    judge_llm_name: LLMRef = Field(..., description="LLM used for the LLM-as-Judge stage.")
    enable_judge: bool = Field(default=True)


_MODEL_FOR_TASK: dict[str, type[BaseModel]] = {
    "behavioral": BehavioralFinding,
    "numeric":    NumericFinding,
    "citation":   CitationFinding,
    "statutory":  StatutoryFinding,
}


def _input_guard(task: str, n_tx: int, n_pol: int, typology: str) -> tuple[bool, str]:
    if task in {"behavioral", "numeric"} and n_tx < 1:
        return False, f"input-guard: {task} needs >=1 transaction"
    if task == "citation" and n_pol < 1:
        return False, "input-guard: citation needs >=1 policy_excerpt"
    if task == "statutory" and typology not in STATUTE_BY_TYPOLOGY:
        return False, f"input-guard: statutory has no statute for typology={typology}"
    return True, "ok"


def _validate(task: str, payload: dict) -> tuple[dict | None, str | None]:
    model_cls = _MODEL_FOR_TASK[task]
    try:
        validated = model_cls.model_validate(payload)
        return validated.model_dump(), None
    except ValidationError as e:
        return None, str(e)[:200]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
@register_function(config_type=AuxGateConfig,
                   framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def aux_gate(config: AuxGateConfig, builder: Builder):
    from langchain_core.messages import HumanMessage, SystemMessage

    judge = None
    if config.enable_judge:
        judge = await builder.get_llm(
            llm_name=config.judge_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

    async def _call_judge(task: str, passage: str, finding: dict) -> tuple[str, str]:
        if judge is None:
            return "PASS", "judge-disabled"
        user = (
            f"task_type: auxiliary_{task}\n\n"
            f"## Passage given to the model\n{passage[:3000]}\n\n"
            f"## Model's finding\n{json.dumps(finding, indent=2)[:2000]}\n"
        )
        try:
            try:
                bound = judge.bind(temperature=0.0, max_tokens=300)
            except Exception:
                bound = judge
            resp = await bound.ainvoke([
                SystemMessage(content=reviewer_prompt(task)),
                HumanMessage(content=user),
            ])
            raw = str(getattr(resp, "content", str(resp)))
            verdict_obj = extract_json(raw) or {}
            verdict = verdict_obj.get("verdict", "PASS")
            explain = verdict_obj.get("explain", "")
            if verdict not in {"PASS", "ISSUES_FOUND"}:
                verdict = "PASS"
            return verdict, explain
        except Exception as e:
            logger.warning("Judge call failed for %s: %s \u2014 defaulting PASS", task, e)
            return "PASS", f"judge-fallback ({e})"

    async def _run(args: AuxGateInput) -> dict:
        decisions: list[GateDecisionEntry] = []
        parsed: dict[str, dict] = {}

        n_tx = len(args.transactions)
        n_pol = len(args.policy_excerpts)

        raw_findings = {
            "behavioral": args.behavioral,
            "numeric":    args.numeric,
            "citation":   args.citation,
            "statutory":  args.statutory,
        }

        for task, finding in raw_findings.items():
            if finding is None:
                decisions.append(GateDecisionEntry(task=task, used=False,
                                                   reason="not_provided"))
                continue

            ok, reason = _input_guard(task, n_tx, n_pol, args.typology)
            if not ok:
                decisions.append(GateDecisionEntry(task=task, used=False, reason=reason))
                continue

            # If the specialist returned an error-shaped dict, drop early
            if isinstance(finding, dict) and finding.get("error"):
                decisions.append(GateDecisionEntry(
                    task=task, used=False,
                    reason=f"specialist-error: {finding.get('error')}",
                ))
                continue

            validated, err = _validate(task, finding)
            if validated is None:
                decisions.append(GateDecisionEntry(
                    task=task, used=False, reason=f"schema: {err}",
                ))
                continue

            verdict, explain = await _call_judge(task, json.dumps(finding)[:3000], validated)
            used = verdict == "PASS"
            decisions.append(GateDecisionEntry(
                task=task, used=used,
                reason=f"judge: {verdict}",
                reviewer_verdict=verdict, reviewer_explain=explain,
            ))
            if used:
                parsed[task] = validated

        af = AuxiliaryFindings()
        if "behavioral" in parsed:
            af.behavioral = [BehavioralFinding.model_validate(parsed["behavioral"])]
        if "numeric" in parsed:
            af.numeric = [NumericFinding.model_validate(parsed["numeric"])]
        if "citation" in parsed:
            af.citation = [CitationFinding.model_validate(parsed["citation"])]
        if "statutory" in parsed:
            af.statutory = [StatutoryFinding.model_validate(parsed["statutory"])]

        used_count = sum(1 for d in decisions if d.used)
        provided = sum(1 for v in raw_findings.values() if v is not None)
        return AuxGateOutput(
            auxiliary_findings=af.model_dump(),
            decisions=[d.model_dump() for d in decisions],
            incomplete=(provided < 4),
            used_count=used_count,
        ).model_dump()

    yield FunctionInfo.from_fn(
        _run,
        description=(
            "Three-stage gate (input guard + schema + LLM-as-Judge) over the four "
            "auxiliary specialist findings. Returns a filtered AuxiliaryFindings "
            "object ready to inline into the SAR bundle."
        ),
        input_schema=AuxGateInput,
    )
