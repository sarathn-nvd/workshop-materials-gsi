"""Aux gate — input guards + schema validity + LLM-as-Judge.

Receives raw aux responses from aux_runner.run_all_aux_calls and decides
which findings to inline into `auxiliary_findings`. The decision is
three-stage per response:

  1) Input-availability guard (deterministic)
  2) Schema validity (Pydantic parse against the relevant *Finding model)
  3) LLM-as-Judge (one reviewer call per surviving response)
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from dataclasses import dataclass

from pydantic import ValidationError

from pipeline.reference_agent.aux_runner import AuxResponse, STATUTE_BY_TYPOLOGY
from pipeline.reference_agent.llm_client import chat
from pipeline.reference_agent.prompts import reviewer_prompt
from pipeline.schemas import (
    AuxiliaryFindings,
    BehavioralFinding,
    CitationFinding,
    KYCProfile,
    NumericFinding,
    PolicyExcerpt,
    StatutoryFinding,
    Transaction,
    Typology,
)

logger = logging.getLogger("pipeline.reference_agent.aux_gate")


@dataclass
class GateDecision:
    task: str
    used: bool
    reason: str
    finding_json: dict | None = None
    reviewer_verdict: str | None = None
    reviewer_explain: str | None = None


# ============================================================================
# Parsing — extract a JSON object from a possibly noisy LLM response
# ============================================================================
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip Markdown fences if any
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
# Schema validation per task type
# ============================================================================
_MODEL_FOR_TASK = {
    "behavioral": BehavioralFinding,
    "numeric":    NumericFinding,
    "citation":   CitationFinding,
    "statutory":  StatutoryFinding,
}


def _validate(task: str, payload: dict) -> tuple[dict | None, str | None]:
    model_cls = _MODEL_FOR_TASK[task]
    try:
        validated = model_cls.model_validate(payload)
        return validated.model_dump(), None
    except ValidationError as e:
        return None, str(e)[:200]


# ============================================================================
# Input-availability guards
# ============================================================================
def _input_guard(
    task: str,
    *,
    transactions: list[Transaction],
    policy_excerpts: list[PolicyExcerpt],
    typology: Typology,
) -> tuple[bool, str]:
    if task in {"behavioral", "numeric"} and len(transactions) < 1:
        return False, f"input-guard: {task} needs ≥1 transaction"
    if task == "citation" and len(policy_excerpts) < 1:
        return False, "input-guard: citation needs ≥1 policy_excerpt"
    if task == "statutory" and typology not in STATUTE_BY_TYPOLOGY:
        return False, f"input-guard: statutory has no statute for typology={typology}"
    return True, "ok"


# ============================================================================
# LLM-as-Judge
# ============================================================================
def _judge(task: str, passage: str, finding: dict) -> tuple[str, str]:
    """Returns (verdict, explain)."""
    user = (
        f"task_type: {'auxiliary_' + task}\n\n"
        f"## Passage given to the model\n{passage[:3000]}\n\n"
        f"## Model's finding\n{json.dumps(finding, indent=2)[:2000]}\n"
    )
    try:
        resp = chat(
            system=reviewer_prompt(task),
            user=user,
            temperature=0.0,
            max_tokens=300,
        )
        verdict_obj = _extract_json(resp.text) or {}
        verdict = verdict_obj.get("verdict", "PASS")
        explain = verdict_obj.get("explain", "")
        if verdict not in {"PASS", "ISSUES_FOUND"}:
            verdict = "PASS"   # conservative fallback
        return verdict, explain
    except Exception as e:
        logger.warning("judge call for %s failed: %s — defaulting PASS", task, e)
        return "PASS", f"judge-fallback (error: {e})"


# ============================================================================
# Public API
# ============================================================================
def gate_responses(
    responses: list[AuxResponse],
    *,
    transactions: list[Transaction],
    kyc_profile: KYCProfile,                 # noqa: ARG001 (carried for symmetry)
    policy_excerpts: list[PolicyExcerpt],
    typology: Typology,
    enable_judge: bool = True,
    judge_concurrency: int = 4,
) -> tuple[AuxiliaryFindings, list[GateDecision]]:
    """Apply guards + schema + judge to all 4 aux responses.

    Returns (AuxiliaryFindings, decisions) where decisions[i] records why
    each response was used or dropped.
    """
    decisions: list[GateDecision] = []
    parsed: dict[str, dict] = {}

    # Step 1+2: per-response guards + schema
    survivors_for_judge: list[tuple[str, str, dict]] = []   # (task, passage, finding)
    for r in responses:
        ok, reason = _input_guard(
            r.task,
            transactions=transactions,
            policy_excerpts=policy_excerpts,
            typology=typology,
        )
        if not ok:
            decisions.append(GateDecision(task=r.task, used=False, reason=reason))
            continue
        if r.raw_text is None:
            decisions.append(GateDecision(
                task=r.task, used=False, reason=f"llm-error: {r.err}",
            ))
            continue
        payload = _extract_json(r.raw_text)
        if payload is None:
            decisions.append(GateDecision(
                task=r.task, used=False,
                reason="schema: response was not valid JSON",
            ))
            continue
        validated, err = _validate(r.task, payload)
        if validated is None:
            decisions.append(GateDecision(
                task=r.task, used=False,
                reason=f"schema: {err}",
            ))
            continue
        survivors_for_judge.append((r.task, r.passage, validated))

    # Step 3: judge calls in parallel
    if enable_judge and survivors_for_judge:
        with concurrent.futures.ThreadPoolExecutor(max_workers=judge_concurrency) as pool:
            futures = {
                pool.submit(_judge, task, passage, finding): (task, finding)
                for task, passage, finding in survivors_for_judge
            }
            for fut in concurrent.futures.as_completed(futures):
                task, finding = futures[fut]
                verdict, explain = fut.result()
                used = verdict == "PASS"
                decisions.append(GateDecision(
                    task=task, used=used,
                    reason=f"judge: {verdict}",
                    finding_json=finding if used else None,
                    reviewer_verdict=verdict,
                    reviewer_explain=explain,
                ))
                if used:
                    parsed[task] = finding
    else:
        for task, _, finding in survivors_for_judge:
            decisions.append(GateDecision(
                task=task, used=True,
                reason="judge-disabled",
                finding_json=finding,
            ))
            parsed[task] = finding

    # Assemble AuxiliaryFindings
    af = AuxiliaryFindings()
    if "behavioral" in parsed:
        af.behavioral = [BehavioralFinding.model_validate(parsed["behavioral"])]
    if "numeric" in parsed:
        af.numeric = [NumericFinding.model_validate(parsed["numeric"])]
    if "citation" in parsed:
        af.citation = [CitationFinding.model_validate(parsed["citation"])]
    if "statutory" in parsed:
        af.statutory = [StatutoryFinding.model_validate(parsed["statutory"])]
    return af, decisions
