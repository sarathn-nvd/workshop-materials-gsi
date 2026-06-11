"""Deterministic end-to-end investigation workflow.

This is the practical entry point for the workshop demo: it executes the
exact per-case sequence specified in
`5.sdg_corpus_mimic/run-v2/AGENT_USAGE_GUIDE.md` §2 without relying on
an LLM-driven orchestrator. Use this when only the trained Custom Task
NIM is deployed (which doesn't support tool-calling), or whenever you
want a reproducible, fully observable run.

For the full agentic showcase (reasoning_agent → tool_calling_agent
→ 4 ReAct sub-agents → leaves), the same registered leaf
functions are wired into configs/workflow.yaml; that path additionally
requires a tool-calling-capable orchestrator LLM.

**v2 (aligned with SFT v3.1)**: the final SAR call (`sar_caller_fn`) is
handed a 6-field bundle (transactions, kyc_profile, sanctions_pep_hits,
policy_excerpts, sop_excerpts, auxiliary_findings) which `sar_caller`
serializes into the canonical 7-key user message. The orchestrator no
longer passes `regulatory_frame`, `typology_inferred`, or
`decision_target` to the caller — those fields are computed for INTERNAL
routing of Tool 4 / Tool 5 + audit metadata in the `CaseTrace`, but
NEVER appear in the user message the trained model sees.
"""
import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionGroupRef, FunctionRef
from nat.data_models.function import FunctionBaseConfig

from aml_app.common.behavioral_features import compute_behavioral_metrics
from aml_app.common.schemas import (
    AuxiliaryFindings, KYCProfile, PolicyExcerpt, SanctionsHit,
    SOPExcerpt, Transaction,
)
from aml_app.skills.prompts import NUMERIC_QUESTION_BY_TYPOLOGY, STATUTE_BY_TYPOLOGY
from aml_app.utils.data_loader import get_data_plane
from aml_app.workflow.trace import CaseTrace, write_trace


logger = logging.getLogger("aml_app.workflow.investigate_case")


# ---------------------------------------------------------------------------
# I/O shapes
# ---------------------------------------------------------------------------
class InvestigateCaseInput(BaseModel):
    """Either pass a case_id (looked up in the manifest) OR a full alert."""
    case_id: Optional[str] = None
    alert_id: Optional[str] = None
    entity_id: Optional[str] = None
    investigation_window_start: Optional[str] = None
    investigation_window_end: Optional[str] = None
    trigger_summary: str = ""


class InvestigateCaseConfig(FunctionBaseConfig, name="investigate_case"):
    data_tools: FunctionGroupRef = Field(..., description="aml_data_tools function group.")
    compute_hints_fn:  FunctionRef = Field(..., description="compute_hints leaf function.")
    aux_behavioral_fn: FunctionRef = Field(...)
    aux_numeric_fn:    FunctionRef = Field(...)
    aux_citation_fn:   FunctionRef = Field(...)
    aux_statutory_fn:  FunctionRef = Field(...)
    aux_gate_fn:       FunctionRef = Field(...)
    sar_caller_fn:     FunctionRef = Field(...)
    enable_aux_gate:   bool = Field(default=True)
    max_counterparty_screens: int = Field(default=10, ge=0, le=100)
    fail_open_on_aux_error: bool = Field(
        default=True,
        description="If a single aux call raises / returns error, log and continue.",
    )


# ---------------------------------------------------------------------------
# Passage rendering — matches what SFT Stage 7 produced as user-message inputs
# ---------------------------------------------------------------------------
_TX_FIELDS  = ("date", "channel", "amount", "currency", "counterparty", "notes")
_KYC_FIELDS = ("entity_id", "entity_type", "expected_monthly_volume",
                "business_purpose", "risk_rating", "incorporation_jurisdiction")
_W_DATE, _W_CHAN, _W_AMT, _W_CCY, _W_CP = 10, 6, 14, 4, 32


def _fmt_tx_row(t: dict) -> str:
    date = str(t.get("date", "")).ljust(_W_DATE)[:_W_DATE]
    chan = str(t.get("channel", "")).ljust(_W_CHAN)[:_W_CHAN]
    try:
        amt_val = float(t.get("amount", 0) or 0)
    except (TypeError, ValueError):
        amt_val = 0.0
    amt = f"{amt_val:,.2f}".rjust(_W_AMT)[:_W_AMT]
    ccy = str(t.get("currency", "USD")).ljust(_W_CCY)[:_W_CCY]
    cp  = str(t.get("counterparty", "")).ljust(_W_CP)[:_W_CP]
    note_raw = str(t.get("notes", "") or "")
    note = note_raw.strip()
    return f"{date} | {chan} | {amt} | {ccy} | {cp} | {note}"


def _render_bundle_passage(txs: list[dict], kyc: dict) -> str:
    """Canonical [transactions] + [kyc_profile] passage. Same shape Stage 6
    produced during SFT for the behavioral / numeric task user messages."""
    max_transactions = 60
    tx_list = list(txs or [])
    shown = tx_list[:max_transactions]
    lines: list[str] = []
    lines.append("[transactions]")
    if shown:
        header = (
            "date".ljust(_W_DATE)
            + " | " + "channel".ljust(_W_CHAN)
            + " | " + "amount".rjust(_W_AMT)
            + " | " + "ccy".ljust(_W_CCY)
            + " | " + "counterparty".ljust(_W_CP)
            + " | notes"
        )
        lines.append(header)
        for t in shown:
            lines.append(_fmt_tx_row(t))
        if len(tx_list) > max_transactions:
            lines.append(f"[... {len(tx_list) - max_transactions} more transactions not shown ...]")
    else:
        lines.append("(no transactions)")
    lines.append("")
    lines.append("[kyc_profile]")
    for k in _KYC_FIELDS:
        lines.append(f"{k}: {kyc.get(k, '')}")
    return "\n".join(lines)


def _behavioral_passage(txs: list[dict], kyc: dict) -> str:
    return _render_bundle_passage(txs, kyc)


def _citation_passage(excerpt: dict) -> str:
    """Canonical `[policy_excerpt]` block. Same shape Stage 6's
    `_format_excerpts_block` emits during SFT."""
    section = str(excerpt.get("section", ""))
    source  = str(excerpt.get("source", ""))
    url     = str(excerpt.get("url", "") or "")
    text    = str(excerpt.get("text", ""))
    lines = ["[policy_excerpt]"]
    if source:  lines.append(f"source: {source}")
    if section: lines.append(f"section: {section}")
    if url:     lines.append(f"url: {url}")
    lines.append("")
    lines.append(text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic behavioral finding (Path A — matches SFT Stage-7 byte-for-byte)
# ---------------------------------------------------------------------------
def _ensure_behavioral_finding(aux: dict | None, txs: list[dict], kyc: dict) -> dict:
    """Return an `auxiliary_findings`-shaped dict whose ``behavioral``
    block is guaranteed to be present and deterministically computed
    from the raw (txs, kyc) — never hallucinated by the LLM.

    Matches the v3.1 SFT Stage-7 helper that generated gold training
    metrics for the behavioral task. Reusing it at runtime restores
    train-serve parity and is the precision lever that took custom-NIM
    F1 from 0.520 (LLM-hallucinated behavioral) → 0.769 (deterministic).
    """
    aux = dict(aux) if isinstance(aux, dict) else {}

    metrics_obj = compute_behavioral_metrics(txs, kyc)
    metrics = metrics_obj.model_dump() if hasattr(metrics_obj, "model_dump") else dict(metrics_obj)

    # Build a templated prose summary citing the metrics verbatim.
    entity_id = kyc.get("entity_id", "")
    chan_mix_str = json.dumps(metrics.get("channel_mix", {}))
    summary = (
        f"Entity {entity_id} executed {metrics.get('tx_count', 0)} transactions "
        f"totaling ${metrics.get('tx_total_usd', 0):,.2f} USD with channel mix "
        f"{chan_mix_str}. "
        f"The 24-hour velocity peak is {metrics.get('velocity_24h_max', 0)} "
        f"against a 30-day average of {metrics.get('velocity_24h_avg_30d', 0)}. "
        f"Volume vs declared monthly is {metrics.get('vs_declared_volume_ratio', 0)}x. "
        f"country_risk_max is {metrics.get('country_risk_max', 0)}; "
        f"loop_detected is {str(metrics.get('loop_detected', False)).lower()}."
    )
    finding = {
        "question": "Analyze the transactional activity bundle and produce a behavioral summary.",
        "summary":  summary,
        "metrics":  metrics,
        "evidence": (f"transactions[0..{max(0, len(txs) - 1)}]; "
                      "kyc_profile.expected_monthly_volume; "
                      "kyc_profile.incorporation_jurisdiction"),
    }
    aux.setdefault("behavioral", [])
    # Replace any prior behavioral with the deterministic one (sole element).
    aux["behavioral"] = [finding]
    return aux


def _statutory_inputs(typology: str, txs: list[dict], kyc: dict) -> tuple[str, str]:
    """Return (statute_text, fact_pattern) for the typology."""
    _, statute_text = STATUTE_BY_TYPOLOGY.get(
        typology,
        ("5318(g)", "31 U.S.C. § 5318(g) — Suspicious activity reporting requirements"),
    )
    bundle = _render_bundle_passage(txs, kyc)
    header = (
        f"Entity {kyc.get('entity_id', '')} ({kyc.get('entity_type', '')}) "
        f"in {kyc.get('incorporation_jurisdiction', '')}"
        f"; declared monthly volume ${float(kyc.get('expected_monthly_volume', 0) or 0):,.0f}"
        f"; observed {len(txs)} transactions in the investigation window."
    )
    fact = header + "\n\n" + bundle
    return statute_text, fact


# ---------------------------------------------------------------------------
# Registered NAT workflow function
# ---------------------------------------------------------------------------
@register_function(config_type=InvestigateCaseConfig)
async def investigate_case(config: InvestigateCaseConfig, builder: Builder):
    # Resolve the function group + look up each child function on it.
    # FunctionGroup namespaces its functions as `<group>__<fn>`.
    data_tools_group = await builder.get_function_group(config.data_tools)
    all_fns = await data_tools_group.get_all_functions()
    group_prefix = data_tools_group.instance_name + "__"
    get_tx        = all_fns[group_prefix + "get_transactions"]
    get_kyc       = all_fns[group_prefix + "get_kyc"]
    screen        = all_fns[group_prefix + "screen_sanctions"]
    retrieve_pol  = all_fns[group_prefix + "retrieve_policy"]
    get_sop       = all_fns[group_prefix + "get_sop"]

    hints_fn      = await builder.get_function(config.compute_hints_fn)
    aux_b         = await builder.get_function(config.aux_behavioral_fn)
    aux_n         = await builder.get_function(config.aux_numeric_fn)
    aux_c         = await builder.get_function(config.aux_citation_fn)
    aux_s         = await builder.get_function(config.aux_statutory_fn)
    aux_gate      = await builder.get_function(config.aux_gate_fn)
    sar_caller    = await builder.get_function(config.sar_caller_fn)

    async def _safe(name: str, coro):
        """Run an aux call; swallow exceptions if fail_open_on_aux_error."""
        try:
            return await coro
        except Exception as e:
            logger.warning("aux %s failed: %s", name, e)
            if not config.fail_open_on_aux_error:
                raise
            return {"error": "exception", "details": str(e)[:300]}

    async def _run(args: InvestigateCaseInput) -> dict:
        if args is None:
            args = InvestigateCaseInput()
        dp = get_data_plane(os.environ.get("NAT_AML_DATA_DIR", "./data"))

        t0 = time.time()
        started = datetime.now(timezone.utc).isoformat()

        # --- resolve the alert (from manifest or passed payload) -----------
        case_id = args.case_id
        if case_id:
            alert = next((m for m in dp.manifest() if m["case_id"] == case_id), None)
            if alert is None:
                return {"error": f"case_id not in manifest: {case_id}",
                         "case_id": case_id}
        elif args.alert_id and args.entity_id:
            alert = {
                "case_id":  args.alert_id,
                "alert_id": args.alert_id,
                "entity_id": args.entity_id,
                "investigation_window_start": args.investigation_window_start or "2026-02-01",
                "investigation_window_end":   args.investigation_window_end   or "2026-12-31",
                "trigger_summary": args.trigger_summary or "",
            }
            case_id = alert["case_id"]
        else:
            return {"error": "must supply at least case_id (and optionally a full alert payload)"}

        entity_id = alert["entity_id"]
        ws = alert["investigation_window_start"]
        we = alert["investigation_window_end"]

        # ---- Phase 1: data fetch -----------------------------------------
        tx_resp, kyc_resp = await asyncio.gather(
            get_tx.ainvoke({"entity_id": entity_id, "window_start": ws, "window_end": we}),
            get_kyc.ainvoke({"entity_id": entity_id}),
        )
        tx  = tx_resp if isinstance(tx_resp, list) else (tx_resp.get("items") or tx_resp or [])
        kyc = kyc_resp if isinstance(kyc_resp, dict) else {}

        # Counterparty screening (deduped)
        unique_cps: list[str] = []
        seen: set[str] = set()
        for t in tx:
            cp = (t.get("counterparty") or "").strip()
            if cp and cp not in seen:
                seen.add(cp)
                unique_cps.append(cp)
        screens = await asyncio.gather(*[
            screen.ainvoke({"name": cp})
            for cp in unique_cps[:config.max_counterparty_screens]
        ])
        hit_list: list[dict] = []
        hit_seen: set[tuple] = set()
        for resp in screens:
            hits = resp if isinstance(resp, list) else (resp.get("items") or [])
            for h in hits:
                if not isinstance(h, dict):
                    continue
                key = (h.get("name"), h.get("list"))
                if key in hit_seen:
                    continue
                hit_seen.add(key)
                hit_list.append(h)

        # ---- Phase 2: internal typology guess (routing only) -------------
        hints = await hints_fn.ainvoke({
            "transactions":       tx,
            "kyc_profile":        kyc,
            "sanctions_pep_hits": hit_list,
            "trigger_summary":    alert.get("trigger_summary", ""),
        })
        typology            = hints.get("typology_inferred",   "none")
        regulatory_frame    = hints.get("regulatory_frame",    "benign")
        decision_target     = hints.get("decision_target",     "not_suspicious")
        activity_descriptor = hints.get("activity_descriptor", "")
        semantic_profile    = hints.get("semantic_profile",    {})

        # ---- Phase 3: retrieval ------------------------------------------
        policy_focused, policy_broad, sops = await asyncio.gather(
            retrieve_pol.ainvoke({"typology": typology, "k": 3}),
            retrieve_pol.ainvoke({"typology": "none",    "k": 2}),
            get_sop.ainvoke({"typology": typology}),
        )
        policy = list(policy_focused or []) + list(policy_broad or [])
        sops   = list(sops or [])

        # ---- Phase 4: aux skill calls (3 LLM + 1 Python) -----------------
        behavioral_q = "Provide a behavioral summary with the metrics described in the schema."
        numeric_q    = NUMERIC_QUESTION_BY_TYPOLOGY.get(
            typology,
            "Summarize transaction volumes and compare to KYC declared monthly volume.",
        )
        # citation passage — first focused policy excerpt, fallback if none
        if policy_focused:
            cit_passage = _citation_passage(policy_focused[0])
            cit_question = (
                f"What does {str(policy_focused[0].get('section') or 'the passage')}"
                f" say about the activity described?"
            )
        else:
            cit_passage = "(no policy excerpts available)"
            cit_question = "No policy excerpts were retrieved; respond accordingly."

        # statutory inputs
        if typology in STATUTE_BY_TYPOLOGY:
            statute_text, fact_pattern = _statutory_inputs(typology, tx, kyc)
            stat_question = f"Does the conduct described fall within {statute_text}?"
        else:
            statute_text = ""
            fact_pattern = _render_bundle_passage(tx, kyc)
            stat_question = "No statute mapped for this typology; respond accordingly."

        # Behavioral mode switch — Path A (deterministic Python, default)
        # vs Path B (LLM-driven with metrics injected). Path A is the
        # default per the SFT-skew fix.
        beh_mode = os.environ.get("NAT_AML_BEHAVIORAL_MODE", "python_only").strip().lower()

        if beh_mode == "python_metrics_llm_summary":
            # Pre-compute metrics, inject into the passage, let LLM write summary.
            metrics = compute_behavioral_metrics(tx, kyc).model_dump()
            passage_with_metrics = (
                _behavioral_passage(tx, kyc)
                + "\n\n[precomputed_metrics]\n"
                + json.dumps(metrics, indent=2)
            )
            beh_r, num_r, cit_r, stat_r = await asyncio.gather(
                _safe("behavioral", aux_b.ainvoke({"passage": passage_with_metrics,
                                                      "question": behavioral_q})),
                _safe("numeric",    aux_n.ainvoke({"passage": _behavioral_passage(tx, kyc),
                                                      "question": numeric_q})),
                _safe("citation",   aux_c.ainvoke({"passage": cit_passage,
                                                      "question": cit_question})),
                _safe("statutory",  aux_s.ainvoke({"statute": statute_text,
                                                      "fact_pattern": fact_pattern,
                                                      "question": stat_question})),
            )
            # Defense in depth: overwrite the LLM's metrics with Python truth.
            if isinstance(beh_r, dict) and "error" not in beh_r:
                beh_r["metrics"] = metrics
        else:
            # Default Path A — Python-only deterministic behavioral.
            beh_r = _ensure_behavioral_finding(None, tx, kyc)["behavioral"][0]
            num_r, cit_r, stat_r = await asyncio.gather(
                _safe("numeric",    aux_n.ainvoke({"passage": _behavioral_passage(tx, kyc),
                                                      "question": numeric_q})),
                _safe("citation",   aux_c.ainvoke({"passage": cit_passage,
                                                      "question": cit_question})),
                _safe("statutory",  aux_s.ainvoke({"statute": statute_text,
                                                      "fact_pattern": fact_pattern,
                                                      "question": stat_question})),
            )

        aux_responses_raw = {
            "behavioral": beh_r, "numeric": num_r,
            "citation":   cit_r, "statutory": stat_r,
        }

        # ---- Phase 5: aux gate -------------------------------------------
        if config.enable_aux_gate:
            gate_resp = await aux_gate.ainvoke({
                "findings":        aux_responses_raw,
                "transactions":    tx,
                "kyc":             kyc,
                "policy_excerpts": policy,
                "typology":        typology,
            })
            aux_findings    = gate_resp.get("auxiliary_findings") or {}
            gate_decisions  = gate_resp.get("decisions")          or []
        else:
            # Gate disabled — inline raw findings as the single accepted finding
            # in each task list. Schema-failed ones are dropped silently.
            aux_findings = {}
            for task in ("behavioral", "numeric", "citation", "statutory"):
                v = aux_responses_raw.get(task)
                if isinstance(v, dict) and "error" not in v:
                    aux_findings[task] = [v]
                else:
                    aux_findings[task] = []
            gate_decisions = [{"task": k, "used": bool(v), "reason": "gate-disabled"}
                              for k, v in aux_findings.items()]

        # Path A: re-apply deterministic behavioral AFTER the gate, so it
        # is always present (the gate can't drop it).
        if (beh_mode == "python_only"
                or os.environ.get("NAT_AML_DETERMINISTIC_BEHAVIORAL", "1") in ("1", "true", "yes")):
            ensured = _ensure_behavioral_finding(aux_findings, tx, kyc)
            aux_findings["behavioral"] = ensured["behavioral"]
            # Decision audit: mark the override.
            gate_decisions = [d for d in gate_decisions if d.get("task") != "behavioral"]
            gate_decisions.append({"task": "behavioral", "used": True,
                                    "reason": "deterministic_sft_replay",
                                    "override": True})

        # ---- Phase 6: SAR judgment ---------------------------------------
        sar_resp = await sar_caller.ainvoke({
            "transactions":       tx,
            "kyc_profile":        kyc,
            "sanctions_pep_hits": hit_list,
            "policy_excerpts":    policy,
            "sop_excerpts":       sops,
            "auxiliary_findings": aux_findings,
        })

        finished = datetime.now(timezone.utc).isoformat()
        wall_clock_ms = round((time.time() - t0) * 1000, 1)

        trace = {
            "case_id":       case_id,
            "alert_id":      alert.get("alert_id", ""),
            "entity_id":     entity_id,
            "started_at":    started,
            "finished_at":   finished,
            "wall_clock_ms": wall_clock_ms,

            "transactions":       tx,
            "kyc_profile":        kyc,
            "sanctions_pep_hits": hit_list,
            "policy_excerpts":    policy,
            "sop_excerpts":       sops,

            "semantic_profile":     semantic_profile,
            "typology_hypothesis":  typology,
            "activity_descriptor":  activity_descriptor,
            "decision_target":      decision_target,

            "planner_plan":       None,
            "orchestrator_calls": [],

            "aux_responses_raw":  aux_responses_raw,
            "aux_gate_decisions": gate_decisions,
            "auxiliary_findings": aux_findings,

            "sar_user_message": sar_resp.get("user_message", ""),
            "sar_raw_text":     sar_resp.get("raw_text", ""),
            "sar_output":       {
                "is_suspicious": sar_resp.get("is_suspicious"),
                "suspicious_activity_report": sar_resp.get("suspicious_activity_report", ""),
            },
            "sar_parse_error":  sar_resp.get("parse_error"),
            "sar_is_suspicious": sar_resp.get("is_suspicious"),
            "sar_narrative":     sar_resp.get("suspicious_activity_report", ""),

            "judge_enabled": config.enable_aux_gate,
            "error":         None,
        }

        # Persist
        try:
            write_trace(trace, dp.traces_dir)
        except Exception as e:
            logger.warning("write_trace failed for %s: %s", case_id, e)

        return trace

    yield FunctionInfo.from_fn(
        _run,
        description=("Deterministic 7-phase investigation workflow — "
                      "end-to-end on one alert."),
        input_schema=InvestigateCaseInput,
    )
