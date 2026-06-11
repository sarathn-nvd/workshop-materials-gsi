"""Misc routes: policy, sops, sanctions, skills, demo, system, health,
investigation trace fetch."""
import glob
import json
import logging
import os
import shutil
import time
from collections import Counter
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from aml_app.tools.data_tools import (
    GetSopInput, RetrievePolicyInput, ScreenSanctionsInput,
    _get_sop, _retrieve_policy, _screen_sanctions,
)
from aml_app.utils.data_loader import get_data_plane
from aml_app.workflow.trace import read_trace


logger = logging.getLogger("aml_app.api.misc")


_ENV_DATA_DIR_KEY = "NAT_AML_DATA_DIR"
_ENV_EVAL_TOKEN   = "NAT_AML_EVAL_TOKEN"


def _dp():
    return get_data_plane(os.environ.get(_ENV_DATA_DIR_KEY, "./data"))


class Empty(BaseModel):
    pass


def _check_token(supplied: str) -> bool:
    """Mirror the bearer-token gate used by every /api/demo/eval/* route.
    Returns True if the request is authorized (token matches or gate is
    disabled)."""
    expected = os.environ.get(_ENV_EVAL_TOKEN, "").strip()
    if not expected:
        return True
    return (supplied or "").strip() == expected


# ---------------------------------------------------------------------------
# 1. Policy / sops / sanctions routes
# ---------------------------------------------------------------------------
class SearchPolicyInput(BaseModel):
    typology: str
    q: Optional[str] = None
    k: int = Field(default=4, ge=1, le=20)


class SearchPolicyConfig(FunctionBaseConfig, name="search_policy"):
    pass


@register_function(config_type=SearchPolicyConfig)
async def search_policy(config: SearchPolicyConfig, builder: Builder):
    async def _run(args: SearchPolicyInput) -> dict:
        if args is None:
            args = SearchPolicyInput(typology="none")
        dp = _dp()
        rows = _retrieve_policy(dp, RetrievePolicyInput(typology=args.typology, k=args.k))
        items = [r.model_dump() for r in rows]
        if args.q:
            q = args.q.lower()
            for it in items:
                pos = it.get("text", "").lower().find(q)
                if pos >= 0:
                    it["match_offset"] = pos
        return {"typology": args.typology, "n": len(items), "items": items}
    yield FunctionInfo.from_fn(_run, description="Query the policy RAG.",
                                input_schema=SearchPolicyInput)


class ListPolicySourcesConfig(FunctionBaseConfig, name="list_policy_sources"):
    pass


@register_function(config_type=ListPolicySourcesConfig)
async def list_policy_sources(config: ListPolicySourcesConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        df = dp.policy_chunks()
        counts = df["source"].value_counts().to_dict()
        return {"sources": {k: int(v) for k, v in counts.items()}}
    yield FunctionInfo.from_fn(_run, description="Policy corpus distribution by source.",
                                input_schema=Empty)


class ListSopsConfig(FunctionBaseConfig, name="list_sops"):
    pass


@register_function(config_type=ListSopsConfig)
async def list_sops(config: ListSopsConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        sops = dp.sops()
        return {"sops": sorted(sops.keys())}
    yield FunctionInfo.from_fn(_run, description="List available typology SOPs.",
                                input_schema=Empty)


class GetSopBodyInput(BaseModel):
    sop_id: str


class GetSopBodyConfig(FunctionBaseConfig, name="get_sop_body"):
    pass


@register_function(config_type=GetSopBodyConfig)
async def get_sop_body(config: GetSopBodyConfig, builder: Builder):
    async def _run(args: GetSopBodyInput) -> dict:
        if args is None or not args.sop_id:
            return {"error": "missing sop_id (NAT GET handler doesn't bind path params; use POST)"}
        dp = _dp()
        sops = dp.sops()
        sections = sops.get(args.sop_id)
        if not sections:
            return {"error": f"sop not found: {args.sop_id}"}
        return {"sop_id": args.sop_id,
                "sections": [{"title": title, "text": body} for title, body in sections]}
    yield FunctionInfo.from_fn(_run, description="Return one SOP body as markdown.",
                                input_schema=GetSopBodyInput)


class ScreenNameConfig(FunctionBaseConfig, name="screen_name"):
    pass


@register_function(config_type=ScreenNameConfig)
async def screen_name(config: ScreenNameConfig, builder: Builder):
    async def _run(args: ScreenSanctionsInput) -> dict:
        if args is None:
            return {"error": "missing screening payload"}
        dp = _dp()
        hits = _screen_sanctions(dp, args)
        return {"name": args.name, "n": len(hits),
                "items": [h.model_dump() for h in hits]}
    yield FunctionInfo.from_fn(_run, description="Fuzzy sanctions / PEP screen.",
                                input_schema=ScreenSanctionsInput)


# ---------------------------------------------------------------------------
# 2. Investigation trace fetch
# ---------------------------------------------------------------------------
class GetTraceInput(BaseModel):
    case_id: str


class GetTraceConfig(FunctionBaseConfig, name="get_trace"):
    pass


@register_function(config_type=GetTraceConfig)
async def get_trace(config: GetTraceConfig, builder: Builder):
    async def _run(args: GetTraceInput) -> dict:
        if args is None or not args.case_id:
            return {"error": "missing case_id (NAT GET handler doesn't bind path params; use POST)"}
        dp = _dp()
        tr = read_trace(args.case_id, dp.traces_dir)
        if tr is None:
            return {"error": f"trace not found: {args.case_id}"}
        return tr
    yield FunctionInfo.from_fn(_run, description="Retrieve persisted case trace.",
                                input_schema=GetTraceInput)


# ---------------------------------------------------------------------------
# 3. Demo eval — scoring helpers (used by /api/demo/eval/*)
# ---------------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _f1(precision: float, recall: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _grounded(narrative: str) -> bool:
    """Cheap heuristic: does the narrative cite at least one specific
    evidence anchor (a $-amount, a date, or a statute reference)?"""
    if not narrative:
        return False
    import re
    if re.search(r"\$\s?\d", narrative):
        return True
    if re.search(r"\d{4}-\d{2}-\d{2}", narrative):
        return True
    if re.search(r"\b\d{1,4}\s*USC\b|\bUSC\s*§?\s*\d", narrative, re.IGNORECASE):
        return True
    if re.search(r"\bCFR\b|\bFinCEN\b|\bFATF\b|\bFFIEC\b", narrative):
        return True
    return False


def _compare_one(case_id: str, gt: dict, trace: Optional[dict]) -> dict:
    """Compare one trace against its ground-truth row from eval_keys."""
    if trace is None or trace.get("error"):
        return {
            "case_id": case_id,
            "entity_id": gt.get("entity_id"),
            "ground_truth": {"label": gt.get("expected_label"),
                              "typology": gt.get("expected_typology"),
                              "near_miss": gt.get("near_miss", False)},
            "prediction": None,
            "is_correct": None,
            "typology_correct": None,
            "outcome": "no_prediction",
            "narrative_grounded": None,
            "wall_clock_ms": None,
        }
    pred_label = trace.get("sar_is_suspicious")
    if pred_label is None:
        outcome = "no_prediction"
    elif gt.get("expected_label"):
        outcome = "TP" if pred_label else "FN"
    else:
        outcome = "FP" if pred_label else "TN"
    narrative = trace.get("sar_narrative", "") or ""
    return {
        "case_id": case_id,
        "entity_id": gt.get("entity_id"),
        "ground_truth": {"label": gt.get("expected_label"),
                          "typology": gt.get("expected_typology"),
                          "near_miss": gt.get("near_miss", False)},
        "prediction": {"label": pred_label,
                        "typology": trace.get("typology_hypothesis"),
                        "narrative_excerpt": narrative[:200]},
        "is_correct": pred_label == gt.get("expected_label") if pred_label is not None else None,
        "typology_correct": (trace.get("typology_hypothesis") == gt.get("expected_typology")
                              if trace.get("typology_hypothesis") else None),
        "outcome": outcome,
        "narrative_grounded": _grounded(narrative),
        "wall_clock_ms": trace.get("wall_clock_ms"),
    }


def _score_run(traces_dir: Path, keys: dict[str, dict]) -> dict:
    """Score every trace in `traces_dir` against `keys` (case_id → gt row).
    Returns a scorecard with confusion + metrics + latency."""
    tp = fp = tn = fn = 0
    fp_clean = fp_nm = tn_clean = tn_nm = 0
    grounded = 0
    n_pred = 0
    n_missing = 0
    parse_errors = 0
    latencies: list[float] = []
    for case_id, gt in keys.items():
        tr_path = traces_dir / f"{case_id}.json"
        if not tr_path.exists():
            n_missing += 1
            continue
        try:
            tr = json.loads(tr_path.read_text())
        except Exception:
            parse_errors += 1
            continue
        if tr.get("error") or tr.get("sar_parse_error"):
            parse_errors += 1
            continue
        pred = tr.get("sar_is_suspicious")
        if pred is None:
            n_missing += 1
            continue
        n_pred += 1
        if tr.get("wall_clock_ms"):
            latencies.append(float(tr["wall_clock_ms"]))
        if _grounded(tr.get("sar_narrative", "")):
            grounded += 1
        is_nm = gt.get("near_miss", False)
        if gt.get("expected_label"):
            if pred: tp += 1
            else:    fn += 1
        else:
            if pred:
                fp += 1
                if is_nm: fp_nm += 1
                else:     fp_clean += 1
            else:
                tn += 1
                if is_nm: tn_nm += 1
                else:     tn_clean += 1
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return {
        "n_predictions": n_pred, "n_missing": n_missing, "parse_errors": parse_errors,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "fp_breakdown": {"near_miss": fp_nm, "clean": fp_clean},
        "metrics": {
            "accuracy":  round(_safe_div(tp + tn, tp + fp + tn + fn), 4),
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(_f1(p, r), 4),
            "macro_f1":  round((_f1(p, r) + _f1(_safe_div(tn, tn+fn), _safe_div(tn, tn+fp))) / 2, 4),
            "false_positive_rate_clean":     round(_safe_div(fp_clean, fp_clean + tn_clean), 4),
            "false_positive_rate_near_miss": round(_safe_div(fp_nm, fp_nm + tn_nm), 4),
            "near_miss_specificity":         round(_safe_div(tn_nm, tn_nm + fp_nm), 4),
            "narrative_grounding_rate":      round(_safe_div(grounded, n_pred), 4),
        },
        "latency": {
            "avg_case_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "n_with_timing": len(latencies),
        },
    }


def _discover_runs(data_dir: Path) -> list[dict]:
    """Discover trace-snapshot directories under data_dir/. Includes the
    active `traces/` plus any sibling `traces_*` that has at least one JSON."""
    items: list[dict] = []
    if (data_dir / "traces").exists():
        n = len(list((data_dir / "traces").glob("*.json")))
        items.append({"name": "traces", "n_traces": n, "is_active": True})
    for child in sorted(data_dir.glob("traces_*")):
        if not child.is_dir():
            continue
        n = len(list(child.glob("*.json")))
        if n == 0:
            continue
        items.append({"name": child.name, "n_traces": n, "is_active": False})
    return items


def _resolve_run_dir(data_dir: Path, name: str) -> Path | None:
    """Validate that `name` refers to a known trace snapshot dir.
    Rejects path-traversal."""
    if not name or "/" in name or name.startswith(".."):
        return None
    if name != "traces" and not name.startswith("traces_"):
        return None
    path = (data_dir / name).resolve()
    if not str(path).startswith(str(data_dir.resolve())):
        return None
    if not path.exists() or not path.is_dir():
        return None
    return path


def _delta(a: float | int, b: float | int) -> dict:
    """{absolute, relative_pct} delta between two scalar values."""
    absolute = a - b
    rel = (absolute / b * 100.0) if b else 0.0
    return {"absolute": round(absolute, 4), "relative_pct": round(rel, 2)}


# ---------------------------------------------------------------------------
# Demo eval endpoints
# ---------------------------------------------------------------------------
class DemoEvalInput(BaseModel):
    token: str = ""


class DemoEvalConfig(FunctionBaseConfig, name="demo_eval"):
    pass


@register_function(config_type=DemoEvalConfig)
async def demo_eval(config: DemoEvalConfig, builder: Builder):
    async def _run(args: DemoEvalInput) -> dict:
        if args is None:
            args = DemoEvalInput()
        if not _check_token(args.token):
            return {"error": "unauthorized"}
        dp = _dp()
        keys = dp.eval_keys()
        if not keys:
            return {"error": "eval_keys.jsonl not found or empty"}
        return _score_run(dp.traces_dir, keys)
    yield FunctionInfo.from_fn(_run, description="Score persisted traces vs ground truth (gated).",
                                input_schema=DemoEvalInput)


class DemoEvalCasesInput(BaseModel):
    token: str = ""
    outcome:  Optional[Literal["TP", "FP", "TN", "FN", "no_prediction"]] = None
    correct:  Optional[bool] = None
    typology: Optional[str] = None
    limit: int = 200
    offset: int = 0


class DemoEvalCasesConfig(FunctionBaseConfig, name="demo_eval_cases"):
    pass


@register_function(config_type=DemoEvalCasesConfig)
async def demo_eval_cases(config: DemoEvalCasesConfig, builder: Builder):
    async def _run(args: DemoEvalCasesInput) -> dict:
        if args is None:
            args = DemoEvalCasesInput()
        if not _check_token(args.token):
            return {"error": "unauthorized"}
        dp = _dp()
        keys = dp.eval_keys()
        out: list[dict] = []
        for case_id, gt in keys.items():
            tr = read_trace(case_id, dp.traces_dir)
            row = _compare_one(case_id, gt, tr)
            if args.outcome and row["outcome"] != args.outcome:
                continue
            if args.correct is not None and row["is_correct"] != args.correct:
                continue
            if args.typology and row["ground_truth"]["typology"] != args.typology:
                continue
            out.append(row)
        total = len(out)
        page = out[args.offset:args.offset + args.limit]
        return {"total": total, "limit": args.limit, "offset": args.offset, "items": page}
    yield FunctionInfo.from_fn(_run, description="Per-case prediction-vs-truth list (gated).",
                                input_schema=DemoEvalCasesInput)


class DemoEvalCaseInput(BaseModel):
    case_id: str
    token: str = ""
    include_full_tool_outputs: bool = True


class DemoEvalCaseConfig(FunctionBaseConfig, name="demo_eval_case"):
    pass


@register_function(config_type=DemoEvalCaseConfig)
async def demo_eval_case(config: DemoEvalCaseConfig, builder: Builder):
    async def _run(args: DemoEvalCaseInput) -> dict:
        if args is None:
            return {"error": "missing case_id"}
        if not _check_token(args.token):
            return {"error": "unauthorized"}
        dp = _dp()
        keys = dp.eval_keys()
        gt = keys.get(args.case_id)
        if not gt:
            return {"error": f"case_id not found in ground truth: {args.case_id}"}
        tr = read_trace(args.case_id, dp.traces_dir)
        comparison = _compare_one(args.case_id, gt, tr)
        if tr is None:
            return {"comparison": comparison, "trace": None,
                    "note": "No persisted trace; run /api/investigation/run on this case first."}

        out: dict = {"comparison": comparison}
        if args.include_full_tool_outputs:
            out["tool_outputs"] = {
                "transactions":       tr.get("transactions", []),
                "kyc_profile":        tr.get("kyc_profile", {}),
                "sanctions_pep_hits": tr.get("sanctions_pep_hits", []),
                "policy_excerpts":    tr.get("policy_excerpts", []),
                "sop_excerpts":       tr.get("sop_excerpts", []),
                "semantic_profile":   tr.get("semantic_profile", {}),
                "compute_hints": {
                    "typology_hypothesis": tr.get("typology_hypothesis"),
                    "activity_descriptor": tr.get("activity_descriptor"),
                },
            }
        else:
            out["tool_outputs"] = {
                "n_transactions":    len(tr.get("transactions", [])),
                "n_policy_excerpts": len(tr.get("policy_excerpts", [])),
                "n_sop_excerpts":    len(tr.get("sop_excerpts", [])),
            }
        out["agentic"] = {"orchestrator_calls": tr.get("orchestrator_calls", [])}
        out["aux"] = {
            "responses_raw":      tr.get("aux_responses_raw", {}),
            "gate_decisions":     tr.get("aux_gate_decisions", []),
            "auxiliary_findings": tr.get("auxiliary_findings", {}),
        }
        out["sar"] = {
            "user_message":  tr.get("sar_user_message", "") if args.include_full_tool_outputs else "(omitted in compact mode)",
            "raw_text":      tr.get("sar_raw_text", "")    if args.include_full_tool_outputs else "(omitted in compact mode)",
            "parsed_output": tr.get("sar_output", {}),
            "parse_error":   tr.get("sar_parse_error"),
        }
        out["timing"] = {
            "started_at":    tr.get("started_at"),
            "finished_at":   tr.get("finished_at"),
            "wall_clock_ms": tr.get("wall_clock_ms"),
        }
        out["error"] = tr.get("error")
        return out
    yield FunctionInfo.from_fn(_run, description="Deep-dive on one eval case (gated).",
                                input_schema=DemoEvalCaseInput)


class DemoEvalRunsConfig(FunctionBaseConfig, name="demo_eval_runs"):
    pass


@register_function(config_type=DemoEvalRunsConfig)
async def demo_eval_runs(config: DemoEvalRunsConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        items = _discover_runs(dp.data_dir)
        return {"n_runs": len(items), "items": items}
    yield FunctionInfo.from_fn(_run, description="List trace-snapshot dirs.",
                                input_schema=Empty)


class DemoEvalCompareInput(BaseModel):
    token: str = ""
    run_a: str = "traces"
    run_b: str
    label_a: Optional[str] = None
    label_b: Optional[str] = None


class DemoEvalCompareConfig(FunctionBaseConfig, name="demo_eval_compare"):
    pass


@register_function(config_type=DemoEvalCompareConfig)
async def demo_eval_compare(config: DemoEvalCompareConfig, builder: Builder):
    async def _run(args: DemoEvalCompareInput) -> dict:
        if args is None or not getattr(args, "run_b", None):
            return {"error": "missing run_b"}
        if not _check_token(args.token):
            return {"error": "unauthorized"}
        dp = _dp()
        keys = dp.eval_keys()
        a_path = _resolve_run_dir(dp.data_dir, args.run_a)
        b_path = _resolve_run_dir(dp.data_dir, args.run_b)
        if a_path is None:  return {"error": "run_a_not_found", "run_a": args.run_a}
        if b_path is None:  return {"error": "run_b_not_found", "run_b": args.run_b}
        if a_path == b_path: return {"error": "run_a_and_run_b_are_same_directory"}

        score_a = _score_run(a_path, keys)
        score_b = _score_run(b_path, keys)

        # GT counts (computed once)
        n_sar    = sum(1 for k in keys.values() if k.get("expected_label"))
        n_no_sar = sum(1 for k in keys.values() if not k.get("expected_label"))
        n_nm     = sum(1 for k in keys.values() if k.get("near_miss"))

        diff_metrics = {}
        for m in score_a["metrics"]:
            diff_metrics[m] = _delta(score_a["metrics"][m], score_b["metrics"][m])
        diff_confusion = {
            k: score_a["confusion"][k] - score_b["confusion"][k]
            for k in score_a["confusion"]
        }
        diff_fp = {
            k: score_a["fp_breakdown"][k] - score_b["fp_breakdown"][k]
            for k in score_a["fp_breakdown"]
        }

        return {
            "ground_truth": {"n_total": len(keys), "n_sar": n_sar,
                              "n_no_sar": n_no_sar, "n_near_miss": n_nm},
            "run_a": {"name": args.run_a, "label": args.label_a or args.run_a, **score_a},
            "run_b": {"name": args.run_b, "label": args.label_b or args.run_b, **score_b},
            "diff": {
                "metrics":      diff_metrics,
                "confusion":    diff_confusion,
                "fp_breakdown": diff_fp,
                "latency_ms":   round(score_a["latency"]["avg_case_ms"] - score_b["latency"]["avg_case_ms"], 1),
            },
        }
    yield FunctionInfo.from_fn(_run, description="Compare two trace snapshots (gated).",
                                input_schema=DemoEvalCompareInput)


class DemoEvalModelComparisonInput(BaseModel):
    report: Optional[str] = None


class DemoEvalModelComparisonConfig(FunctionBaseConfig, name="demo_eval_model_comparison"):
    """Legacy stub. The active endpoint is registered in
    `aml_app/api/eval_comparison.py` as `demo_eval_model_comparison_report`
    (workflow.yaml routes `/api/demo/eval/model_comparison` there)."""


@register_function(config_type=DemoEvalModelComparisonConfig)
async def demo_eval_model_comparison(config: DemoEvalModelComparisonConfig, builder: Builder):
    async def _run(args: DemoEvalModelComparisonInput) -> dict:
        return {"error": "use_demo_eval_model_comparison_report",
                "report": args.report if args else None}
    yield FunctionInfo.from_fn(_run, description="Legacy stub; see eval_comparison.py.",
                                input_schema=DemoEvalModelComparisonInput)


# ---------------------------------------------------------------------------
# Demo seed: pre-load a baseline rollout into ./data/traces/
# ---------------------------------------------------------------------------
class DemoSeedTracesConfig(FunctionBaseConfig, name="demo_seed_traces"):
    pass


@register_function(config_type=DemoSeedTracesConfig)
async def demo_seed_traces(config: DemoSeedTracesConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        seed_dir = dp.data_dir / "seed_traces"
        target = dp.traces_dir
        if not seed_dir.exists():
            return {"n_seeds": 0, "n_written": 0,
                    "note": f"no seed_traces dir at {seed_dir}"}
        target.mkdir(parents=True, exist_ok=True)
        n_seeds = 0
        n_written = 0
        for src in sorted(seed_dir.glob("*.json")):
            n_seeds += 1
            dst = target / src.name
            if dst.exists():
                continue
            shutil.copyfile(src, dst)
            n_written += 1
        return {"n_seeds": n_seeds, "n_written": n_written}
    yield FunctionInfo.from_fn(_run, description="Pre-load baseline trace rollout.",
                                input_schema=Empty)


# ---------------------------------------------------------------------------
# System / health
# ---------------------------------------------------------------------------
class HealthConfig(FunctionBaseConfig, name="health"):
    pass


@register_function(config_type=HealthConfig)
async def health(config: HealthConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        try:
            dp = _dp()
            return {"ok": True,
                    "n_transactions": int(len(dp.transactions())),
                    "n_entities":     int(len(dp.kyc())),
                    "ts": time.time()}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    yield FunctionInfo.from_fn(_run, description="Liveness + data plane sanity.",
                                input_schema=Empty)


class SystemConfigConfig(FunctionBaseConfig, name="system_config"):
    pass


@register_function(config_type=SystemConfigConfig)
async def system_config(config: SystemConfigConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        try:
            tx_schema = dp.tx_schema()
        except Exception:
            tx_schema = {}
        try:
            tx_stats = dp.tx_stats()
        except Exception:
            tx_stats = {}
        try:
            kyc_schema = dp.kyc_schema()
        except Exception:
            kyc_schema = {}
        try:
            kyc_stats = dp.kyc_stats()
        except Exception:
            kyc_stats = {}
        try:
            policy_sources = dp.policy_chunks()["source"].value_counts().to_dict()
            policy_sources = {k: int(v) for k, v in policy_sources.items()}
        except Exception:
            policy_sources = {}
        try:
            sop_count = len(dp.sops())
        except Exception:
            sop_count = 0
        return {
            "data_dir": str(dp.data_dir),
            "transactions_schema": tx_schema,
            "transactions_stats":  tx_stats,
            "kyc_schema": kyc_schema,
            "kyc_stats":  kyc_stats,
            "policy_sources": policy_sources,
            "sop_count": sop_count,
        }
    yield FunctionInfo.from_fn(_run, description="Show wired data + model config.",
                                input_schema=Empty)


class SystemComponentsConfig(FunctionBaseConfig, name="system_components"):
    pass


@register_function(config_type=SystemComponentsConfig)
async def system_components(config: SystemComponentsConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        # Best-effort listing — pull from the NAT registry.
        try:
            from nat.cli.type_registry import GlobalTypeRegistry
            reg = GlobalTypeRegistry.get()
            fns = sorted({f.local_name for f in reg.get_registered_functions()})
            grps = sorted({g.local_name for g in reg.get_registered_function_groups()})
            llms = sorted({p.local_name for p in reg.get_registered_llm_providers()})
            return {"registered_function_groups": grps,
                    "registered_functions": fns,
                    "registered_llm_providers": llms}
        except Exception as e:
            return {"error": str(e)[:200]}
    yield FunctionInfo.from_fn(_run, description="NAT component inventory.",
                                input_schema=Empty)
