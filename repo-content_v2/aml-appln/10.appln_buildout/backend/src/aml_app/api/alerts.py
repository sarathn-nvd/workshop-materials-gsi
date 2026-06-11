"""Alert queue / case management routes.

- list_alerts        GET  /api/alerts
- get_alert          GET  /api/alerts/{alert_id}
- post_disposition   POST /api/alerts/{alert_id}/disposition
- alerts_stats       GET  /api/alerts/stats
"""

import json
import os
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

from aml_app.utils.data_loader import get_data_plane
from aml_app.workflow.trace import read_trace

_ENV_DATA_DIR_KEY = "NAT_AML_DATA_DIR"


def _dp():
    data_dir = os.environ.get(_ENV_DATA_DIR_KEY, "./data")
    return get_data_plane(data_dir)


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------
class ListAlertsInput(BaseModel):
    status: Optional[Literal["open", "in_progress", "closed"]] = None
    typology_hypothesis: Optional[str] = None
    q: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


def _alert_status(case_id: str, dp) -> str:
    if (dp.dispositions_dir / f"{case_id}.json").exists():
        return "closed"
    if (dp.traces_dir / f"{case_id}.json").exists():
        return "in_progress"
    return "open"


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------
class ListAlertsConfig(FunctionBaseConfig, name="list_alerts"):
    pass


@register_function(config_type=ListAlertsConfig)
async def list_alerts(config: ListAlertsConfig, builder: Builder):
    async def _run(args: ListAlertsInput) -> dict:
        # NAT 1.5 GET handler passes None for args (it doesn't bind
        # query/path params to non-Empty schemas). Use defaults when None.
        if args is None:
            args = ListAlertsInput()
        dp = _dp()
        alerts = list(dp.manifest())
        out: list[dict] = []
        for a in alerts:
            status = _alert_status(a["case_id"], dp)
            out.append({**a, "status": status})
        if args.status:
            out = [a for a in out if a["status"] == args.status]
        if args.q:
            q = args.q.lower()
            out = [a for a in out if q in a.get("trigger_summary", "").lower()
                   or q in a["entity_id"].lower() or q in a["alert_id"].lower()]
        if args.typology_hypothesis:
            # Approximate filter: peek into trace if present
            kept = []
            for a in out:
                tr = read_trace(a["case_id"], dp.traces_dir)
                if tr and tr.get("typology_hypothesis") == args.typology_hypothesis:
                    kept.append(a)
            out = kept
        total = len(out)
        page = out[args.offset:args.offset + args.limit]
        return {"total": total, "limit": args.limit, "offset": args.offset, "items": page}

    yield FunctionInfo.from_fn(_run, description="List alerts from the demo manifest.",
                                input_schema=ListAlertsInput)


class GetAlertInput(BaseModel):
    alert_id: str


class GetAlertConfig(FunctionBaseConfig, name="get_alert"):
    pass


@register_function(config_type=GetAlertConfig)
async def get_alert(config: GetAlertConfig, builder: Builder):
    async def _run(args: GetAlertInput) -> dict:
        dp = _dp()
        for a in dp.manifest():
            if a["alert_id"] == args.alert_id or a["case_id"] == args.alert_id:
                trace = read_trace(a["case_id"], dp.traces_dir)
                disp_path = dp.dispositions_dir / f"{a['case_id']}.json"
                disposition = (
                    json.loads(disp_path.read_text()) if disp_path.exists() else None
                )
                entity_id = a["entity_id"]
                kyc = dp.kyc().get(entity_id)
                kyc_snippet = None
                if kyc is not None:
                    kyc_snippet = {
                        k: kyc.get(k)
                        for k in ("entity_id", "entity_type", "business_purpose",
                                  "risk_rating", "incorporation_jurisdiction")
                    }
                return {
                    "alert": a,
                    "status": _alert_status(a["case_id"], dp),
                    "kyc_snippet": kyc_snippet,
                    "trace": trace,
                    "disposition": disposition,
                }
        return {"error": f"alert not found: {args.alert_id}"}

    yield FunctionInfo.from_fn(_run, description="Fetch one alert with snippets and trace.",
                                input_schema=GetAlertInput)


class PostDispositionInput(BaseModel):
    alert_id: str
    verdict: Literal["file_sar", "dismiss", "escalate"]
    note: str = ""


class PostDispositionConfig(FunctionBaseConfig, name="post_disposition"):
    pass


@register_function(config_type=PostDispositionConfig)
async def post_disposition(config: PostDispositionConfig, builder: Builder):
    async def _run(args: PostDispositionInput) -> dict:
        dp = _dp()
        case_id = None
        for a in dp.manifest():
            if a["alert_id"] == args.alert_id or a["case_id"] == args.alert_id:
                case_id = a["case_id"]
                break
        if case_id is None:
            return {"error": f"alert not found: {args.alert_id}"}
        path = dp.dispositions_dir / f"{case_id}.json"
        obj = {
            "case_id": case_id,
            "alert_id": args.alert_id,
            "verdict": args.verdict,
            "note": args.note,
            "ts": time.time(),
        }
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return {"ok": True, "path": str(path)}

    yield FunctionInfo.from_fn(_run, description="Record an analyst disposition for an alert.",
                                input_schema=PostDispositionInput)


class AlertsStatsInput(BaseModel):
    pass


class AlertsStatsConfig(FunctionBaseConfig, name="alerts_stats"):
    pass


@register_function(config_type=AlertsStatsConfig)
async def alerts_stats(config: AlertsStatsConfig, builder: Builder):
    async def _run(args: AlertsStatsInput) -> dict:
        if args is None:
            args = AlertsStatsInput()
        dp = _dp()
        alerts = dp.manifest()
        status_counter = Counter()
        typology_counter = Counter()
        for a in alerts:
            status_counter[_alert_status(a["case_id"], dp)] += 1
            tr = read_trace(a["case_id"], dp.traces_dir)
            typology_counter[tr.get("typology_hypothesis") if tr else "unknown"] += 1
        return {
            "total": len(alerts),
            "by_status": dict(status_counter),
            "by_typology": dict(typology_counter),
        }

    yield FunctionInfo.from_fn(_run, description="Summary counts for the alert queue.",
                                input_schema=AlertsStatsInput)
