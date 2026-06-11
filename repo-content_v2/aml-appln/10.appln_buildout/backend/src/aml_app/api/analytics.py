"""Analytics dashboard routes."""

import json
import os
from collections import Counter
from collections.abc import AsyncGenerator
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from aml_app.utils.data_loader import get_data_plane


_ENV_DATA_DIR_KEY = "NAT_AML_DATA_DIR"


def _dp():
    return get_data_plane(os.environ.get(_ENV_DATA_DIR_KEY, "./data"))


class Empty(BaseModel):
    pass


def _list_traces() -> list[dict]:
    dp = _dp()
    out = []
    for p in dp.traces_dir.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


class AnalyticsOverviewConfig(FunctionBaseConfig, name="analytics_overview"):
    pass


@register_function(config_type=AnalyticsOverviewConfig)
async def analytics_overview(config: AnalyticsOverviewConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        alerts = dp.manifest()
        n_total = len(alerts)
        traces = _list_traces()
        n_closed = sum(1 for _ in dp.dispositions_dir.glob("*.json"))
        n_run = len(traces)
        n_open = n_total - max(n_run, n_closed)
        avg_lat = (
            round(sum(t.get("wall_clock_ms", 0.0) for t in traces) / max(1, n_run))
            if traces else 0
        )
        n_sars = sum(1 for t in traces if t.get("sar_is_suspicious"))
        return {
            "n_alerts_total": n_total,
            "n_alerts_open": n_open,
            "n_alerts_in_progress": n_run - n_closed,
            "n_alerts_closed": n_closed,
            "n_entities": len(dp.kyc()),
            "n_transactions": len(dp.transactions()),
            "n_sars_drafted": n_sars,
            "avg_case_latency_ms": avg_lat,
        }

    yield FunctionInfo.from_fn(_run, description="Top-line analytics cards.",
                                input_schema=Empty)


class AnalyticsTypologyConfig(FunctionBaseConfig, name="analytics_typology"):
    pass


@register_function(config_type=AnalyticsTypologyConfig)
async def analytics_typology(config: AnalyticsTypologyConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        strat = dp.stratification()
        per = strat.get("per_typology", {})
        from_traces = Counter(t.get("typology_hypothesis", "unknown")
                              for t in _list_traces())
        return {"seeded": per, "from_traces": dict(from_traces)}

    yield FunctionInfo.from_fn(_run, description="Per-typology distribution.",
                                input_schema=Empty)


class AnalyticsRiskHeatmapConfig(FunctionBaseConfig, name="analytics_risk_heatmap"):
    pass


@register_function(config_type=AnalyticsRiskHeatmapConfig)
async def analytics_risk_heatmap(config: AnalyticsRiskHeatmapConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        entities = dp.kyc()
        manifest = dp.manifest()
        by_juris: Counter = Counter()
        risk_counter: Counter = Counter()
        for a in manifest:
            e = entities.get(a["entity_id"], {})
            by_juris[e.get("incorporation_jurisdiction", "UNKNOWN")] += 1
            risk_counter[e.get("risk_rating", "unknown")] += 1
        return {
            "alerts_by_jurisdiction": dict(by_juris),
            "alerts_by_risk_rating": dict(risk_counter),
        }

    yield FunctionInfo.from_fn(_run, description="Heatmap data by jurisdiction / risk.",
                                input_schema=Empty)


class AnalyticsTimelineConfig(FunctionBaseConfig, name="analytics_timeline"):
    pass


@register_function(config_type=AnalyticsTimelineConfig)
async def analytics_timeline(config: AnalyticsTimelineConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        df = dp.transactions()
        by_day = df.groupby(df["_date_parsed"].dt.date).size()
        return {
            "daily_tx_count": [
                {"date": str(d), "n_tx": int(n)} for d, n in by_day.items()
            ],
        }

    yield FunctionInfo.from_fn(_run, description="Tx counts per day.",
                                input_schema=Empty)


class AnalyticsChannelMixConfig(FunctionBaseConfig, name="analytics_channel_mix"):
    pass


@register_function(config_type=AnalyticsChannelMixConfig)
async def analytics_channel_mix(config: AnalyticsChannelMixConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        df = dp.transactions()
        if "typology_tag" in df.columns:
            grp = df.groupby(["typology_tag", "channel"]).size().unstack(fill_value=0)
            return {"by_typology": json.loads(grp.to_json(orient="index"))}
        return {"by_typology": {}}

    yield FunctionInfo.from_fn(_run, description="Channel mix per typology.",
                                input_schema=Empty)


class AnalyticsTopCpConfig(FunctionBaseConfig, name="analytics_top_cp"):
    pass


@register_function(config_type=AnalyticsTopCpConfig)
async def analytics_top_cp(config: AnalyticsTopCpConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        df = dp.transactions()
        cp_stats = df.groupby("counterparty").agg(
            n_tx=("amount", "count"),
            total_usd=("amount", "sum"),
        ).sort_values("total_usd", ascending=False).head(20)
        return {
            "top_by_volume": [
                {"counterparty": idx, "n_tx": int(r["n_tx"]),
                 "total_usd": round(float(r["total_usd"]), 2)}
                for idx, r in cp_stats.iterrows()
            ]
        }

    yield FunctionInfo.from_fn(_run, description="Top counterparties by volume.",
                                input_schema=Empty)


class AnalyticsAuxUsageConfig(FunctionBaseConfig, name="analytics_aux_usage"):
    pass


@register_function(config_type=AnalyticsAuxUsageConfig)
async def analytics_aux_usage(config: AnalyticsAuxUsageConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        used = Counter()
        dropped = Counter()
        n = 0
        for t in _list_traces():
            n += 1
            for d in t.get("aux_gate_decisions", []):
                task = d.get("task", "?")
                (used if d.get("used") else dropped)[task] += 1
        return {"n_cases": n, "used": dict(used), "dropped": dict(dropped)}

    yield FunctionInfo.from_fn(_run,
        description="How often each aux finding was USED vs DROPPED.",
        input_schema=Empty)


class AnalyticsAgentPerfConfig(FunctionBaseConfig, name="analytics_agent_perf"):
    pass


@register_function(config_type=AnalyticsAgentPerfConfig)
async def analytics_agent_perf(config: AnalyticsAgentPerfConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        keys = dp.eval_keys()
        traces = {t["case_id"]: t for t in _list_traces() if "case_id" in t}
        per_typology: dict[str, dict] = {}
        for case_id, gt in keys.items():
            tr = traces.get(case_id)
            if tr is None:
                continue
            typ = gt.get("expected_typology", "none")
            bucket = per_typology.setdefault(typ, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            pred = bool(tr.get("sar_is_suspicious"))
            actual = bool(gt.get("expected_label"))
            if pred and actual:
                bucket["tp"] += 1
            elif pred and not actual:
                bucket["fp"] += 1
            elif not pred and actual:
                bucket["fn"] += 1
            else:
                bucket["tn"] += 1
        # derive recall / precision
        for typ, b in per_typology.items():
            denom_r = b["tp"] + b["fn"]
            denom_p = b["tp"] + b["fp"]
            b["recall"] = round(b["tp"] / denom_r, 3) if denom_r else None
            b["precision"] = round(b["tp"] / denom_p, 3) if denom_p else None
        return {"per_typology": per_typology, "n_traces": len(traces)}

    yield FunctionInfo.from_fn(_run,
        description="Per-typology recall / precision against eval_keys.",
        input_schema=Empty)


class AnalyticsProfileConfig(FunctionBaseConfig, name="analytics_profile"):
    pass


@register_function(config_type=AnalyticsProfileConfig)
async def analytics_profile(config: AnalyticsProfileConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        # Pull whatever the NAT profiler has emitted into the working dir
        candidates = [
            Path(".tmp/nat/profiler"),
            Path(".profiler"),
            Path("data/profiler"),
        ]
        found = []
        for c in candidates:
            if c.exists():
                for f in c.rglob("*.json"):
                    found.append(str(f))
        return {
            "n_profiler_files": len(found),
            "files": found[:50],
            "note": (
                "NAT profiler data is captured when the workflow is run via "
                "`nat eval --profile=...`; raw artifacts are listed here."
            ),
        }

    yield FunctionInfo.from_fn(_run,
        description="NAT profiler summary artifacts.",
        input_schema=Empty)
