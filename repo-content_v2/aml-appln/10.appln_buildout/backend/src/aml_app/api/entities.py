"""Entity 360 routes.

- list_entities                 GET /api/entities
- get_entity                    GET /api/entities/{entity_id}
- get_entity_tx                 GET /api/entities/{entity_id}/transactions
- get_entity_behavioral         GET /api/entities/{entity_id}/behavioral_summary
- get_entity_risk               GET /api/entities/{entity_id}/risk_score
- get_entity_network            GET /api/entities/{entity_id}/network
- get_entity_timeline           GET /api/entities/{entity_id}/timeline
"""

import os
from collections import Counter
from collections.abc import AsyncGenerator
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from aml_app.common.behavioral_features import compute_behavioral_metrics
from aml_app.utils.data_loader import get_data_plane
from aml_app.utils.network_graph import build_entity_network
from aml_app.utils.risk_score import risk_score

_ENV_DATA_DIR_KEY = "NAT_AML_DATA_DIR"


def _dp():
    return get_data_plane(os.environ.get(_ENV_DATA_DIR_KEY, "./data"))


_STRIP_KYC = {"source_pool", "_archetype"}


def _clean_kyc(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _STRIP_KYC}


_STRIP_TX = {"source_pool", "typology_tag", "_date_parsed"}


def _clean_tx(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in _STRIP_TX} for r in rows]


# ---------------------------------------------------------------------------
class ListEntitiesInput(BaseModel):
    risk_rating: Optional[str] = None
    entity_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    q: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class ListEntitiesConfig(FunctionBaseConfig, name="list_entities"):
    pass


@register_function(config_type=ListEntitiesConfig)
async def list_entities(config: ListEntitiesConfig, builder: Builder):
    async def _run(args: ListEntitiesInput) -> dict:
        # NAT 1.5 GET handler passes None for args (no path/query binding
        # for non-Empty schemas). Use defaults when None.
        if args is None:
            args = ListEntitiesInput()
        dp = _dp()
        idx = dp.kyc()
        rows = list(idx.values())
        if args.risk_rating:
            rows = [r for r in rows if r.get("risk_rating") == args.risk_rating]
        if args.entity_type:
            rows = [r for r in rows if r.get("entity_type") == args.entity_type]
        if args.jurisdiction:
            rows = [r for r in rows if r.get("incorporation_jurisdiction") == args.jurisdiction]
        if args.q:
            q = args.q.lower()
            rows = [
                r for r in rows
                if q in str(r.get("entity_id", "")).lower()
                or q in str(r.get("business_purpose", "")).lower()
            ]
        total = len(rows)
        page = [_clean_kyc(r) for r in rows[args.offset:args.offset + args.limit]]
        return {"total": total, "limit": args.limit, "offset": args.offset, "items": page}

    yield FunctionInfo.from_fn(_run, description="Search KYC entities.",
                                input_schema=ListEntitiesInput)


class EntityIdInput(BaseModel):
    entity_id: str


class GetEntityConfig(FunctionBaseConfig, name="get_entity"):
    pass


@register_function(config_type=GetEntityConfig)
async def get_entity(config: GetEntityConfig, builder: Builder):
    async def _run(args: EntityIdInput) -> dict:
        dp = _dp()
        row = dp.kyc().get(args.entity_id)
        if row is None:
            return {"error": f"entity not found: {args.entity_id}"}
        tx_df = dp.transactions()
        sub = tx_df[tx_df["entity_id"] == args.entity_id]
        n_tx = int(len(sub))
        related_alerts = [a for a in dp.manifest() if a["entity_id"] == args.entity_id]
        return {
            "kyc": _clean_kyc(row),
            "n_tx_total": n_tx,
            "n_unique_counterparties": int(sub["counterparty"].nunique()) if n_tx else 0,
            "channel_mix": Counter(sub["channel"].tolist()) if n_tx else {},
            "n_related_alerts": len(related_alerts),
            "related_alerts": related_alerts,
        }

    yield FunctionInfo.from_fn(_run, description="Full entity 360 profile.",
                                input_schema=EntityIdInput)


class GetEntityTxInput(BaseModel):
    entity_id: str
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    limit: int = Field(default=200, ge=1, le=2000)
    offset: int = Field(default=0, ge=0)


class GetEntityTxConfig(FunctionBaseConfig, name="get_entity_tx"):
    pass


@register_function(config_type=GetEntityTxConfig)
async def get_entity_tx(config: GetEntityTxConfig, builder: Builder):
    async def _run(args: GetEntityTxInput) -> dict:
        dp = _dp()
        df = dp.transactions()
        sub = df[df["entity_id"] == args.entity_id]
        if args.window_start:
            sub = sub[sub["_date_parsed"] >= pd.Timestamp(args.window_start)]
        if args.window_end:
            sub = sub[sub["_date_parsed"] <= pd.Timestamp(args.window_end)]
        total = int(len(sub))
        page = sub.iloc[args.offset:args.offset + args.limit].to_dict("records")
        return {"total": total, "limit": args.limit, "offset": args.offset,
                "items": _clean_tx(page)}

    yield FunctionInfo.from_fn(_run, description="Paginated tx history for one entity.",
                                input_schema=GetEntityTxInput)


class GetEntityBehavioralConfig(FunctionBaseConfig, name="get_entity_behavioral"):
    pass


@register_function(config_type=GetEntityBehavioralConfig)
async def get_entity_behavioral(config: GetEntityBehavioralConfig, builder: Builder):
    async def _run(args: EntityIdInput) -> dict:
        dp = _dp()
        kyc = dp.kyc().get(args.entity_id)
        if kyc is None:
            return {"error": f"entity not found: {args.entity_id}"}
        df = dp.transactions()
        sub = df[df["entity_id"] == args.entity_id]
        txs = _clean_tx(sub.to_dict("records"))
        metrics = compute_behavioral_metrics(txs, _clean_kyc(kyc))
        return {
            "entity_id": args.entity_id,
            "n_transactions": int(len(sub)),
            "metrics": metrics.model_dump(),
        }

    yield FunctionInfo.from_fn(_run, description="Deterministic behavioral metrics for one entity.",
                                input_schema=EntityIdInput)


class GetEntityRiskConfig(FunctionBaseConfig, name="get_entity_risk"):
    pass


@register_function(config_type=GetEntityRiskConfig)
async def get_entity_risk(config: GetEntityRiskConfig, builder: Builder):
    async def _run(args: EntityIdInput) -> dict:
        dp = _dp()
        kyc = dp.kyc().get(args.entity_id)
        if kyc is None:
            return {"error": f"entity not found: {args.entity_id}"}
        df = dp.transactions()
        sub = df[df["entity_id"] == args.entity_id]
        txs = _clean_tx(sub.to_dict("records"))
        metrics = compute_behavioral_metrics(txs, _clean_kyc(kyc))
        score = risk_score(
            kyc_risk_rating=str(kyc.get("risk_rating", "medium")),
            country_risk_max=metrics.country_risk_max,
            vs_declared_volume_ratio=metrics.vs_declared_volume_ratio,
            amount_z_score_max=metrics.amount_z_score_max,
        )
        return {
            "entity_id": args.entity_id,
            "score": score,
            "components": {
                "kyc_risk_rating": kyc.get("risk_rating"),
                "country_risk_max": metrics.country_risk_max,
                "vs_declared_volume_ratio": metrics.vs_declared_volume_ratio,
                "amount_z_score_max": metrics.amount_z_score_max,
            },
        }

    yield FunctionInfo.from_fn(_run, description="Derived 0-100 risk score for one entity.",
                                input_schema=EntityIdInput)


class GetEntityNetworkInput(BaseModel):
    entity_id: str
    depth: int = Field(default=2, ge=1, le=3)
    window_start: Optional[str] = None
    window_end: Optional[str] = None


class GetEntityNetworkConfig(FunctionBaseConfig, name="get_entity_network"):
    pass


@register_function(config_type=GetEntityNetworkConfig)
async def get_entity_network(config: GetEntityNetworkConfig, builder: Builder):
    async def _run(args: GetEntityNetworkInput) -> dict:
        dp = _dp()
        return build_entity_network(
            dp.transactions(),
            args.entity_id,
            depth=args.depth,
            window_start=args.window_start,
            window_end=args.window_end,
        )

    yield FunctionInfo.from_fn(_run, description="N-hop counterparty graph for one entity.",
                                input_schema=GetEntityNetworkInput)


class GetEntityTimelineConfig(FunctionBaseConfig, name="get_entity_timeline"):
    pass


@register_function(config_type=GetEntityTimelineConfig)
async def get_entity_timeline(config: GetEntityTimelineConfig, builder: Builder):
    async def _run(args: EntityIdInput) -> dict:
        dp = _dp()
        df = dp.transactions()
        sub = df[df["entity_id"] == args.entity_id]
        if sub.empty:
            return {"entity_id": args.entity_id, "daily": []}
        by_day = sub.groupby(sub["_date_parsed"].dt.date).agg(
            n_tx=("amount", "count"),
            total_usd=("amount", "sum"),
        ).reset_index().rename(columns={"_date_parsed": "date"})
        rows = [
            {"date": str(r["date"]), "n_tx": int(r["n_tx"]),
             "total_usd": round(float(r["total_usd"]), 2)}
            for _, r in by_day.iterrows()
        ]
        return {"entity_id": args.entity_id, "daily": rows}

    yield FunctionInfo.from_fn(_run, description="Daily tx volume timeline for one entity.",
                                input_schema=EntityIdInput)
