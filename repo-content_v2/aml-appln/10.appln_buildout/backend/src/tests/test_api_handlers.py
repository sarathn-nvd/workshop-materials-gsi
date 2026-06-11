"""Smoke tests for the API handler functions.

Exercises each handler by calling its inner `_run` against the real
data plane, without going through NAT's builder. Confirms the route
contracts (shapes + filtering + no-leakage of internal columns) without
needing a live LLM."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Helper: peel out a registered handler's inner _run for direct invocation
# ---------------------------------------------------------------------------
async def _invoke_handler(builder_fn, config):
    """Enter NAT's @asynccontextmanager wrapper around the async-generator
    factory and pull out the inner callable wrapped by FunctionInfo."""
    cm = builder_fn(config, builder=None)
    fi = await cm.__aenter__()
    inner = (getattr(fi, "single_fn", None)
             or getattr(fi, "fn", None)
             or getattr(fi, "_fn", None))
    if inner is None:
        raise AssertionError(
            f"Could not extract inner function from FunctionInfo "
            f"(attributes: {dir(fi)})"
        )
    return inner


@pytest.fixture
def configs():
    from aml_app.api.alerts import (
        AlertsStatsConfig, GetAlertConfig, ListAlertsConfig,
        PostDispositionConfig,
    )
    from aml_app.api.analytics import (
        AnalyticsAuxUsageConfig, AnalyticsChannelMixConfig,
        AnalyticsOverviewConfig, AnalyticsRiskHeatmapConfig,
        AnalyticsTimelineConfig, AnalyticsTopCpConfig,
        AnalyticsTypologyConfig,
    )
    from aml_app.api.entities import (
        GetEntityBehavioralConfig, GetEntityConfig, GetEntityNetworkConfig,
        GetEntityRiskConfig, GetEntityTimelineConfig, GetEntityTxConfig,
        ListEntitiesConfig,
    )
    from aml_app.api.misc import (
        DemoSeedTracesConfig, GetSopBodyConfig, HealthConfig,
        ListPolicySourcesConfig, ListSopsConfig, ScreenNameConfig,
        SearchPolicyConfig, SystemComponentsConfig, SystemConfigConfig,
    )
    from aml_app.api.network import (
        GetGlobalNetworkConfig, GetNetworkPatternsConfig,
    )
    return SimpleNamespace(
        alerts_stats=AlertsStatsConfig(),
        list_alerts=ListAlertsConfig(),
        get_alert=GetAlertConfig(),
        post_disposition=PostDispositionConfig(),
        list_entities=ListEntitiesConfig(),
        get_entity=GetEntityConfig(),
        get_entity_tx=GetEntityTxConfig(),
        get_entity_behavioral=GetEntityBehavioralConfig(),
        get_entity_risk=GetEntityRiskConfig(),
        get_entity_network=GetEntityNetworkConfig(),
        get_entity_timeline=GetEntityTimelineConfig(),
        get_global_network=GetGlobalNetworkConfig(),
        get_network_patterns=GetNetworkPatternsConfig(),
        analytics_overview=AnalyticsOverviewConfig(),
        analytics_typology=AnalyticsTypologyConfig(),
        analytics_risk_heatmap=AnalyticsRiskHeatmapConfig(),
        analytics_timeline=AnalyticsTimelineConfig(),
        analytics_channel_mix=AnalyticsChannelMixConfig(),
        analytics_top_cp=AnalyticsTopCpConfig(),
        analytics_aux_usage=AnalyticsAuxUsageConfig(),
        list_policy_sources=ListPolicySourcesConfig(),
        list_sops=ListSopsConfig(),
        get_sop_body=GetSopBodyConfig(),
        search_policy=SearchPolicyConfig(),
        screen_name=ScreenNameConfig(),
        health=HealthConfig(),
        system_config=SystemConfigConfig(),
        system_components=SystemComponentsConfig(),
        demo_seed_traces=DemoSeedTracesConfig(),
    )


# ---------------------------------------------------------------------------
# Helpers for picking real ids out of the data plane
# ---------------------------------------------------------------------------
def _first_alert(dp):
    return dp.manifest()[0]


def _first_entity(dp):
    return next(iter(dp.kyc()))


# ---------------------------------------------------------------------------
async def _call(fn, payload_cls, **kwargs):
    """Invoke the inner handler with a typed input model."""
    args = payload_cls(**kwargs)
    return await fn(args)


# ---------------------------------------------------------------------------
async def test_health(configs):
    from aml_app.api.misc import Empty, health
    fn = await _invoke_handler(health, configs.health)
    out = await fn(Empty())
    assert out["ok"] is True
    assert out["n_transactions"] > 0
    assert out["n_entities"] > 0


async def test_system_config(configs):
    from aml_app.api.misc import Empty, system_config
    fn = await _invoke_handler(system_config, configs.system_config)
    out = await fn(Empty())
    assert "data_dir" in out
    assert out["sop_count"] >= 8


async def test_alerts_stats(configs, dp):
    from aml_app.api.alerts import AlertsStatsInput, alerts_stats
    fn = await _invoke_handler(alerts_stats, configs.alerts_stats)
    out = await fn(AlertsStatsInput())
    assert out["total"] == len(dp.manifest())
    assert "by_status" in out


async def test_list_alerts_pagination(configs):
    from aml_app.api.alerts import ListAlertsInput, list_alerts
    fn = await _invoke_handler(list_alerts, configs.list_alerts)
    out = await fn(ListAlertsInput(limit=5, offset=0))
    assert len(out["items"]) == 5
    assert out["total"] >= 5


async def test_get_alert_unknown(configs):
    from aml_app.api.alerts import GetAlertInput, get_alert
    fn = await _invoke_handler(get_alert, configs.get_alert)
    out = await fn(GetAlertInput(alert_id="NOPE_DOES_NOT_EXIST"))
    assert "error" in out


async def test_get_alert_real(configs, dp):
    from aml_app.api.alerts import GetAlertInput, get_alert
    alert = _first_alert(dp)
    fn = await _invoke_handler(get_alert, configs.get_alert)
    out = await fn(GetAlertInput(alert_id=alert["alert_id"]))
    assert out["alert"]["case_id"] == alert["case_id"]
    assert out["status"] in {"open", "in_progress", "closed"}


async def test_post_disposition(configs, dp):
    from aml_app.api.alerts import PostDispositionInput, post_disposition
    alert = _first_alert(dp)
    fn = await _invoke_handler(post_disposition, configs.post_disposition)
    out = await fn(PostDispositionInput(
        alert_id=alert["alert_id"], verdict="dismiss", note="unit test"))
    assert out.get("ok") is True
    path = Path(out["path"])
    assert path.exists()
    obj = json.loads(path.read_text())
    assert obj["verdict"] == "dismiss"


async def test_list_entities(configs):
    from aml_app.api.entities import ListEntitiesInput, list_entities
    fn = await _invoke_handler(list_entities, configs.list_entities)
    out = await fn(ListEntitiesInput(limit=10))
    assert out["total"] > 0
    # Sidecars stripped
    for item in out["items"]:
        assert "source_pool" not in item
        assert "_archetype" not in item


async def test_get_entity_and_tx_and_behavioral(configs, dp):
    from aml_app.api.entities import (
        EntityIdInput, GetEntityTxInput,
        get_entity, get_entity_tx, get_entity_behavioral, get_entity_risk,
    )
    df = dp.transactions()
    eid = df["entity_id"].iloc[0]
    fn_e = await _invoke_handler(get_entity, configs.get_entity)
    out = await fn_e(EntityIdInput(entity_id=eid))
    assert "kyc" in out and out["kyc"]["entity_id"] == eid

    fn_tx = await _invoke_handler(get_entity_tx, configs.get_entity_tx)
    out_tx = await fn_tx(GetEntityTxInput(entity_id=eid, limit=5))
    assert out_tx["total"] >= 1
    for r in out_tx["items"]:
        assert "source_pool" not in r
        assert "typology_tag" not in r

    fn_b = await _invoke_handler(get_entity_behavioral, configs.get_entity_behavioral)
    out_b = await fn_b(EntityIdInput(entity_id=eid))
    assert "metrics" in out_b
    assert out_b["metrics"]["tx_count"] >= 1

    fn_r = await _invoke_handler(get_entity_risk, configs.get_entity_risk)
    out_r = await fn_r(EntityIdInput(entity_id=eid))
    assert 0 <= out_r["score"] <= 100


async def test_get_entity_network(configs, dp):
    from aml_app.api.entities import GetEntityNetworkInput, get_entity_network
    eid = dp.transactions()["entity_id"].iloc[0]
    fn = await _invoke_handler(get_entity_network, configs.get_entity_network)
    out = await fn(GetEntityNetworkInput(entity_id=eid, depth=1))
    assert out["n_nodes"] >= 1


async def test_search_policy(configs):
    from aml_app.api.misc import SearchPolicyInput, search_policy
    fn = await _invoke_handler(search_policy, configs.search_policy)
    out = await fn(SearchPolicyInput(typology="structuring", k=3))
    assert "items" in out


async def test_list_policy_sources(configs):
    from aml_app.api.misc import Empty, list_policy_sources
    fn = await _invoke_handler(list_policy_sources, configs.list_policy_sources)
    out = await fn(Empty())
    assert out["n_chunks"] > 0
    assert "FinCEN" in out["sources"]


async def test_list_sops_and_body(configs):
    from aml_app.api.misc import Empty, GetSopBodyInput, list_sops, get_sop_body
    fn_list = await _invoke_handler(list_sops, configs.list_sops)
    out_list = await fn_list(Empty())
    assert len(out_list["sop_ids"]) >= 8
    sop_id = out_list["sop_ids"][0]
    fn_body = await _invoke_handler(get_sop_body, configs.get_sop_body)
    out_body = await fn_body(GetSopBodyInput(sop_id=sop_id))
    assert "body_markdown" in out_body and out_body["body_markdown"].startswith("## ")


async def test_screen_name(configs):
    from aml_app.tools.data_tools import ScreenSanctionsInput
    from aml_app.api.misc import screen_name
    fn = await _invoke_handler(screen_name, configs.screen_name)
    out = await fn(ScreenSanctionsInput(name="ZZZ unknown"))
    assert "hits" in out


async def test_analytics_overview(configs):
    from aml_app.api.misc import Empty
    from aml_app.api.analytics import analytics_overview
    fn = await _invoke_handler(analytics_overview, configs.analytics_overview)
    out = await fn(Empty())
    assert out["n_alerts_total"] > 0
    assert out["n_entities"] > 0


async def test_analytics_typology(configs):
    from aml_app.api.misc import Empty
    from aml_app.api.analytics import analytics_typology
    fn = await _invoke_handler(analytics_typology, configs.analytics_typology)
    out = await fn(Empty())
    assert "seeded" in out


async def test_analytics_top_cp(configs):
    from aml_app.api.misc import Empty
    from aml_app.api.analytics import analytics_top_cp
    fn = await _invoke_handler(analytics_top_cp, configs.analytics_top_cp)
    out = await fn(Empty())
    assert len(out["top_by_volume"]) > 0


async def test_get_global_network(configs):
    from aml_app.api.misc import Empty
    from aml_app.api.network import get_global_network
    fn = await _invoke_handler(get_global_network, configs.get_global_network)
    out = await fn(Empty())
    assert out["n_nodes"] > 0
    assert len(out["top_hubs"]) > 0


async def test_demo_seed_traces(configs):
    from aml_app.api.misc import Empty, demo_seed_traces
    fn = await _invoke_handler(demo_seed_traces, configs.demo_seed_traces)
    out = await fn(Empty())
    assert out["n_seeds"] > 0
    assert out["n_written"] >= 0
