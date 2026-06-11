"""End-to-end registration sanity: every NAT component the workflow YAML
expects is importable and discoverable, and every function name referenced
under `endpoints:` resolves to a registered function.
"""
from __future__ import annotations

from pathlib import Path

import yaml

import aml_app.register  # noqa: F401 \u2014 fires every @register_function

# Trigger NAT's entry-point discovery so built-in plugins (react_agent /
# tool_calling_agent / reasoning_agent from nvidia-nat-langchain, etc.)
# show up in the global registry.
from nat.runtime.loader import PluginTypes, discover_and_register_plugins
discover_and_register_plugins(PluginTypes.COMPONENT)

from nat.cli.type_registry import GlobalTypeRegistry


EXPECTED_FUNCTIONS = [
    "compute_hints",
    "aux_call",
    "aux_gate",
    "sar_judgment_caller",
    # API handlers
    "list_alerts", "get_alert", "post_disposition", "alerts_stats",
    "list_entities", "get_entity", "get_entity_tx",
    "get_entity_behavioral", "get_entity_risk",
    "get_entity_network", "get_entity_timeline",
    "get_global_network", "get_network_patterns", "get_network_path",
    "skill_behavioral", "skill_numeric", "skill_citation",
    "skill_statutory", "skill_sar",
    "search_policy", "list_policy_sources", "list_sops", "get_sop_body",
    "screen_name",
    "analytics_overview", "analytics_typology", "analytics_risk_heatmap",
    "analytics_timeline", "analytics_channel_mix", "analytics_top_cp",
    "analytics_aux_usage", "analytics_agent_perf", "analytics_profile",
    "demo_eval", "demo_eval_cases", "demo_eval_case", "demo_seed_traces",
    "get_trace", "health", "system_config", "system_components",
]


def _registered_function_names() -> set[str]:
    reg = GlobalTypeRegistry.get()
    return {
        info.discovery_metadata.component_name
        for info in reg.get_registered_functions()
    }


def _registered_function_group_names() -> set[str]:
    reg = GlobalTypeRegistry.get()
    groups = getattr(reg, "get_registered_function_groups", lambda: [])() or []
    return {info.discovery_metadata.component_name for info in groups}


def test_all_expected_functions_registered():
    names = _registered_function_names()
    missing = [n for n in EXPECTED_FUNCTIONS if n not in names]
    assert not missing, f"Missing from registry: {missing}"


def test_data_tools_function_group_registered():
    groups = _registered_function_group_names()
    assert "aml_data_tools" in groups


def test_workflow_yaml_function_names_resolve():
    """Every function_name referenced under endpoints: must be defined under
    functions: in the YAML (NAT will check this at build time, but we
    catch any typo here without spinning up the full builder)."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "workflow.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    declared = set(cfg.get("functions", {}).keys())
    endpoints = cfg.get("general", {}).get("front_end", {}).get("endpoints", [])
    referenced = set()
    for ep in endpoints:
        fn = ep.get("function_name")
        if fn:
            referenced.add(fn)
    missing = referenced - declared
    assert not missing, f"Endpoint function_names with no `functions:` entry: {missing}"


def test_workflow_yaml_function_types_registered():
    """Every `_type:` value under `functions:` must be a registered function
    OR a built-in NAT function (e.g. react_agent / tool_calling_agent)."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "workflow.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    declared_types = set()
    for f in cfg.get("functions", {}).values():
        if isinstance(f, dict) and "_type" in f:
            declared_types.add(f["_type"])
    available = _registered_function_names()
    missing = declared_types - available
    assert not missing, f"`functions[*]._type` values not in registry: {missing}"
