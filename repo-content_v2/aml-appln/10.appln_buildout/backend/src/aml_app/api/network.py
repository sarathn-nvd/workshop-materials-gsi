"""Network / graph analysis routes."""

import os
from collections.abc import AsyncGenerator

import networkx as nx
from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from aml_app.utils.data_loader import get_data_plane
from aml_app.utils.network_graph import detect_cycles, global_summary

_ENV_DATA_DIR_KEY = "NAT_AML_DATA_DIR"


def _dp():
    return get_data_plane(os.environ.get(_ENV_DATA_DIR_KEY, "./data"))


class Empty(BaseModel):
    pass


class GetGlobalNetworkConfig(FunctionBaseConfig, name="get_global_network"):
    pass


@register_function(config_type=GetGlobalNetworkConfig)
async def get_global_network(config: GetGlobalNetworkConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        return global_summary(dp.transactions(), top_k=15)

    yield FunctionInfo.from_fn(_run, description="Global counterparty-graph summary.",
                                input_schema=Empty)


class GetNetworkPatternsConfig(FunctionBaseConfig, name="get_network_patterns"):
    pass


@register_function(config_type=GetNetworkPatternsConfig)
async def get_network_patterns(config: GetNetworkPatternsConfig, builder: Builder):
    async def _run(args: Empty) -> dict:
        dp = _dp()
        cycles = detect_cycles(dp.transactions(), max_cycles=20)
        return {"n_cycles": len(cycles), "cycles": cycles}

    yield FunctionInfo.from_fn(_run, description="Pre-computed loop / cycle detections.",
                                input_schema=Empty)


class PathInput(BaseModel):
    source: str
    target: str


class GetNetworkPathConfig(FunctionBaseConfig, name="get_network_path"):
    pass


@register_function(config_type=GetNetworkPathConfig)
async def get_network_path(config: GetNetworkPathConfig, builder: Builder):
    async def _run(args: PathInput) -> dict:
        dp = _dp()
        df = dp.transactions()
        g = nx.DiGraph()
        for src, cp in zip(df["entity_id"], df["counterparty"]):
            g.add_edge(src, cp)
        try:
            path = nx.shortest_path(g, source=args.source, target=args.target)
            return {"found": True, "length": len(path), "path": path}
        except nx.NetworkXNoPath:
            return {"found": False, "reason": "no path"}
        except nx.NodeNotFound as e:
            return {"found": False, "reason": str(e)}

    yield FunctionInfo.from_fn(_run, description="Shortest path between two entities.",
                                input_schema=PathInput)
