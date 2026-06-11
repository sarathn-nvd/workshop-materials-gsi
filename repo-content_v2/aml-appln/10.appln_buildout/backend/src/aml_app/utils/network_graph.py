"""NetworkX-based counterparty graph helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import networkx as nx
import pandas as pd


def build_entity_network(
    tx_df: pd.DataFrame,
    entity_id: str,
    *,
    depth: int = 2,
    window_start: str | None = None,
    window_end: str | None = None,
) -> dict:
    """Build an N-hop counterparty graph centered on `entity_id`.

    Returns {nodes:[{id,type,risk?,n_tx?,total_usd?}], edges:[{source,target,
    n_tx,total_usd}]}.
    """
    df = tx_df
    if window_start:
        df = df[df["_date_parsed"] >= pd.Timestamp(window_start)]
    if window_end:
        df = df[df["_date_parsed"] <= pd.Timestamp(window_end)]

    g = nx.DiGraph()
    seen: set[str] = {entity_id}
    frontier: set[str] = {entity_id}

    for _ in range(depth):
        new_frontier: set[str] = set()
        for src in frontier:
            sub = df[df["entity_id"] == src]
            cp_counts = sub.groupby("counterparty").agg(
                n_tx=("amount", "count"),
                total_usd=("amount", "sum"),
            )
            for cp, row in cp_counts.iterrows():
                g.add_edge(src, cp,
                           n_tx=int(row["n_tx"]),
                           total_usd=float(row["total_usd"]))
                if cp not in seen:
                    seen.add(cp)
                    new_frontier.add(cp)
        frontier = new_frontier
        if not frontier:
            break

    nodes = []
    for n in g.nodes():
        out_deg = g.out_degree(n)
        in_deg = g.in_degree(n)
        nodes.append({
            "id": n,
            "type": "entity" if n == entity_id else "counterparty",
            "in_degree": int(in_deg),
            "out_degree": int(out_deg),
        })

    edges = [
        {
            "source": u, "target": v,
            "n_tx": d.get("n_tx", 1),
            "total_usd": round(d.get("total_usd", 0.0), 2),
        }
        for u, v, d in g.edges(data=True)
    ]

    return {
        "center": entity_id,
        "depth": depth,
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "nodes": nodes,
        "edges": edges,
    }


def detect_cycles(tx_df: pd.DataFrame, max_cycles: int = 25) -> list[dict]:
    """Find simple cycles in the global counterparty graph."""
    g = nx.DiGraph()
    for src, cp in zip(tx_df["entity_id"], tx_df["counterparty"]):
        g.add_edge(src, cp)
    out = []
    try:
        for i, cyc in enumerate(nx.simple_cycles(g)):
            if i >= max_cycles:
                break
            if 2 <= len(cyc) <= 6:
                out.append({"length": len(cyc), "nodes": cyc})
    except Exception:
        pass
    return out


def global_summary(tx_df: pd.DataFrame, top_k: int = 10) -> dict:
    """Global counterparty graph summary."""
    g = nx.DiGraph()
    for src, cp in zip(tx_df["entity_id"], tx_df["counterparty"]):
        g.add_edge(src, cp)
    if g.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0, "top_hubs": []}
    pr = nx.pagerank(g, alpha=0.85, max_iter=50)
    top_hubs = sorted(pr.items(), key=lambda kv: -kv[1])[:top_k]
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "top_hubs": [{"id": k, "pagerank": round(v, 5)} for k, v in top_hubs],
    }
