"use client";

import { useMemo } from "react";
import type { EntityNetwork } from "@/lib/api";

/**
 * Self-contained SVG force-style layout (no Cytoscape dep).
 * Nodes are placed on a fan/ring around the center entity.
 */
export function EntityNetworkGraph({
  data,
  height = 360,
}: {
  data: EntityNetwork;
  height?: number;
}) {
  const layout = useMemo(() => layoutGraph(data), [data]);
  if (data.n_nodes === 0) {
    return (
      <div className="p-6 text-center text-xs text-muted">
        No counterparties in window.
      </div>
    );
  }
  const padding = 24;
  return (
    <svg
      viewBox={`0 0 ${layout.w} ${height}`}
      className="w-full"
      style={{ height }}
    >
      {/* edges */}
      {layout.edges.map((e, i) => (
        <line
          key={i}
          x1={e.x1}
          y1={e.y1}
          x2={e.x2}
          y2={e.y2}
          stroke={e.heavy ? "#76b900" : "currentColor"}
          strokeOpacity={e.heavy ? 0.6 : 0.18}
          strokeWidth={e.heavy ? 1.6 : 1}
        />
      ))}
      {/* nodes */}
      {layout.nodes.map((n) => (
        <g key={n.id} transform={`translate(${n.x},${n.y})`}>
          <circle
            r={n.center ? 11 : 6}
            fill={n.center ? "#76b900" : "rgba(118,185,0,0.15)"}
            stroke={n.center ? "#76b900" : "rgba(118,185,0,0.6)"}
            strokeWidth={n.center ? 0 : 1.2}
          />
          <text
            x={n.center ? 16 : 10}
            y={4}
            fontSize={n.center ? 12 : 10}
            fill="currentColor"
            style={{ pointerEvents: "none" }}
            opacity={0.85}
          >
            {trunc(n.id, n.center ? 32 : 22)}
          </text>
        </g>
      ))}
      {/* padding sentinel */}
      <rect width={layout.w} height={height} fill="none" />
      <text x={padding} y={height - 8} fontSize={10} fill="currentColor" opacity={0.4}>
        {data.n_nodes} nodes · {data.n_edges} edges
      </text>
    </svg>
  );
}

function trunc(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function layoutGraph(data: EntityNetwork) {
  const w = 760;
  const cx = w / 2;
  const cy = 180;
  const radius = 130;
  const cps = data.nodes.filter((n) => n.id !== data.center);
  const placed = new Map<string, { x: number; y: number }>();
  placed.set(data.center, { x: cx, y: cy });
  const angleStep = (2 * Math.PI) / Math.max(1, cps.length);
  cps.forEach((n, i) => {
    placed.set(n.id, {
      x: cx + radius * Math.cos(i * angleStep - Math.PI / 2),
      y: cy + radius * Math.sin(i * angleStep - Math.PI / 2),
    });
  });
  const usds = data.edges.map((e) => e.total_usd);
  const heavyCut = usds.length
    ? usds.slice().sort((a, b) => b - a)[Math.min(4, usds.length - 1)]
    : 0;
  return {
    w,
    nodes: data.nodes.map((n) => ({
      id: n.id,
      center: n.id === data.center,
      ...(placed.get(n.id) || { x: cx, y: cy }),
    })),
    edges: data.edges.map((e) => {
      const s = placed.get(e.source) || { x: cx, y: cy };
      const t = placed.get(e.target) || { x: cx, y: cy };
      return {
        x1: s.x,
        y1: s.y,
        x2: t.x,
        y2: t.y,
        heavy: e.total_usd >= heavyCut,
      };
    }),
  };
}
