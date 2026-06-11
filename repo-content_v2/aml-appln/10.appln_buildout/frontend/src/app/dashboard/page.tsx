"use client";

import useSWR from "swr";
import {
  analyticsOverview,
  analyticsTypology,
  analyticsRiskHeatmap,
  analyticsTimeline,
  analyticsChannelMix,
  analyticsTopCp,
  analyticsAuxUsage,
  analyticsAgentPerf,
} from "@/lib/api";
import { Kpi, Panel, PageHeader, Spinner } from "@/components/ui";
import {
  AlertOctagon,
  Banknote,
  Clock4,
  FileSearch,
  Users2,
} from "lucide-react";
import { fmtNum, fmtMs, fmtPct, titleCase, typologyHue, palette } from "@/lib/format";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
  Legend,
} from "recharts";

export default function DashboardPage() {
  const { data: overview } = useSWR("overview", analyticsOverview);
  const { data: typology } = useSWR("typology", analyticsTypology);
  const { data: heat } = useSWR("heat", analyticsRiskHeatmap);
  const { data: tline } = useSWR("tline", analyticsTimeline);
  const { data: chmix } = useSWR("chmix", analyticsChannelMix);
  const { data: topcp } = useSWR("topcp", analyticsTopCp);
  const { data: aux } = useSWR("aux", analyticsAuxUsage);
  const { data: perf } = useSWR("perf", analyticsAgentPerf);

  return (
    <div className="pb-10">
      <PageHeader
        title="Analytics Dashboard"
        subtitle="Top-of-funnel view across alerts, entities, SAR pipeline, and agent performance."
      />

      {/* KPI strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 px-6">
        <Kpi
          label="Total Alerts"
          icon={<AlertOctagon size={16} className="text-nv-green" />}
          value={overview ? fmtNum(overview.n_alerts_total) : "—"}
          hint={
            overview
              ? `${overview.n_alerts_open} open · ${overview.n_alerts_in_progress} in-progress · ${overview.n_alerts_closed} closed`
              : ""
          }
          highlight
        />
        <Kpi
          label="Entities under monitoring"
          icon={<Users2 size={16} />}
          value={overview ? fmtNum(overview.n_entities) : "—"}
          hint="Distinct KYC profiles in the data plane"
        />
        <Kpi
          label="Transactions ingested"
          icon={<Banknote size={16} />}
          value={overview ? fmtNum(overview.n_transactions) : "—"}
          hint={`SARs drafted: ${overview ? fmtNum(overview.n_sars_drafted) : "—"}`}
        />
        <Kpi
          label="Avg. case latency"
          icon={<Clock4 size={16} />}
          value={overview ? fmtMs(overview.avg_case_latency_ms) : "—"}
          hint="reasoning → SAR pipeline wall clock"
        />
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 px-6 mt-6">
        <Panel
          title="Typology distribution"
          subtitle={(() => {
            if (!typology) return "Alerts grouped by inferred typology";
            const fromTracesTotal = Object.values(typology.from_traces).reduce((s, v) => s + v, 0);
            return fromTracesTotal >= 5
              ? "Inferred from persisted case traces"
              : "Ground-truth manifest (run investigations to switch to trace-based view)";
          })()}
          className="lg:col-span-1"
        >
          <div className="h-72 p-3">
            {typology ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={typologyDataPicked(typology)}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={48}
                    outerRadius={84}
                    paddingAngle={2}
                  >
                    {typologyDataPicked(typology).map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<DarkTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <Spinner />
            )}
          </div>
          <div className="px-4 pb-3 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            {typology
              ? typologyDataPicked(typology).map((d) => (
                  <div key={d.name} className="flex items-center gap-2">
                    <span
                      className="h-2 w-2 rounded-sm shrink-0"
                      style={{ background: d.color }}
                    />
                    <span className="text-muted truncate">{titleCase(d.name)}</span>
                    <span className="ml-auto font-mono">{d.value}</span>
                  </div>
                ))
              : null}
          </div>
        </Panel>

        <Panel
          title="Transaction volume"
          subtitle="Daily ingested transaction count"
          className="lg:col-span-2"
        >
          <div className="h-72 p-3">
            {tline ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tline.daily_tx_count.map((d) => ({ ...d }))}>
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    minTickGap={32}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    width={36}
                  />
                  <Tooltip content={<DarkTooltip />} />
                  <Line
                    type="monotone"
                    dataKey="n_tx"
                    stroke="#76b900"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Spinner />
            )}
          </div>
        </Panel>
      </div>

      {/* Charts row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 px-6 mt-4">
        <Panel
          title="Alerts by risk rating"
          subtitle="Distribution of monitored entities by KYC risk band"
        >
          <div className="h-64 p-3">
            {heat ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={Object.entries(heat.alerts_by_risk_rating).map(
                    ([k, v]) => ({ name: titleCase(k), value: v }),
                  )}
                >
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11, fill: "currentColor" }}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    width={36}
                  />
                  <Tooltip content={<DarkTooltip />} />
                  <Bar dataKey="value" fill="#76b900" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Spinner />
            )}
          </div>
        </Panel>

        <Panel
          title="Channel mix per typology"
          subtitle="Where the money moves, by typology"
        >
          <div className="h-64 p-3">
            {chmix ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={channelData(chmix.by_typology)} stackOffset="expand">
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                  <XAxis
                    dataKey="typology"
                    tick={{ fontSize: 10, fill: "currentColor" }}
                    interval={0}
                    angle={-30}
                    textAnchor="end"
                    height={70}
                    tickFormatter={(v: string) =>
                      v.length > 14 ? v.slice(0, 12) + "…" : v
                    }
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    width={36}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                  />
                  <Tooltip content={<DarkTooltip percent />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {["cash", "wire", "ach", "card", "cheque", "crypto"].map((ch, i) => (
                    <Bar
                      key={ch}
                      dataKey={ch}
                      stackId="a"
                      fill={palette[i]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Spinner />
            )}
          </div>
        </Panel>

        <Panel
          title="Top counterparties"
          subtitle="Highest aggregate USD flow across the corpus"
        >
          <div className="h-64 overflow-auto">
            {topcp ? (
              <table className="w-full text-xs">
                <thead className="text-muted">
                  <tr className="text-left">
                    <th className="py-2 px-3">Counterparty</th>
                    <th className="py-2 px-3 text-right">n_tx</th>
                    <th className="py-2 px-3 text-right">USD volume</th>
                  </tr>
                </thead>
                <tbody>
                  {topcp.top_by_volume.slice(0, 10).map((r) => (
                    <tr key={r.counterparty} className="t-row border-t">
                      <td className="py-1.5 px-3 truncate max-w-[180px]">
                        {r.counterparty}
                      </td>
                      <td className="py-1.5 px-3 text-right font-mono">
                        {r.n_tx.toLocaleString()}
                      </td>
                      <td className="py-1.5 px-3 text-right font-mono">
                        ${r.total_usd.toLocaleString(undefined, {
                          maximumFractionDigits: 0,
                        })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <Spinner />
            )}
          </div>
        </Panel>
      </div>

      {/* Aux gate + agent perf */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 px-6 mt-4">
        <Panel
          title="Aux-gate decisions"
          subtitle="How often each auxiliary finding made it into the SAR bundle"
        >
          <div className="h-64 p-3">
            {aux ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={auxData(aux)} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11, fill: "currentColor" }}
                  />
                  <YAxis
                    type="category"
                    dataKey="task"
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    width={80}
                  />
                  <Tooltip content={<DarkTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="used" stackId="a" fill="#76b900" />
                  <Bar dataKey="dropped" stackId="a" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Spinner />
            )}
          </div>
        </Panel>

        <Panel
          title="Per-typology performance"
          subtitle={
            perf
              ? `Recall × precision vs ground truth · scored over ${perf.n_traces} trace${perf.n_traces === 1 ? "" : "s"}`
              : "Recall × precision vs ground truth"
          }
          className="lg:col-span-2"
        >
          <div className="overflow-auto">
            {!perf ? (
              <Spinner />
            ) : perf.n_traces < 5 ? (
              <div className="p-6 text-center">
                <div className="text-sm">
                  Only <span className="font-mono">{perf.n_traces}</span> trace
                  {perf.n_traces === 1 ? "" : "s"} on disk — not enough to compute
                  per-typology recall / precision.
                </div>
                <div className="text-xs text-muted mt-1.5 leading-relaxed">
                  Open the <a href="/alerts" className="text-nv-green hover:underline">Alert Queue</a>{" "}
                  and click <strong>Run</strong> on a few alerts (or run{" "}
                  <code className="font-mono text-nv-green">scripts/run_batch.py</code>{" "}
                  to batch-investigate the manifest). This panel will populate
                  once at least a handful of traces have been written.
                </div>
              </div>
            ) : (
              <table className="w-full text-xs">
                <thead className="text-muted">
                  <tr className="text-left">
                    <th className="py-2 px-3">Typology</th>
                    <th className="py-2 px-3 text-right">TP</th>
                    <th className="py-2 px-3 text-right">FP</th>
                    <th className="py-2 px-3 text-right">FN</th>
                    <th className="py-2 px-3 text-right">TN</th>
                    <th className="py-2 px-3 text-right">Recall</th>
                    <th className="py-2 px-3 text-right">Precision</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(perf.per_typology)
                    .filter(([k]) => k !== "none")
                    .map(([k, v]) => (
                    <tr key={k} className="t-row border-t">
                      <td className="py-1.5 px-3">
                        <span
                          className="inline-block h-2 w-2 rounded-sm mr-2 align-middle"
                          style={{ background: typologyHue[k] ?? "#64748b" }}
                        />
                        {titleCase(k)}
                      </td>
                      <td className="py-1.5 px-3 text-right font-mono">{v.tp}</td>
                      <td className="py-1.5 px-3 text-right font-mono">{v.fp}</td>
                      <td className="py-1.5 px-3 text-right font-mono">{v.fn}</td>
                      <td className="py-1.5 px-3 text-right font-mono">{v.tn}</td>
                      <td className="py-1.5 px-3 text-right font-mono">
                        {v.recall != null ? fmtPct(v.recall) : "—"}
                      </td>
                      <td className="py-1.5 px-3 text-right font-mono">
                        {v.precision != null ? fmtPct(v.precision) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Panel>
      </div>

      <div className="px-6 mt-4 text-[11px] text-muted flex items-center gap-2">
        <FileSearch size={12} />
        Data sourced live from <code className="font-mono text-nv-green">/api/analytics/*</code> via NAT FastAPI front-end.
      </div>
    </div>
  );
}

function typologyData(obj: Record<string, number>) {
  return Object.entries(obj)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({
      name,
      value,
      color: typologyHue[name] ?? "#64748b",
    }))
    .sort((a, b) => b.value - a.value);
}

// Prefer trace-derived distribution once enough cases have been investigated;
// otherwise fall back to the ground-truth manifest so the donut is meaningful
// on a fresh install.
function typologyDataPicked(t: { seeded: Record<string, number>; from_traces: Record<string, number> }) {
  const fromTracesTotal = Object.values(t.from_traces).reduce((s, v) => s + v, 0);
  return typologyData(fromTracesTotal >= 5 ? t.from_traces : t.seeded);
}

function channelData(
  byTyp: Record<string, Record<string, number>>,
): Record<string, number | string>[] {
  // Filter out internal near-miss sidecars and the "none" bucket — those are
  // labelling artefacts (seeded ground-truth tags), not real money-laundering
  // typologies the analyst wants to see in a channel breakdown.
  return Object.entries(byTyp)
    .filter(([k]) => k !== "none" && !k.startsWith("near_miss"))
    .map(([typology, mix]) => {
      const total = Object.values(mix).reduce((s, v) => s + v, 0);
      const row: Record<string, number | string> = { typology: titleCase(typology) };
      ["cash", "wire", "ach", "card", "cheque", "crypto"].forEach((ch) => {
        row[ch] = total ? (mix[ch] ?? 0) / total : 0;
      });
      return row;
    });
}

function auxData(aux: {
  used: Record<string, number>;
  dropped: Record<string, number>;
}) {
  return ["behavioral", "numeric", "citation", "statutory"].map((task) => ({
    task: titleCase(task),
    used: aux.used[task] ?? 0,
    dropped: aux.dropped[task] ?? 0,
  }));
}

function DarkTooltip({
  active,
  payload,
  label,
  percent,
}: {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: string | number;
  percent?: boolean;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border divider surface px-2.5 py-1.5 text-xs shadow-lg">
      {label && <div className="font-medium mb-1">{label}</div>}
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span
            className="h-2 w-2 rounded-sm"
            style={{ background: p.color }}
          />
          <span className="text-muted">{p.name}</span>
          <span className="ml-auto font-mono">
            {percent
              ? `${(Number(p.value) * 100).toFixed(1)}%`
              : Number(p.value).toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  );
}
