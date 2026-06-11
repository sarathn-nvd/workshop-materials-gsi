"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import {
  getEntity,
  getEntityTx,
  getEntityBehavioral,
  getEntityRisk,
  getEntityNetwork,
  getEntityTimeline,
} from "@/lib/api";
import { PageHeader, Panel, Kpi, Spinner } from "@/components/ui";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
} from "recharts";
import {
  ArrowLeftCircle,
  Building2,
  User2,
  GaugeCircle,
  Banknote,
  Network,
  Activity,
} from "lucide-react";
import clsx from "clsx";
import { riskColor, titleCase, palette, fmtUsd } from "@/lib/format";
import { EntityNetworkGraph } from "@/components/EntityNetworkGraph";

const TABS = [
  { key: "overview", label: "Overview", icon: GaugeCircle },
  { key: "tx", label: "Transactions", icon: Banknote },
  { key: "behavioral", label: "Behavioral", icon: Activity },
  { key: "network", label: "Network", icon: Network },
] as const;

export default function EntityDetailPage({
  params,
}: {
  params: { entityId: string };
}) {
  const { entityId } = params;
  const [tab, setTab] = useState<(typeof TABS)[number]["key"]>("overview");

  const { data: entity } = useSWR(["entity", entityId], () => getEntity(entityId));
  const { data: risk } = useSWR(["risk", entityId], () => getEntityRisk(entityId));
  const { data: timeline } = useSWR(["timeline", entityId], () =>
    getEntityTimeline(entityId),
  );
  const { data: behavioral } = useSWR(["beh", entityId], () =>
    getEntityBehavioral(entityId),
  );
  const { data: tx } = useSWR(["tx", entityId], () => getEntityTx(entityId, 200));
  const { data: network } = useSWR(["net", entityId, 2], () =>
    getEntityNetwork(entityId, 2),
  );

  return (
    <div className="pb-10">
      <PageHeader
        title={
          <span className="flex items-center gap-3">
            <span className="font-mono text-nv-green">{entityId}</span>
            {entity?.kyc.entity_type === "business" ? (
              <Building2 size={16} className="text-muted" />
            ) : (
              <User2 size={16} className="text-muted" />
            )}
            {entity && (
              <span className={`text-xs ${riskColor(entity.kyc.risk_rating)}`}>
                {titleCase(entity.kyc.risk_rating || "")}
              </span>
            )}
          </span>
        }
        subtitle={entity?.kyc.business_purpose}
        actions={
          <Link href="/entities" className="btn btn-outline">
            <ArrowLeftCircle size={14} /> All entities
          </Link>
        }
      />

      {!entity ? (
        <Spinner label="Loading profile…" />
      ) : (
        <>
          {/* KPI strip */}
          <div className="px-6 grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            <Kpi
              label="Transactions"
              value={entity.n_tx_total.toLocaleString()}
              hint={`${entity.n_unique_counterparties} unique counterparties`}
            />
            <Kpi
              label="Risk score"
              value={
                <span
                  className={clsx(
                    "font-mono",
                    risk && risk.score >= 70
                      ? "text-red-500"
                      : risk && risk.score >= 50
                        ? "text-amber-500"
                        : "text-emerald-500",
                  )}
                >
                  {risk?.score ?? "—"}
                </span>
              }
              hint="Weighted blend (KYC × tx × sanctions)"
              highlight={risk ? risk.score >= 70 : false}
            />
            <Kpi
              label="Expected monthly"
              value={
                entity.kyc.expected_monthly_volume
                  ? fmtUsd(entity.kyc.expected_monthly_volume, 0)
                  : "—"
              }
            />
            <Kpi
              label="Related alerts"
              value={entity.n_related_alerts}
              hint="Linked to this entity"
            />
          </div>

          {/* Tabs */}
          <div className="px-6 mb-3 flex items-center gap-1">
            {TABS.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={clsx(
                  "btn",
                  tab === key
                    ? "border-nv-green/40 bg-nv-green/10 text-nv-green"
                    : "btn-ghost",
                )}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>

          {tab === "overview" && (
            <div className="px-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
              <Panel title="KYC profile" className="lg:col-span-1">
                <div className="p-4 grid grid-cols-2 gap-x-2 gap-y-2 text-xs">
                  <Row label="Entity ID" value={entity.kyc.entity_id} mono />
                  <Row label="Type" value={titleCase(entity.kyc.entity_type || "")} />
                  <Row
                    label="Risk rating"
                    value={
                      <span className={riskColor(entity.kyc.risk_rating)}>
                        {titleCase(entity.kyc.risk_rating || "")}
                      </span>
                    }
                  />
                  <Row
                    label="Jurisdiction"
                    value={entity.kyc.incorporation_jurisdiction || "—"}
                    mono
                  />
                  <Row
                    label="Expected monthly"
                    value={
                      entity.kyc.expected_monthly_volume
                        ? fmtUsd(entity.kyc.expected_monthly_volume, 0)
                        : "—"
                    }
                    mono
                  />
                  <div className="col-span-2 mt-3">
                    <div className="text-muted text-[11px]">Business purpose</div>
                    <p className="mt-0.5 leading-relaxed">
                      {entity.kyc.business_purpose}
                    </p>
                  </div>
                </div>
              </Panel>

              <Panel title="Channel mix" className="lg:col-span-1">
                <div className="h-56 p-3">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={Object.entries(entity.channel_mix).map(([k, v], i) => ({
                        name: titleCase(k),
                        value: v,
                        fill: palette[i],
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: "currentColor" }} />
                      <YAxis tick={{ fontSize: 11, fill: "currentColor" }} width={28} />
                      <Tooltip
                        contentStyle={{
                          background: "rgb(var(--panel))",
                          border: "1px solid rgb(var(--line))",
                          borderRadius: 6,
                          fontSize: 11,
                        }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]} fill="#76b900" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>

              <Panel title="Daily volume" className="lg:col-span-1">
                <div className="h-56 p-3">
                  {timeline ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={timeline.daily}>
                        <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 10, fill: "currentColor" }}
                          minTickGap={32}
                        />
                        <YAxis
                          tick={{ fontSize: 10, fill: "currentColor" }}
                          width={28}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "rgb(var(--panel))",
                            border: "1px solid rgb(var(--line))",
                            borderRadius: 6,
                            fontSize: 11,
                          }}
                        />
                        <Line
                          dataKey="total_usd"
                          stroke="#76b900"
                          dot={false}
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <Spinner />
                  )}
                </div>
              </Panel>

              <Panel
                title="Related alerts"
                className="lg:col-span-3"
                subtitle={`${entity.n_related_alerts} alert${entity.n_related_alerts === 1 ? "" : "s"} linked to this entity`}
              >
                {entity.related_alerts.length === 0 ? (
                  <div className="p-4 text-xs text-muted">No related alerts.</div>
                ) : (
                  <table className="w-full text-sm">
                    <thead className="text-xs text-muted">
                      <tr className="text-left border-b divider">
                        <th className="py-2 px-4">Case</th>
                        <th className="py-2 px-4">Alert</th>
                        <th className="py-2 px-4">Trigger</th>
                        <th className="py-2 px-4">Window</th>
                      </tr>
                    </thead>
                    <tbody>
                      {entity.related_alerts.map((a) => (
                        <tr key={a.case_id} className="t-row border-b">
                          <td className="py-2 px-4">
                            <Link
                              href={`/cockpit/${a.case_id}`}
                              className="font-mono text-nv-green hover:underline"
                            >
                              {a.case_id}
                            </Link>
                          </td>
                          <td className="py-2 px-4 font-mono text-xs text-muted">
                            {a.alert_id}
                          </td>
                          <td className="py-2 px-4 text-sm">
                            {a.trigger_summary}
                          </td>
                          <td className="py-2 px-4 text-xs font-mono text-muted">
                            {a.investigation_window_start} →{" "}
                            {a.investigation_window_end}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </Panel>
            </div>
          )}

          {tab === "tx" && (
            <Panel
              className="mx-6"
              title={tx ? `${tx.total.toLocaleString()} transactions` : "Transactions"}
              subtitle="Latest 200, filterable on the backend with window_start/end"
            >
              {!tx ? (
                <Spinner />
              ) : (
                <div className="overflow-auto max-h-[480px]">
                  <table className="w-full text-xs">
                    <thead className="text-muted sticky top-0 surface">
                      <tr className="text-left border-b divider">
                        <th className="py-2 px-3">Date</th>
                        <th className="py-2 px-3">Counterparty</th>
                        <th className="py-2 px-3">Channel</th>
                        <th className="py-2 px-3 text-right">Amount</th>
                        <th className="py-2 px-3">Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tx.items.map((t, i) => (
                        <tr key={i} className="t-row border-b">
                          <td className="py-1.5 px-3 font-mono text-muted">
                            {t.date}
                          </td>
                          <td className="py-1.5 px-3 truncate max-w-[200px]">
                            {t.counterparty}
                          </td>
                          <td className="py-1.5 px-3">
                            <span className="chip chip-neutral">{t.channel}</span>
                          </td>
                          <td className="py-1.5 px-3 text-right font-mono">
                            {fmtUsd(t.amount, 2)}
                          </td>
                          <td className="py-1.5 px-3 text-muted truncate max-w-[260px]">
                            {t.notes}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Panel>
          )}

          {tab === "behavioral" && (
            <div className="px-6">
              <Panel
                title="Deterministic behavioral metrics"
                subtitle="Same schema the trained model's auxiliary_behavioral task produces"
              >
                {!behavioral ? (
                  <Spinner />
                ) : (
                  <div className="p-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                    <Metric label="tx_count" value={behavioral.metrics.tx_count} />
                    <Metric
                      label="tx_total_usd"
                      value={`$${behavioral.metrics.tx_total_usd.toLocaleString(undefined, {
                        maximumFractionDigits: 0,
                      })}`}
                    />
                    <Metric
                      label="velocity_24h_max"
                      value={behavioral.metrics.velocity_24h_max}
                    />
                    <Metric
                      label="velocity_24h_avg_30d"
                      value={behavioral.metrics.velocity_24h_avg_30d.toFixed(2)}
                    />
                    <Metric
                      label="unique_cps_7d"
                      value={behavioral.metrics.unique_counterparties_7d}
                    />
                    <Metric
                      label="amount_z_score_max"
                      value={behavioral.metrics.amount_z_score_max.toFixed(2)}
                    />
                    <Metric
                      label="country_risk_max"
                      value={behavioral.metrics.country_risk_max.toFixed(2)}
                    />
                    <Metric
                      label="vs_declared_volume_ratio"
                      value={behavioral.metrics.vs_declared_volume_ratio.toFixed(2)}
                      highlight={behavioral.metrics.vs_declared_volume_ratio > 2}
                    />
                    <Metric
                      label="loop_detected"
                      value={behavioral.metrics.loop_detected ? "yes" : "no"}
                      highlight={behavioral.metrics.loop_detected}
                    />
                    <div className="md:col-span-2">
                      <Metric label="channel_mix" value="">
                        <div className="flex gap-1 mt-1">
                          {Object.entries(behavioral.metrics.channel_mix).map(
                            ([k, v]) => (
                              <span key={k} className="chip chip-neutral">
                                {k}
                                <span className="text-muted">
                                  {" "}
                                  {(v * 100).toFixed(0)}%
                                </span>
                              </span>
                            ),
                          )}
                        </div>
                      </Metric>
                    </div>
                  </div>
                )}
              </Panel>
              {risk && (
                <Panel title="Risk-score components" className="mt-4">
                  <div className="p-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(risk.components).map(([k, v]) => (
                      <Metric key={k} label={k} value={String(v)} />
                    ))}
                  </div>
                </Panel>
              )}
            </div>
          )}

          {tab === "network" && (
            <Panel
              className="mx-6"
              title="Counterparty network"
              subtitle={
                network
                  ? `Depth 2 · ${network.n_nodes} nodes · ${network.n_edges} edges`
                  : ""
              }
            >
              {!network ? (
                <Spinner />
              ) : (
                <EntityNetworkGraph data={network} />
              )}
            </Panel>
          )}
        </>
      )}
    </div>
  );
}

function Row({
  label,
  value,
  mono,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <>
      <div className="text-muted">{label}</div>
      <div className={clsx("text-right", mono && "font-mono")}>{value}</div>
    </>
  );
}

function Metric({
  label,
  value,
  highlight,
  children,
}: {
  label: string;
  value: React.ReactNode;
  highlight?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div
      className={clsx(
        "surface-muted rounded-md p-3",
        highlight && "ring-1 ring-nv-green/40",
      )}
    >
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-0.5 font-mono text-sm">{value}</div>
      {children}
    </div>
  );
}
