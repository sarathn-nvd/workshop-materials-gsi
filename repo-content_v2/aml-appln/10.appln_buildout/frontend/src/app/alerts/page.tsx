"use client";

import useSWR, { mutate } from "swr";
import { listAlerts, alertStats, runInvestigation } from "@/lib/api";
import { PageHeader, Panel, Kpi, Spinner, EmptyState } from "@/components/ui";
import { statusChip, titleCase, typologyHue } from "@/lib/format";
import Link from "next/link";
import { ArrowRight, Filter, Play, Search } from "lucide-react";
import { useState } from "react";

export default function AlertsPage() {
  const [status, setStatus] = useState<string>("");
  const [q, setQ] = useState<string>("");
  const [page, setPage] = useState(0);
  const pageSize = 25;

  const key = ["alerts", status, q, page] as const;
  const { data } = useSWR(key, () =>
    listAlerts({ status: status || undefined, q: q || undefined, limit: pageSize, offset: page * pageSize }),
  );
  const { data: stats } = useSWR("alerts-stats", alertStats);

  return (
    <div className="pb-10">
      <PageHeader
        title="Alert Queue"
        subtitle="Triage incoming monitoring alerts; click any row to open the cockpit."
        actions={
          <Link href="/dashboard" className="btn btn-outline">
            <ArrowRight size={14} className="rotate-180" />
            Back to dashboard
          </Link>
        }
      />

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 px-6 mb-4">
        <Kpi
          label="Total"
          value={stats?.total ?? "—"}
          hint="Across all statuses"
        />
        <Kpi
          label="Open"
          value={stats?.by_status?.open ?? 0}
          hint="No trace yet"
        />
        <Kpi
          label="In progress"
          value={stats?.by_status?.in_progress ?? 0}
          hint="Trace persisted"
        />
        <Kpi
          label="Closed"
          value={stats?.by_status?.closed ?? 0}
          hint="Analyst dispositioned"
        />
      </div>

      <Panel
        title="Alerts"
        subtitle={
          data ? `${data.total.toLocaleString()} match${data.total === 1 ? "" : "es"}` : ""
        }
        actions={
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search
                size={14}
                className="absolute left-2 top-1/2 -translate-y-1/2 text-muted pointer-events-none"
              />
              <input
                value={q}
                onChange={(e) => {
                  setQ(e.target.value);
                  setPage(0);
                }}
                placeholder="Search trigger / entity / alert id"
                className="pl-7 pr-2 py-1.5 text-xs border divider rounded-md surface-muted w-72 focus:outline-none focus:border-nv-green/60"
              />
            </div>
            <select
              value={status}
              onChange={(e) => {
                setStatus(e.target.value);
                setPage(0);
              }}
              className="text-xs border divider rounded-md surface-muted py-1.5 px-2 focus:outline-none focus:border-nv-green/60"
            >
              <option value="">All statuses</option>
              <option value="open">Open</option>
              <option value="in_progress">In progress</option>
              <option value="closed">Closed</option>
            </select>
            <button className="btn btn-outline" title="Filter">
              <Filter size={14} />
            </button>
          </div>
        }
        className="mx-6"
      >
        {!data ? (
          <Spinner label="Loading alerts" />
        ) : data.items.length === 0 ? (
          <EmptyState title="No matching alerts" hint="Try a different filter or search term." />
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-muted">
                <tr className="text-left border-b divider">
                  <th className="py-2.5 px-4">Case</th>
                  <th className="py-2.5 px-4">Alert</th>
                  <th className="py-2.5 px-4">Entity</th>
                  <th className="py-2.5 px-4">Trigger</th>
                  <th className="py-2.5 px-4">Window</th>
                  <th className="py-2.5 px-4">Status</th>
                  <th className="py-2.5 px-4 text-right" />
                </tr>
              </thead>
              <tbody>
                {data.items.map((a) => (
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
                    <td className="py-2 px-4">
                      <Link
                        href={`/entities/${a.entity_id}`}
                        className="font-mono text-xs hover:text-nv-green"
                      >
                        {a.entity_id}
                      </Link>
                    </td>
                    <td className="py-2 px-4">
                      <div className="max-w-[360px] truncate" title={a.trigger_summary}>
                        {a.trigger_summary}
                      </div>
                    </td>
                    <td className="py-2 px-4 text-xs text-muted font-mono">
                      {a.investigation_window_start} → {a.investigation_window_end}
                    </td>
                    <td className="py-2 px-4">
                      <span className={statusChip(a.status || "open")}>
                        {titleCase(a.status || "open")}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-right">
                      <div className="inline-flex items-center gap-3">
                        {(a.status || "open") === "open" && (
                          <RunButton caseId={a.case_id} />
                        )}
                        <Link
                          href={`/cockpit/${a.case_id}`}
                          className="text-xs text-muted hover:text-nv-green inline-flex items-center gap-1"
                        >
                          Open
                          <ArrowRight size={12} />
                        </Link>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* pagination */}
        {data && data.total > pageSize && (
          <div className="flex items-center justify-between px-4 py-3 border-t divider text-xs text-muted">
            <span>
              Page {page + 1} of {Math.ceil(data.total / pageSize)} ·{" "}
              {data.items.length} / {data.total}
            </span>
            <div className="flex items-center gap-2">
              <button
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                className="btn btn-outline disabled:opacity-30"
              >
                Prev
              </button>
              <button
                disabled={(page + 1) * pageSize >= data.total}
                onClick={() => setPage((p) => p + 1)}
                className="btn btn-outline disabled:opacity-30"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </Panel>

      {/* (RunButton is defined below) */}

      {/* Typology mini-breakdown */}
      {stats && (
        <Panel
          title="Typology mix"
          subtitle="Inferred from persisted traces"
          className="mx-6 mt-4"
        >
          <div className="flex flex-wrap gap-2 p-4">
            {Object.entries(stats.by_typology)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => (
                <span
                  key={k}
                  className="chip border-[rgba(118,185,0,0.3)] bg-[rgba(118,185,0,0.05)]"
                  style={{ borderColor: typologyHue[k] + "55" }}
                >
                  <span
                    className="h-1.5 w-1.5 rounded-sm"
                    style={{ background: typologyHue[k] ?? "#64748b" }}
                  />
                  {titleCase(k)} · <span className="font-mono">{v}</span>
                </span>
              ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

function RunButton({ caseId }: { caseId: string }) {
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  async function go(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (busy) return;
    setBusy(true);
    try {
      await runInvestigation(caseId);
      setDone(true);
      // Invalidate caches that depend on this case
      await mutate((k) => Array.isArray(k) && k[0] === "alerts");
      await mutate("alerts-stats");
    } finally {
      setBusy(false);
    }
  }
  return (
    <button
      onClick={go}
      disabled={busy || done}
      className="text-[11px] inline-flex items-center gap-1 px-2 py-1 rounded border border-nv-green/40 text-nv-green hover:bg-nv-green/10 disabled:opacity-50"
      title="POST /api/investigation/run"
    >
      <Play size={10} />
      {busy ? "Running…" : done ? "Done" : "Run"}
    </button>
  );
}
