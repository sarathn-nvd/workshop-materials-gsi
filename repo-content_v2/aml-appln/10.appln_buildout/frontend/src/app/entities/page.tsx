"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { listEntities } from "@/lib/api";
import { PageHeader, Panel, Spinner, EmptyState } from "@/components/ui";
import { riskColor, titleCase } from "@/lib/format";
import { Search } from "lucide-react";

export default function EntitiesPage() {
  const [q, setQ] = useState("");
  const [risk, setRisk] = useState("");
  const [type, setType] = useState("");
  const [page, setPage] = useState(0);
  const pageSize = 25;

  const { data } = useSWR(["entities", q, risk, type, page], () =>
    listEntities({
      q: q || undefined,
      risk_rating: risk || undefined,
      entity_type: type || undefined,
      limit: pageSize,
      offset: page * pageSize,
    }),
  );

  return (
    <div className="pb-10">
      <PageHeader
        title="Entity 360"
        subtitle="Search the customer book; click a row for the full profile."
      />
      <Panel
        className="mx-6"
        title="KYC search"
        subtitle={data ? `${data.total.toLocaleString()} matching entities` : ""}
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
                placeholder="entity_id or business_purpose"
                className="pl-7 pr-2 py-1.5 text-xs border divider rounded-md surface-muted w-72 focus:outline-none focus:border-nv-green/60"
              />
            </div>
            <select
              className="text-xs border divider rounded-md surface-muted py-1.5 px-2"
              value={risk}
              onChange={(e) => {
                setRisk(e.target.value);
                setPage(0);
              }}
            >
              <option value="">All risk</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="enhanced">Enhanced</option>
              <option value="prohibited">Prohibited</option>
            </select>
            <select
              className="text-xs border divider rounded-md surface-muted py-1.5 px-2"
              value={type}
              onChange={(e) => {
                setType(e.target.value);
                setPage(0);
              }}
            >
              <option value="">All types</option>
              <option value="individual">Individual</option>
              <option value="business">Business</option>
            </select>
          </div>
        }
      >
        {!data ? (
          <Spinner />
        ) : data.items.length === 0 ? (
          <EmptyState title="No entities match" />
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-muted">
                <tr className="text-left border-b divider">
                  <th className="py-2.5 px-4">Entity</th>
                  <th className="py-2.5 px-4">Type</th>
                  <th className="py-2.5 px-4">Risk</th>
                  <th className="py-2.5 px-4">Jurisdiction</th>
                  <th className="py-2.5 px-4">Business purpose</th>
                  <th className="py-2.5 px-4 text-right">Expected monthly</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((e) => (
                  <tr key={e.entity_id} className="t-row border-b">
                    <td className="py-2 px-4">
                      <Link
                        href={`/entities/${e.entity_id}`}
                        className="font-mono text-nv-green hover:underline"
                      >
                        {e.entity_id}
                      </Link>
                    </td>
                    <td className="py-2 px-4 text-xs">
                      {titleCase(e.entity_type || "—")}
                    </td>
                    <td className={`py-2 px-4 text-xs font-medium ${riskColor(e.risk_rating)}`}>
                      {titleCase(e.risk_rating || "—")}
                    </td>
                    <td className="py-2 px-4 text-xs font-mono">
                      {e.incorporation_jurisdiction || "—"}
                    </td>
                    <td className="py-2 px-4 text-xs">
                      <div className="max-w-[300px] truncate" title={e.business_purpose}>
                        {e.business_purpose}
                      </div>
                    </td>
                    <td className="py-2 px-4 text-xs text-right font-mono">
                      {e.expected_monthly_volume
                        ? `$${e.expected_monthly_volume.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {data && data.total > pageSize && (
          <div className="flex items-center justify-between px-4 py-3 border-t divider text-xs text-muted">
            <span>
              Page {page + 1} of {Math.ceil(data.total / pageSize)}
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
    </div>
  );
}
