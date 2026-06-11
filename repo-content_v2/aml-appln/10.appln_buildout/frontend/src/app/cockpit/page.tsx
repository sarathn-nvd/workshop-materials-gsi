"use client";
import useSWR from "swr";
import Link from "next/link";
import { listAlerts } from "@/lib/api";
import { PageHeader, Panel, Spinner } from "@/components/ui";
import { Telescope, ArrowRight } from "lucide-react";

export default function CockpitIndex() {
  const { data } = useSWR("cockpit-recent", () => listAlerts({ limit: 12 }));
  return (
    <div className="pb-10">
      <PageHeader
        title="Investigation Cockpit"
        subtitle="Pick a case below to see the planner plan, orchestrator tool calls, aux findings, and the SAR narrative."
      />
      <Panel
        className="mx-6"
        title="Recently triaged"
        subtitle="First page of the manifest"
      >
        {!data ? (
          <Spinner />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 p-4">
            {data.items.map((a) => (
              <Link
                key={a.case_id}
                href={`/cockpit/${a.case_id}`}
                className="surface rounded-lg p-4 hover:border-nv-green/40 hover:shadow-nv-glow transition-shadow group"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-wider text-muted">
                    {a.case_id}
                  </span>
                  <Telescope size={14} className="text-nv-green" />
                </div>
                <div className="mt-1 text-sm font-medium line-clamp-2 min-h-[40px]">
                  {a.trigger_summary}
                </div>
                <div className="mt-3 flex items-center justify-between text-[11px] text-muted">
                  <span className="font-mono">{a.entity_id}</span>
                  <span className="inline-flex items-center gap-1 text-nv-green opacity-0 group-hover:opacity-100 transition-opacity">
                    Open
                    <ArrowRight size={12} />
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
