"use client";

import { useEffect, useState } from "react";
import { modelComparison, type ModelComparisonReport } from "@/lib/api";
import { PageHeader, Panel, Spinner } from "@/components/ui";
import { fmtPct, fmtMs } from "@/lib/format";
import { Trophy, AlertTriangle } from "lucide-react";
import clsx from "clsx";

export default function LeaderboardPage() {
  const [report, setReport] = useState<ModelComparisonReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const r = await modelComparison("latest");
        if ("error" in r) {
          setErr(r.error);
        } else {
          setReport(r);
        }
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
    })();
  }, []);

  return (
    <div className="pb-10">
      <PageHeader
        title={
          <span className="flex items-center gap-2">
            <Trophy size={18} className="text-nv-green" />
            Multi-model Leaderboard
          </span>
        }
        subtitle="Pre-compiled N-way scorecard from data/benchmarks/. Shows what fine-tuning actually bought us."
      />

      <div className="px-6 space-y-4">
        {busy && <Spinner label="Loading benchmark report…" />}
        {err && <NotAvailable err={err} />}
        {report && <Report r={report} />}
      </div>
    </div>
  );
}

function NotAvailable({ err }: { err: string }) {
  const notRegistered = err.toLowerCase().includes("not found");
  return (
    <Panel title="Benchmark report unavailable">
      <div className="p-6 text-xs space-y-2">
        <div className="flex items-start gap-2">
          <AlertTriangle size={14} className="text-amber-500 mt-0.5" />
          <div>
            {notRegistered ? (
              <>
                The backend route{" "}
                <code className="font-mono">POST /api/demo/eval/model_comparison</code>{" "}
                is not registered on the current NAT deployment.
              </>
            ) : (
              <>Backend error: {err}</>
            )}
          </div>
        </div>
        <div className="text-muted leading-relaxed mt-3">
          To populate this view, run the offline pipeline in{" "}
          <code className="font-mono">/backend/scripts</code>:
        </div>
        <pre className="font-mono text-[11px] surface-muted rounded p-3 overflow-auto">
{`python -m scripts.compare_endpoints --concurrency 6 ...
python -m scripts.score_traces --traces data/traces_<run> ...
python -m scripts.build_model_comparison_report \\
    --eval data/eval_<run>.json:"<label>" ... \\
    --update-latest \\
    --out data/benchmarks/four_way_<ts>.json`}
        </pre>
        <div className="text-muted">
          Once <code className="font-mono">data/benchmarks/latest.json</code>{" "}
          exists and the route is registered in <code className="font-mono">workflow.yaml</code>, this page will load it.
        </div>
      </div>
    </Panel>
  );
}

function Report({ r }: { r: ModelComparisonReport }) {
  const labels = Object.keys(r.headline_metrics[0]?.values ?? {});
  return (
    <>
      <Panel
        title={r.report_file}
        subtitle={
          <span className="font-mono text-[11px]">
            ran_at {r.ran_at} · demo {r.demo_size ?? "?"} ({r.demo_version ?? "—"})
          </span>
        }
      >
        {r.notes && (
          <div className="p-4 border-b divider text-xs text-muted leading-relaxed">
            {r.notes}
          </div>
        )}
      </Panel>

      <Panel
        title="Headline metrics"
        subtitle="Per-metric winner highlighted"
      >
        <div className="overflow-auto">
          <table className="w-full text-xs">
            <thead className="text-muted">
              <tr className="text-left border-b divider">
                <th className="py-2 px-3">Metric</th>
                {labels.map((l) => (
                  <th key={l} className="py-2 px-3 text-right">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {r.headline_metrics.map((m) => (
                <tr key={m.metric} className="t-row border-b">
                  <td className="py-2 px-3 font-medium">
                    {m.metric}
                    <span className="text-[10px] text-muted ml-2">
                      {m.higher_is_better ? "↑ better" : "↓ better"}
                    </span>
                  </td>
                  {labels.map((l) => {
                    const v = m.values[l];
                    const win = l === m.winner_label;
                    return (
                      <td
                        key={l}
                        className={clsx(
                          "py-2 px-3 text-right font-mono",
                          win
                            ? "text-nv-green font-semibold"
                            : "text-muted",
                        )}
                      >
                        {v != null ? fmtPct(v, 1) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Confusion matrix" subtitle="TP / FP / TN / FN per endpoint">
        <table className="w-full text-xs">
          <thead className="text-muted">
            <tr className="text-left border-b divider">
              <th className="py-2 px-3">Endpoint</th>
              <th className="py-2 px-3 text-right">TP</th>
              <th className="py-2 px-3 text-right">FP</th>
              <th className="py-2 px-3 text-right">TN</th>
              <th className="py-2 px-3 text-right">FN</th>
            </tr>
          </thead>
          <tbody>
            {r.confusion.map((c) => (
              <tr key={c.label} className="t-row border-b">
                <td className="py-2 px-3">{c.label}</td>
                <td className="py-2 px-3 text-right font-mono text-emerald-500">
                  {c.tp}
                </td>
                <td className="py-2 px-3 text-right font-mono text-rose-500">
                  {c.fp}
                </td>
                <td className="py-2 px-3 text-right font-mono text-emerald-500">
                  {c.tn}
                </td>
                <td className="py-2 px-3 text-right font-mono text-rose-500">
                  {c.fn}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      {r.wall_clock_ms && (
        <Panel title="Latency" subtitle="Wall-clock ms (lower is better)">
          <table className="w-full text-xs">
            <thead className="text-muted">
              <tr className="text-left border-b divider">
                <th className="py-2 px-3">Statistic</th>
                {labels.map((l) => (
                  <th key={l} className="py-2 px-3 text-right">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {r.wall_clock_ms.map((row) => (
                <tr key={row.field} className="t-row border-b">
                  <td className="py-2 px-3 font-medium">{row.field}</td>
                  {labels.map((l) => (
                    <td key={l} className="py-2 px-3 text-right font-mono">
                      {row.values[l] != null ? fmtMs(row.values[l]) : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}

      {r.per_typology_recall && (
        <Panel title="Per-typology recall" subtitle="Best endpoint per row highlighted">
          <table className="w-full text-xs">
            <thead className="text-muted">
              <tr className="text-left border-b divider">
                <th className="py-2 px-3">Typology</th>
                {labels.map((l) => (
                  <th key={l} className="py-2 px-3 text-right">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {r.per_typology_recall.map((row) => (
                <tr key={row.typology} className="t-row border-b">
                  <td className="py-2 px-3">{row.typology}</td>
                  {labels.map((l) => {
                    const v = row.values[l];
                    const win = l === row.winner_label;
                    return (
                      <td
                        key={l}
                        className={clsx(
                          "py-2 px-3 text-right font-mono",
                          win
                            ? "text-nv-green font-semibold"
                            : "text-muted",
                        )}
                      >
                        {v != null ? fmtPct(v, 1) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </>
  );
}
