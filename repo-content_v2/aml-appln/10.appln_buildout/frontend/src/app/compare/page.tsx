"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";
import { evalRuns, evalCompare, type EvalCompareResp } from "@/lib/api";
import { PageHeader, Panel, Spinner } from "@/components/ui";
import { fmtPct, fmtMs } from "@/lib/format";
import clsx from "clsx";
import { GitCompare, TrendingUp, TrendingDown, Minus } from "lucide-react";

export default function ComparePage() {
  const { data: runs } = useSWR("eval-runs", evalRuns);
  const [runA, setRunA] = useState<string>("");
  const [runB, setRunB] = useState<string>("");
  const [result, setResult] = useState<EvalCompareResp | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Default selection on load
  useEffect(() => {
    if (!runs || runA) return;
    setRunA("traces");
    const fallback = runs.items.find(
      (r) => r.name !== "traces" && r.name.startsWith("traces_"),
    );
    if (fallback) setRunB(fallback.name);
  }, [runs, runA]);

  async function run() {
    if (!runA || !runB) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await evalCompare({
        run_a: runA,
        run_b: runB,
        label_a: prettyLabel(runA),
        label_b: prettyLabel(runB),
      });
      setResult(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="pb-10">
      <PageHeader
        title="Model Comparison"
        subtitle="Score any two trace snapshots against the same ground truth."
      />

      <Panel
        title="Pick two runs"
        subtitle="Snapshots under /backend/data/ — anything starting with traces_ is eligible"
        className="mx-6"
        actions={
          <button
            onClick={run}
            disabled={!runA || !runB || runA === runB || busy}
            className="btn btn-primary"
          >
            <GitCompare size={14} />
            {busy ? "Scoring…" : "Compare"}
          </button>
        }
      >
        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          <RunSelect
            label="Run A (left)"
            runs={runs?.items}
            value={runA}
            onChange={setRunA}
          />
          <RunSelect
            label="Run B (right)"
            runs={runs?.items}
            value={runB}
            onChange={setRunB}
          />
        </div>
        {err && (
          <div className="px-4 pb-3 text-xs text-rose-500">{err}</div>
        )}
      </Panel>

      {result && (
        <div className="px-6 mt-4 space-y-4">
          <GroundTruthBar gt={result.ground_truth} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <RunCard run={result.run_a} accent="#76b900" />
            <RunCard run={result.run_b} accent="#06b6d4" />
          </div>

          <DiffMatrix result={result} />
        </div>
      )}

      {!result && !busy && runs && (
        <div className="px-6 mt-4 text-xs text-muted">
          Press <strong>Compare</strong> to score both runs and render side-by-side metrics.
        </div>
      )}
    </div>
  );
}

function RunSelect({
  label,
  runs,
  value,
  onChange,
}: {
  label: string;
  runs?: { name: string; n_traces: number; is_active: boolean }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
        {label}
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full border divider rounded-md surface-muted py-2 px-2 text-sm focus:outline-none focus:border-nv-green/60"
      >
        <option value="">Select a snapshot…</option>
        {runs?.map((r) => (
          <option key={r.name} value={r.name}>
            {prettyLabel(r.name)} ({r.n_traces} traces{r.is_active ? " · active" : ""})
          </option>
        ))}
      </select>
    </label>
  );
}

function GroundTruthBar({
  gt,
}: {
  gt: { n_total: number; n_sar: number; n_no_sar: number; n_near_miss: number };
}) {
  return (
    <Panel title="Ground truth" subtitle={`${gt.n_total} cases in eval_keys.jsonl`}>
      <div className="grid grid-cols-4 divide-x divider">
        <Cell label="Total" value={gt.n_total} />
        <Cell label="SAR" value={gt.n_sar} accent="text-nv-green" />
        <Cell label="No SAR" value={gt.n_no_sar} />
        <Cell label="Near-miss" value={gt.n_near_miss} accent="text-amber-500" />
      </div>
    </Panel>
  );
}

function Cell({
  label,
  value,
  accent,
}: {
  label: string;
  value: number;
  accent?: string;
}) {
  return (
    <div className="p-4">
      <div className="text-[11px] uppercase tracking-wider text-muted">
        {label}
      </div>
      <div className={clsx("mt-1 text-2xl font-semibold", accent)}>{value}</div>
    </div>
  );
}

function RunCard({
  run,
  accent,
}: {
  run: EvalCompareResp["run_a"];
  accent: string;
}) {
  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          <span
            className="h-2.5 w-2.5 rounded-full"
            style={{ background: accent }}
          />
          {run.label}
        </span>
      }
      subtitle={
        <span className="font-mono text-[11px]">
          {run.name} · {run.n_predictions} predictions · {run.parse_errors} parse errors
        </span>
      }
    >
      <div className="p-4 space-y-4">
        <Confusion conf={run.confusion} fp={run.fp_breakdown} />
        <Metrics m={run.metrics} />
        <div className="text-xs text-muted">
          Latency · avg{" "}
          <span className="font-mono">{fmtMs(run.latency.avg_case_ms)}</span> over{" "}
          {run.latency.n_with_timing} cases
        </div>
      </div>
    </Panel>
  );
}

function Confusion({
  conf,
  fp,
}: {
  conf: { tp: number; fp: number; tn: number; fn: number };
  fp: { near_miss: number; clean: number };
}) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-muted mb-2">
        Confusion matrix
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <Quad label="TP" value={conf.tp} good />
        <Quad label="FP" value={conf.fp} bad sub={`${fp.near_miss} near · ${fp.clean} clean`} />
        <Quad label="FN" value={conf.fn} bad />
        <Quad label="TN" value={conf.tn} good />
      </div>
    </div>
  );
}

function Quad({
  label,
  value,
  good,
  bad,
  sub,
}: {
  label: string;
  value: number;
  good?: boolean;
  bad?: boolean;
  sub?: string;
}) {
  return (
    <div
      className={clsx(
        "surface-muted rounded-md p-3 border-l-2",
        good && "border-emerald-500",
        bad && "border-rose-500",
      )}
    >
      <div className="text-[10px] uppercase tracking-wider text-muted">
        {label}
      </div>
      <div className="text-xl font-semibold font-mono mt-0.5">{value}</div>
      {sub && <div className="text-[10px] text-muted mt-0.5">{sub}</div>}
    </div>
  );
}

function Metrics({ m }: { m: Record<string, number> }) {
  const ROWS = [
    ["accuracy", "Accuracy"],
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1", "F1"],
    ["macro_f1", "Macro F1"],
    ["false_positive_rate_clean", "FPR (clean)"],
    ["false_positive_rate_near_miss", "FPR (near-miss)"],
    ["near_miss_specificity", "Near-miss specificity"],
    ["narrative_grounding_rate", "Grounding rate"],
  ] as const;
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
      {ROWS.map(([k, lab]) => (
        <div key={k} className="flex justify-between">
          <span className="text-muted">{lab}</span>
          <span className="font-mono">
            {m[k] != null ? fmtPct(m[k]) : "—"}
          </span>
        </div>
      ))}
    </div>
  );
}

function DiffMatrix({ result }: { result: EvalCompareResp }) {
  const ENTRIES: { key: keyof EvalCompareResp["diff"]["metrics"]; label: string; goodIsUp: boolean }[] = [
    { key: "f1", label: "F1", goodIsUp: true },
    { key: "precision", label: "Precision", goodIsUp: true },
    { key: "recall", label: "Recall", goodIsUp: true },
    { key: "macro_f1", label: "Macro F1", goodIsUp: true },
  ];
  return (
    <Panel
      title="Run A − Run B"
      subtitle="Positive = run A is better on that metric"
    >
      <div className="p-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        {ENTRIES.map((e) => {
          const d = result.diff.metrics[e.key];
          if (!d) return null;
          const positive = d.absolute > 0;
          const good = e.goodIsUp ? positive : !positive;
          const Icon = d.absolute === 0 ? Minus : positive ? TrendingUp : TrendingDown;
          return (
            <div
              key={e.key}
              className={clsx(
                "surface-muted rounded-md p-3",
                good ? "ring-1 ring-emerald-500/40" : "ring-1 ring-rose-500/40",
              )}
            >
              <div className="text-[10px] uppercase tracking-wider text-muted">
                {e.label}
              </div>
              <div
                className={clsx(
                  "mt-0.5 text-lg font-mono flex items-center gap-1",
                  good ? "text-emerald-500" : "text-rose-500",
                )}
              >
                <Icon size={14} />
                {(d.absolute * 100).toFixed(1)} pp
              </div>
              <div className="text-[10px] text-muted mt-0.5">
                rel {d.relative_pct.toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>

      <div className="border-t divider p-4 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
        <DiffStat label="Δ TP" v={result.diff.confusion.tp} positiveIsGood />
        <DiffStat label="Δ FP" v={result.diff.confusion.fp} positiveIsGood={false} />
        <DiffStat label="Δ FN" v={result.diff.confusion.fn} positiveIsGood={false} />
        <DiffStat label="Δ TN" v={result.diff.confusion.tn} positiveIsGood />
        <DiffStat
          label="Δ latency (ms)"
          v={result.diff.latency_ms}
          positiveIsGood={false}
        />
        <DiffStat
          label="Δ FP near-miss"
          v={result.diff.fp_breakdown.near_miss}
          positiveIsGood={false}
        />
        <DiffStat
          label="Δ FP clean"
          v={result.diff.fp_breakdown.clean}
          positiveIsGood={false}
        />
      </div>
    </Panel>
  );
}

function DiffStat({
  label,
  v,
  positiveIsGood,
}: {
  label: string;
  v: number;
  positiveIsGood: boolean;
}) {
  const pos = v > 0;
  const good = positiveIsGood ? pos : !pos;
  return (
    <div className="flex justify-between">
      <span className="text-muted">{label}</span>
      <span
        className={clsx(
          "font-mono",
          v === 0 ? "" : good ? "text-emerald-500" : "text-rose-500",
        )}
      >
        {v > 0 ? "+" : ""}
        {Number.isInteger(v) ? v : v.toFixed(0)}
      </span>
    </div>
  );
}

function prettyLabel(name: string): string {
  if (name === "traces") return "Active (traces/)";
  return name
    .replace(/^traces_/, "")
    .split("_")
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}
