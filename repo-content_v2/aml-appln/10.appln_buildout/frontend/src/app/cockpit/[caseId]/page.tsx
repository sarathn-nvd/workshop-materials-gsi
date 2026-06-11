"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";
import Link from "next/link";
import {
  getAlert,
  getTrace,
  postDisposition,
  runInvestigation,
  type Trace,
  type GateDecision,
} from "@/lib/api";
import { PageHeader, Panel, Spinner } from "@/components/ui";
import { statusChip, titleCase, typologyHue, fmtUsd, fmtMs } from "@/lib/format";
import {
  Activity,
  Wrench,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Scale,
  Calculator,
  Quote,
  Gavel,
  ChevronDown,
  ChevronRight,
  ArrowLeftCircle,
  ShieldCheck,
  Play,
  Download,
} from "lucide-react";
import clsx from "clsx";

export default function CockpitDetailPage({
  params,
}: {
  params: { caseId: string };
}) {
  const { caseId } = params;
  const { data: alert } = useSWR(["alert", caseId], () => getAlert(caseId));
  const { data: traceResp } = useSWR(["trace", caseId], () =>
    getTrace(caseId).catch((e) => ({ error: String(e) } as unknown as Trace)),
  );
  // The CaseTrace schema carries `error: null` on success, so treat the trace
  // as "missing" only when (a) we got an error envelope from the network, or
  // (b) the case_id field is absent from the response (the backend's
  // "trace not found" branch returns `{error: "..."}` with no other fields).
  const errVal =
    traceResp && typeof traceResp === "object" && "error" in (traceResp as object)
      ? (traceResp as { error?: string | null }).error
      : null;
  const hasCaseId =
    !!traceResp &&
    typeof traceResp === "object" &&
    !!(traceResp as { case_id?: string }).case_id;
  const trace: Trace | null = hasCaseId ? (traceResp as Trace) : null;
  const traceMissing = !!traceResp && !hasCaseId && !!errVal;

  return (
    <div className="pb-10">
      <PageHeader
        title={
          <span className="flex items-center gap-3">
            <span className="font-mono text-nv-green">{caseId}</span>
            {trace?.typology_hypothesis && (
              <span
                className="chip"
                style={{
                  background:
                    (typologyHue[trace.typology_hypothesis] ?? "#64748b") + "20",
                  borderColor:
                    (typologyHue[trace.typology_hypothesis] ?? "#64748b") + "55",
                  color: typologyHue[trace.typology_hypothesis] ?? "#64748b",
                }}
              >
                {titleCase(trace.typology_hypothesis)}
              </span>
            )}
            {trace && (
              <span
                className={clsx(
                  "chip",
                  trace.sar_is_suspicious ? "chip-danger" : "chip-success",
                )}
              >
                {trace.sar_is_suspicious ? "SUSPICIOUS" : "NOT SUSPICIOUS"}
              </span>
            )}
          </span>
        }
        subtitle={
          alert ? (
            <span className="font-mono text-xs">
              {alert.alert.alert_id} · entity {alert.alert.entity_id} ·{" "}
              {alert.alert.investigation_window_start} →{" "}
              {alert.alert.investigation_window_end}
            </span>
          ) : null
        }
        actions={
          <div className="flex items-center gap-2">
            {trace && (
              <button
                onClick={() => downloadTrace(trace)}
                className="btn btn-outline"
                title="Download the full CaseTrace JSON"
              >
                <Download size={14} />
                Trace JSON
              </button>
            )}
            <Link href="/alerts" className="btn btn-outline">
              <ArrowLeftCircle size={14} />
              Queue
            </Link>
          </div>
        }
      />

      {!alert ? (
        <Spinner label="Loading alert…" />
      ) : traceMissing ? (
        <NoTraceState caseId={caseId} alert={alert.alert} />
      ) : !trace ? (
        <Spinner label="Loading trace…" />
      ) : (
        <div className="px-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Left column — investigation timeline */}
          <div className="lg:col-span-2 space-y-4">
            <Header alert={alert.alert} trace={trace} />
            <PhasePanel trace={trace} />
            <OrchestratorPanel calls={trace.orchestrator_calls} />
            <AuxFindingsPanel trace={trace} />
            <GatePanel decisions={trace.aux_gate_decisions} />
            <SarPanel trace={trace} />
          </div>

          {/* Right column — evidence + disposition */}
          <div className="space-y-4">
            <SidePanelSummary trace={trace} kycSnippet={alert.kyc_snippet} />
            <EvidencePanel trace={trace} />
            <DispositionPanel
              caseId={caseId}
              status={alert.status}
              existing={alert.disposition}
            />
          </div>
        </div>
      )}
    </div>
  );
}

/* -------------------------- Sub-components -------------------------- */

function NoTraceState({
  caseId,
  alert,
}: {
  caseId: string;
  alert: { trigger_summary: string; entity_id: string };
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function go() {
    setBusy(true);
    setErr(null);
    try {
      await runInvestigation(caseId);
      await mutate(["trace", caseId]);
      await mutate(["alert", caseId]);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="px-6">
      <Panel
        title="No trace yet for this case"
        subtitle="Run the deterministic 7-phase workflow to produce a CaseTrace and a SAR verdict"
      >
        <div className="p-6 space-y-4">
          <div className="text-sm">
            <span className="text-muted">Trigger: </span>
            {alert.trigger_summary}
          </div>
          <div className="text-xs text-muted">
            POST <code className="font-mono text-nv-green">/api/investigation/run</code> with{" "}
            <code className="font-mono">{`{ "case_id": "${caseId}" }`}</code>. The model produces aux findings
            and the final SAR judgment; the trace is persisted to{" "}
            <code className="font-mono">./data/traces/{caseId}.json</code>.
          </div>
          <button
            disabled={busy}
            onClick={go}
            className="btn btn-primary"
          >
            <Play size={14} />
            {busy ? "Investigation in progress (15–30s)…" : "Run investigation"}
          </button>
          {err && (
            <div className="text-xs text-rose-500 mt-2">
              {err}
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}

function downloadTrace(trace: Trace) {
  const blob = new Blob([JSON.stringify(trace, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${trace.case_id}_trace.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function Header({
  alert,
  trace,
}: {
  alert: { trigger_summary: string };
  trace: Trace;
}) {
  return (
    <Panel className="!p-0">
      <div className="p-4">
        <div className="text-xs uppercase tracking-wider text-muted">
          Trigger
        </div>
        <div className="text-sm mt-1">{alert.trigger_summary}</div>
        <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
          <div className="surface-muted rounded-md p-2">
            <div className="text-muted">Activity descriptor</div>
            <div className="mt-0.5 leading-snug">{trace.activity_descriptor || "—"}</div>
          </div>
          <div className="surface-muted rounded-md p-2">
            <div className="text-muted">Typology (internal routing)</div>
            <div className="mt-0.5 font-mono">
              {trace.typology_hypothesis || "—"}
            </div>
          </div>
          <div className="surface-muted rounded-md p-2">
            <div className="text-muted">Wall clock</div>
            <div className="mt-0.5 font-mono">
              {fmtMs(trace.wall_clock_ms || 0)}
            </div>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function PhasePanel({ trace }: { trace: Trace }) {
  const phases = [
    { n: 1, label: "Data fetch", detail: `${trace.transactions.length} tx · KYC · ${trace.sanctions_pep_hits.length} sanctions hits` },
    { n: 2, label: "Typology guess (internal)", detail: trace.typology_hypothesis || "—" },
    { n: 3, label: "Retrieval", detail: `${trace.policy_excerpts.length} policy · ${trace.sop_excerpts.length} SOP excerpts` },
    { n: 4, label: "Aux skills", detail: "behavioral (Python) · numeric · citation · statutory" },
    { n: 5, label: "Aux gate", detail: `${trace.aux_gate_decisions.filter((d) => d.used).length}/${trace.aux_gate_decisions.length} findings kept` },
    { n: 6, label: "SAR judgment", detail: trace.sar_is_suspicious ? "suspicious" : "not suspicious" },
    { n: 7, label: "Trace persisted", detail: `wall clock ${fmtMs(trace.wall_clock_ms || 0)}` },
  ];
  return (
    <Panel
      title="Deterministic 7-phase workflow"
      subtitle="The NAT workflow walks fixed phases — the trained model is asked only to produce auxiliary findings and the final SAR verdict"
    >
      <ol className="rail relative pl-6 py-3">
        {phases.map((p) => (
          <li key={p.n} className="relative pl-2 py-1.5">
            <span className="absolute left-[-19px] top-2.5 h-2.5 w-2.5 rounded-full bg-nv-green ring-4 ring-nv-green/15" />
            <div className="flex items-center gap-2 text-sm">
              <span className="font-mono text-muted">Phase {p.n}</span>
              <span className="font-medium">{p.label}</span>
            </div>
            <div className="mt-0.5 text-xs text-muted">{p.detail}</div>
          </li>
        ))}
      </ol>
    </Panel>
  );
}

function OrchestratorPanel({
  calls,
}: {
  calls: Trace["orchestrator_calls"];
}) {
  if (!calls || calls.length === 0)
    return (
      <Panel
        title="Orchestrator · tool calls"
        subtitle="Deterministic NAT workflow (no LLM tool-calling for the workshop)"
      >
        <div className="px-4 py-3 text-xs text-muted">
          This deployment uses a deterministic `investigate_case` workflow, so the
          orchestrator does not emit explicit tool-calls. See the
          inputs/findings/SAR sections below for the actual artifacts.
        </div>
      </Panel>
    );
  return (
    <Panel
      title="Orchestrator · tool calls"
      subtitle="Ordered list of tools the orchestrator chose to invoke"
    >
      <ol className="rail relative pl-6 py-3">
        {calls.map((c, i) => (
          <li key={i} className="relative pl-2 py-2">
            <span className="absolute left-[-19px] top-3 h-2.5 w-2.5 rounded-full bg-nv-green ring-4 ring-nv-green/15" />
            <div className="flex items-center gap-2 text-sm">
              <Wrench size={12} className="text-nv-green" />
              <span className="font-mono">{c.tool}</span>
            </div>
            <div className="mt-0.5 text-xs text-muted">{c.result_summary}</div>
          </li>
        ))}
      </ol>
    </Panel>
  );
}

function AuxFindingsPanel({ trace }: { trace: Trace }) {
  const f = trace.auxiliary_findings || {};
  const TABS = [
    { key: "behavioral", icon: Activity, color: "#76b900" },
    { key: "numeric", icon: Calculator, color: "#06b6d4" },
    { key: "citation", icon: Quote, color: "#a855f7" },
    { key: "statutory", icon: Scale, color: "#f97316" },
  ] as const;
  const [tab, setTab] = useState<(typeof TABS)[number]["key"]>("behavioral");
  const active = TABS.find((t) => t.key === tab)!;
  const findings = (f[tab] || []) as unknown as Record<string, unknown>[];
  return (
    <Panel
      title="Auxiliary specialist findings"
      subtitle="3 LLM specialists (numeric · citation · statutory) + 1 deterministic Python computer (behavioral)"
      actions={
        <div className="flex items-center gap-1">
          {TABS.map((t) => {
            const Icon = t.icon;
            return (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={clsx(
                  "btn",
                  tab === t.key
                    ? "border-nv-green/40 bg-nv-green/10 text-nv-green"
                    : "btn-ghost",
                )}
              >
                <Icon size={12} />
                {titleCase(t.key)}
              </button>
            );
          })}
        </div>
      }
    >
      <div className="p-4">
        {findings.length === 0 ? (
          <p className="text-xs text-muted">
            No <strong>{tab}</strong> finding for this case (either the gate
            dropped it, or the specialist returned nothing).
          </p>
        ) : (
          <div className="space-y-3">
            {findings.map((finding, idx) => (
              <FindingCard
                key={idx}
                kind={tab}
                color={active.color}
                data={finding}
              />
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}

function FindingCard({
  kind,
  color,
  data,
}: {
  kind: string;
  color: string;
  data: Record<string, unknown>;
}) {
  return (
    <div
      className="surface-muted rounded-lg p-3 border-l-2"
      style={{ borderLeftColor: color }}
    >
      {kind === "behavioral" && (
        <>
          <div className="text-xs text-muted">Summary</div>
          <div className="text-sm mt-0.5">{String(data.summary ?? "")}</div>
          <div className="text-xs text-muted mt-2">Metrics</div>
          <pre className="mt-1 font-mono text-[11px] surface-muted rounded p-2 overflow-auto">
            {JSON.stringify(data.metrics, null, 2)}
          </pre>
        </>
      )}
      {kind === "numeric" && (
        <>
          <div className="text-xs text-muted">Question</div>
          <div className="text-sm mt-0.5">{String(data.question ?? "")}</div>
          <div className="text-xs text-muted mt-2">Answer</div>
          <div className="text-sm mt-0.5 font-medium">{String(data.answer ?? "")}</div>
          <div className="text-xs text-muted mt-2">Calculation</div>
          <pre className="mt-1 font-mono text-[11px] surface-muted rounded p-2 whitespace-pre-wrap">
            {String(data.calculation ?? "")}
          </pre>
        </>
      )}
      {kind === "citation" && (
        <>
          <div className="text-xs text-muted">Answer</div>
          <div className="text-sm mt-0.5">{String(data.answer ?? "")}</div>
          <div className="text-xs text-muted mt-2">Evidence span</div>
          <blockquote className="mt-0.5 text-xs italic border-l-2 border-nv-green/30 pl-2">
            {String(data.evidence_span ?? "")}
          </blockquote>
        </>
      )}
      {kind === "statutory" && (
        <>
          <div className="flex items-center gap-2">
            <span
              className={clsx(
                "chip",
                data.label === "entailment"
                  ? "chip-success"
                  : data.label === "contradiction"
                    ? "chip-danger"
                    : "chip-warn",
              )}
            >
              {titleCase(String(data.label ?? "neutral"))}
            </span>
          </div>
          <div className="text-sm mt-2">{String(data.answer ?? "")}</div>
          <div className="text-xs text-muted mt-2">Reasoning</div>
          <pre className="mt-0.5 text-xs whitespace-pre-wrap text-muted leading-relaxed">
            {String(data.reasoning ?? "")}
          </pre>
        </>
      )}
    </div>
  );
}

function GatePanel({ decisions }: { decisions: GateDecision[] }) {
  return (
    <Panel
      title="Aux-gate inspector"
      subtitle="Per-finding USE / DROP outcome and the judge's rationale"
    >
      <table className="w-full text-xs">
        <thead className="text-muted">
          <tr className="text-left border-b divider">
            <th className="py-2 px-4">Specialist</th>
            <th className="py-2 px-4">Outcome</th>
            <th className="py-2 px-4">Reason</th>
            <th className="py-2 px-4">Reviewer</th>
          </tr>
        </thead>
        <tbody>
          {decisions.map((d) => (
            <tr key={d.task} className="t-row border-b">
              <td className="py-2 px-4 font-medium">{titleCase(d.task)}</td>
              <td className="py-2 px-4">
                <span
                  className={clsx("chip", d.used ? "chip-success" : "chip-danger")}
                >
                  {d.used ? (
                    <>
                      <CheckCircle2 size={10} /> USED
                    </>
                  ) : (
                    <>
                      <XCircle size={10} /> DROPPED
                    </>
                  )}
                </span>
              </td>
              <td className="py-2 px-4 text-muted">{d.reason || "—"}</td>
              <td className="py-2 px-4 max-w-[280px]">
                <div className="text-muted truncate" title={d.reviewer_explain}>
                  {d.reviewer_verdict ? (
                    <span
                      className={clsx(
                        "chip mr-1",
                        d.reviewer_verdict === "PASS"
                          ? "chip-success"
                          : "chip-warn",
                      )}
                    >
                      {d.reviewer_verdict}
                    </span>
                  ) : null}
                  {d.reviewer_explain || ""}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function SarPanel({ trace }: { trace: Trace }) {
  const [showRaw, setShowRaw] = useState(false);
  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          <Gavel size={14} className="text-nv-green" />
          SAR narrative
          {trace.sar_parse_error && (
            <span className="chip chip-warn">
              <AlertTriangle size={10} /> parse error
            </span>
          )}
        </span>
      }
      subtitle="Final output of `sar_judgment_caller` on the trained model"
      actions={
        <button onClick={() => setShowRaw(!showRaw)} className="btn btn-outline">
          {showRaw ? "Hide" : "View"} raw bundle
        </button>
      }
    >
      <div className="p-4">
        <div className="surface-muted border-l-2 border-nv-green/60 p-3 rounded">
          <div className="text-[11px] uppercase tracking-wide text-muted">
            {trace.sar_is_suspicious ? "SUSPICIOUS ACTIVITY REPORT" : "VERDICT"}
          </div>
          <p className="mt-1 text-sm leading-relaxed whitespace-pre-wrap">
            {trace.sar_narrative ||
              (trace.sar_output
                ? trace.sar_output.suspicious_activity_report
                : "—")}
          </p>
        </div>

        {showRaw && (
          <div className="mt-3 space-y-3">
            <Collapsible title="user_message (sent to NIM)">
              <pre className="text-[11px] font-mono surface-muted rounded p-3 overflow-auto max-h-72">
                {trace.sar_user_message}
              </pre>
            </Collapsible>
            <Collapsible title="raw_text (NIM completion)">
              <pre className="text-[11px] font-mono surface-muted rounded p-3 overflow-auto max-h-72">
                {trace.sar_raw_text}
              </pre>
            </Collapsible>
          </div>
        )}
      </div>
    </Panel>
  );
}

function Collapsible({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border divider rounded">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-muted hover:text-[rgb(var(--fg))]"
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {title}
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </div>
  );
}

function SidePanelSummary({
  trace,
  kycSnippet,
}: {
  trace: Trace;
  kycSnippet: Record<string, string> | null;
}) {
  return (
    <Panel title="Case summary">
      <div className="p-4 space-y-3">
        {kycSnippet && (
          <div>
            <div className="text-xs uppercase tracking-wider text-muted">KYC</div>
            <Link
              href={`/entities/${kycSnippet.entity_id}`}
              className="text-sm font-mono text-nv-green hover:underline mt-0.5 block"
            >
              {kycSnippet.entity_id}
            </Link>
            <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
              <div className="text-muted">Type</div>
              <div className="text-right">
                {titleCase(kycSnippet.entity_type || "—")}
              </div>
              <div className="text-muted">Risk rating</div>
              <div className="text-right">
                {titleCase(kycSnippet.risk_rating || "—")}
              </div>
              <div className="text-muted">Jurisdiction</div>
              <div className="text-right font-mono">
                {kycSnippet.incorporation_jurisdiction || "—"}
              </div>
            </div>
            <div className="mt-2 text-xs text-muted line-clamp-3">
              {kycSnippet.business_purpose}
            </div>
          </div>
        )}

        {trace.sanctions_pep_hits && trace.sanctions_pep_hits.length > 0 && (
          <div>
            <div className="text-xs uppercase tracking-wider text-muted">
              Sanctions / PEP hits
            </div>
            <ul className="mt-1 space-y-1">
              {trace.sanctions_pep_hits.slice(0, 4).map((h, i) => (
                <li
                  key={i}
                  className="text-xs flex items-center justify-between"
                >
                  <span className="truncate">{h.name}</span>
                  <span className="font-mono ml-2 text-amber-500">
                    {(h.match_score * 100).toFixed(0)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div>
          <div className="text-xs uppercase tracking-wider text-muted">
            Transaction snapshot
          </div>
          <div className="mt-1 text-sm">
            {trace.transactions.length.toLocaleString()} tx ·{" "}
            <span className="font-mono">
              {fmtUsd(
                trace.transactions.reduce(
                  (s, t) => s + (Number(t.amount) || 0),
                  0,
                ),
                0,
              )}
            </span>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function EvidencePanel({ trace }: { trace: Trace }) {
  return (
    <Panel title="Policy & SOP excerpts" subtitle="What the agent grounded against">
      <div className="p-4 space-y-3 text-xs max-h-96 overflow-auto">
        {trace.policy_excerpts.length === 0 && (
          <div className="text-muted">No policy excerpts retrieved.</div>
        )}
        {trace.policy_excerpts.slice(0, 4).map((p, i) => (
          <div key={i} className="surface-muted rounded p-2 border-l-2 border-nv-green/40">
            <div className="flex items-center justify-between">
              <span className="font-medium">{p.source}</span>
              <span className="text-muted">{p.section}</span>
            </div>
            <p className="mt-1 line-clamp-3 text-muted">{p.text}</p>
          </div>
        ))}
        {trace.sop_excerpts.length > 0 && (
          <>
            <div className="text-[11px] uppercase tracking-wider text-muted mt-2">
              SOP
            </div>
            {trace.sop_excerpts.slice(0, 2).map((s, i) => (
              <div
                key={i}
                className="surface-muted rounded p-2 border-l-2 border-sky-500/40"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{s.sop_id}</span>
                  <span className="text-muted">{s.section}</span>
                </div>
                <p className="mt-1 line-clamp-3 text-muted">{s.text}</p>
              </div>
            ))}
          </>
        )}
      </div>
    </Panel>
  );
}

function DispositionPanel({
  caseId,
  status,
  existing,
}: {
  caseId: string;
  status: string;
  existing: { verdict: string; note: string; ts: number } | null;
}) {
  const [verdict, setVerdict] = useState<string>(existing?.verdict || "file_sar");
  const [note, setNote] = useState<string>(existing?.note || "");
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          <ShieldCheck size={14} className="text-nv-green" />
          Analyst disposition
        </span>
      }
      subtitle="Records the case verdict to /api/alerts/{id}/disposition"
    >
      <div className="p-4 space-y-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">Status</span>
          <span className={statusChip(status)}>{titleCase(status)}</span>
        </div>

        <div>
          <div className="text-xs text-muted mb-1">Verdict</div>
          <div className="grid grid-cols-3 gap-1">
            {(["file_sar", "dismiss", "escalate"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setVerdict(v)}
                className={clsx(
                  "px-2 py-1.5 rounded-md text-xs border",
                  verdict === v
                    ? "bg-nv-green text-black border-nv-green"
                    : "border-[rgb(var(--line))] hover:bg-[rgb(var(--line))]",
                )}
              >
                {titleCase(v)}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-muted mb-1">Rationale</div>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={3}
            className="w-full text-xs border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
            placeholder="Why this verdict?"
          />
        </div>

        <button
          disabled={busy}
          className="btn btn-primary w-full justify-center"
          onClick={async () => {
            setBusy(true);
            try {
              await postDisposition(caseId, { verdict, note });
              setSaved(true);
              setTimeout(() => setSaved(false), 2000);
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Saving…" : saved ? "Saved" : existing ? "Update" : "File"}
        </button>

        {existing && (
          <div className="text-[11px] text-muted">
            Previously recorded {new Date(existing.ts * 1000).toLocaleString()} as{" "}
            <strong>{titleCase(existing.verdict)}</strong>.
          </div>
        )}
      </div>
    </Panel>
  );
}
