"use client";

import { useState } from "react";
import {
  skillBehavioral,
  skillNumeric,
  skillCitation,
  skillStatutory,
  skillSar,
} from "@/lib/api";
import { PageHeader, Panel, Spinner } from "@/components/ui";
import { Activity, Calculator, Quote, Scale, Gavel, Play, AlertTriangle } from "lucide-react";
import clsx from "clsx";

/* ----------------------- Samples ----------------------- */

const SAMPLE_BEHAVIORAL = `## KYC profile
entity_id: SYN_22da3c8f
entity_type: business
expected_monthly_volume: 50000
business_purpose: Cash-intensive retail operator (single-location convenience store).
risk_rating: low
incorporation_jurisdiction: US-NY

## Transactions
date,amount,currency,counterparty,channel,notes
2026-03-01,9500,USD,Branch ATM Deposit,cash,
2026-03-02,9800,USD,Branch ATM Deposit,cash,
2026-03-03,9750,USD,Branch ATM Deposit,cash,
2026-03-04,9900,USD,Branch ATM Deposit,cash,
2026-03-05,9650,USD,Branch ATM Deposit,cash,
2026-03-06,9700,USD,Branch ATM Deposit,cash,
2026-03-07,9550,USD,Branch ATM Deposit,cash,
2026-03-08,9450,USD,Branch ATM Deposit,cash,
`;

const SAMPLE_POLICY = `## Policy excerpt
source: FinCEN
section: Advisory FIN-2014-A005
text: Multiple cash deposits structured below the $10,000 threshold by cash-intensive businesses may constitute structuring under 31 U.S.C. § 5324. Indicators include repeated sub-threshold deposits across short windows, geographic dispersion across branches, and use of multiple bank accounts.`;

const SAMPLE_STATUTE_STATUTE = `31 U.S.C. § 5324(a)(3) — "No person shall, for the purpose of evading the reporting requirements... structure or assist in structuring, or attempt to structure or assist in structuring, any transaction with one or more domestic financial institutions."`;

const SAMPLE_STATUTE_FACTS = `Entity SYN_22da3c8f made 8 cash deposits of 9,450 – 9,900 USD across 4 branches in 8 days, all sub-$10K threshold. KYC declares expected monthly volume of $50,000; observed volume is $77,300 in 8 days.`;

const SAMPLE_SAR_BUNDLE = `{
  "transactions": [
    {"date": "2026-03-01", "amount": 9500, "currency": "USD", "counterparty": "Branch ATM Deposit", "channel": "cash", "notes": ""},
    {"date": "2026-03-02", "amount": 9800, "currency": "USD", "counterparty": "Branch ATM Deposit", "channel": "cash", "notes": ""},
    {"date": "2026-03-03", "amount": 9750, "currency": "USD", "counterparty": "Branch ATM Deposit", "channel": "cash", "notes": ""},
    {"date": "2026-03-04", "amount": 9900, "currency": "USD", "counterparty": "Branch ATM Deposit", "channel": "cash", "notes": ""},
    {"date": "2026-03-05", "amount": 9650, "currency": "USD", "counterparty": "Branch ATM Deposit", "channel": "cash", "notes": ""},
    {"date": "2026-03-06", "amount": 9700, "currency": "USD", "counterparty": "Branch ATM Deposit", "channel": "cash", "notes": ""}
  ],
  "kyc_profile": {
    "entity_id": "SYN_22da3c8f",
    "entity_type": "business",
    "expected_monthly_volume": 50000,
    "business_purpose": "Cash-intensive retail operator",
    "risk_rating": "low",
    "incorporation_jurisdiction": "US-NY"
  },
  "sanctions_pep_hits": [],
  "policy_excerpts": [
    {
      "source": "FinCEN",
      "section": "Advisory FIN-2014-A005",
      "url": "",
      "text": "Multiple cash deposits structured below the $10,000 threshold may constitute structuring under 31 U.S.C. § 5324."
    }
  ],
  "sop_excerpts": [
    {"sop_id": "SOP-STRUCTURING-01", "section": "Investigation Steps", "text": "Confirm sub-threshold pattern; verify aggregation across branches."}
  ],
  "auxiliary_findings": null
}`;

const TABS = [
  {
    key: "behavioral",
    label: "Behavioral",
    icon: Activity,
    color: "#76b900",
    sample: SAMPLE_BEHAVIORAL,
    fn: skillBehavioral,
    hasQuestion: false,
  },
  {
    key: "numeric",
    label: "Numeric",
    icon: Calculator,
    color: "#06b6d4",
    sample: SAMPLE_BEHAVIORAL,
    fn: skillNumeric,
    hasQuestion: true,
  },
  {
    key: "citation",
    label: "Citation",
    icon: Quote,
    color: "#a855f7",
    sample: SAMPLE_POLICY,
    fn: skillCitation,
    hasQuestion: true,
  },
  {
    key: "statutory",
    label: "Statutory",
    icon: Scale,
    color: "#f97316",
    sample: "", // handled separately (two-field input)
    fn: null,
    hasQuestion: true,
  },
  {
    key: "sar",
    label: "Final SAR",
    icon: Gavel,
    color: "#eab308",
    sample: "", // handled separately
    fn: null,
    hasQuestion: false,
  },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export default function SkillsPage() {
  const [tab, setTab] = useState<TabKey>("behavioral");
  const meta = TABS.find((t) => t.key === tab)!;

  return (
    <div className="pb-10">
      <PageHeader
        title="Skill Playgrounds"
        subtitle="Invoke a single auxiliary skill (or the final SAR call) directly against the trained model."
      />

      {/* Tab bar */}
      <div className="px-6 mb-3 flex flex-wrap items-center gap-1">
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
              <Icon size={14} />
              {t.label}
            </button>
          );
        })}
      </div>

      <div className="px-6">
        {tab === "sar" ? (
          <SarPlayground />
        ) : tab === "statutory" ? (
          <StatutoryPlayground />
        ) : (
          <SkillPlayground meta={meta} />
        )}
      </div>
    </div>
  );
}

function SkillPlayground({
  meta,
}: {
  meta: (typeof TABS)[number];
}) {
  const [passage, setPassage] = useState<string>(meta.sample);
  const [question, setQuestion] = useState(
    meta.key === "numeric"
      ? "Sum the cash deposits in this passage and compare to the declared monthly volume."
      : meta.key === "citation"
        ? "What does this excerpt say about structuring?"
        : meta.key === "statutory"
          ? "Does the conduct fall within 31 U.S.C. § 5324(a)(3)?"
          : "",
  );
  const [out, setOut] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function go() {
    setBusy(true);
    setErr(null);
    setOut(null);
    try {
      const r = await meta.fn!({ passage, question });
      setOut(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Panel
        title={`Input · ${meta.label.toLowerCase()}`}
        subtitle={`POST /api/skills/${meta.key}`}
      >
        <div className="p-4 space-y-3">
          {meta.hasQuestion && (
            <div>
              <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
                Question
              </div>
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="w-full text-sm border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
              />
            </div>
          )}
          <div>
            <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
              Passage
            </div>
            <textarea
              value={passage}
              onChange={(e) => setPassage(e.target.value)}
              rows={16}
              className="w-full text-xs font-mono border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
            />
          </div>
          <button onClick={go} disabled={busy} className="btn btn-primary">
            <Play size={14} />
            {busy ? "Running…" : "Run skill"}
          </button>
        </div>
      </Panel>

      <Panel
        title="Trained-model response"
        subtitle={`Sent with task_type=auxiliary_${meta.key}`}
      >
        <div className="p-4">
          {busy && <Spinner label="Calling Custom Task NIM…" />}
          {err && <ErrorBox text={err} />}
          {!busy && !err && out == null && (
            <p className="text-xs text-muted">
              Output will render here once the model returns.
            </p>
          )}
          {out != null && <ResponseRender data={out as Record<string, unknown>} />}
        </div>
      </Panel>
    </div>
  );
}

function ResponseRender({ data }: { data: Record<string, unknown> }) {
  if ("error" in data) {
    return (
      <div className="space-y-2">
        <ErrorBox text={String(data.error)} />
        {"raw" in data && data.raw ? (
          <div className="text-[11px] text-muted">Raw output:</div>
        ) : null}
        {"raw" in data && data.raw ? (
          <pre className="font-mono text-[11px] surface-muted rounded p-2 max-h-72 overflow-auto whitespace-pre-wrap">
            {String(data.raw)}
          </pre>
        ) : null}
      </div>
    );
  }
  return (
    <div className="space-y-3 text-sm">
      {Object.entries(data).map(([k, v]) => (
        <div key={k}>
          <div className="text-[10px] uppercase tracking-wider text-muted">
            {k}
          </div>
          {typeof v === "object" ? (
            <pre className="font-mono text-[11px] surface-muted rounded p-2 max-h-72 overflow-auto">
              {JSON.stringify(v, null, 2)}
            </pre>
          ) : (
            <div className="mt-0.5 whitespace-pre-wrap leading-relaxed">
              {String(v)}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ErrorBox({ text }: { text: string }) {
  return (
    <div className="surface-muted border-l-2 border-rose-500 rounded p-2 text-xs flex items-start gap-2">
      <AlertTriangle size={14} className="text-rose-500 mt-0.5 shrink-0" />
      <span className="text-muted whitespace-pre-wrap break-words">{text}</span>
    </div>
  );
}

function StatutoryPlayground() {
  const [statute, setStatute] = useState(SAMPLE_STATUTE_STATUTE);
  const [factPattern, setFactPattern] = useState(SAMPLE_STATUTE_FACTS);
  const [question, setQuestion] = useState(
    "Does the conduct fall within 31 U.S.C. § 5324(a)(3)?",
  );
  const [out, setOut] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function go() {
    setBusy(true);
    setErr(null);
    setOut(null);
    try {
      const r = await skillStatutory({
        statute,
        fact_pattern: factPattern,
        question,
      });
      setOut(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Panel
        title="Input · statutory"
        subtitle="POST /api/skills/statutory — separate statute + fact-pattern fields"
      >
        <div className="p-4 space-y-3">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
              Question
            </div>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="w-full text-sm border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
            />
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
              Statute
            </div>
            <textarea
              value={statute}
              onChange={(e) => setStatute(e.target.value)}
              rows={6}
              className="w-full text-xs font-mono border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
            />
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
              Fact pattern
            </div>
            <textarea
              value={factPattern}
              onChange={(e) => setFactPattern(e.target.value)}
              rows={8}
              className="w-full text-xs font-mono border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
            />
          </div>
          <button onClick={go} disabled={busy} className="btn btn-primary">
            <Play size={14} />
            {busy ? "Running…" : "Run skill"}
          </button>
        </div>
      </Panel>

      <Panel
        title="Trained-model response"
        subtitle="Sent with task_type=auxiliary_statutory"
      >
        <div className="p-4">
          {busy && <Spinner label="Calling Custom Task NIM…" />}
          {err && <ErrorBox text={err} />}
          {!busy && !err && out == null && (
            <p className="text-xs text-muted">
              Output will render here once the model returns.
            </p>
          )}
          {out != null && <ResponseRender data={out as Record<string, unknown>} />}
        </div>
      </Panel>
    </div>
  );
}

function SarPlayground() {
  const [body, setBody] = useState(SAMPLE_SAR_BUNDLE);
  const [out, setOut] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function go() {
    setBusy(true);
    setErr(null);
    setOut(null);
    try {
      const parsed = JSON.parse(body);
      const r = await skillSar(parsed);
      setOut(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Panel
        title="6-key evidence bundle (SAR caller)"
        subtitle="Pydantic extra='forbid' — only these 6 keys are accepted. task_type is added server-side."
      >
        <div className="p-4 space-y-3">
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={22}
            className="w-full text-xs font-mono border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
          />
          <button onClick={go} disabled={busy} className="btn btn-primary">
            <Play size={14} />
            {busy ? "Running…" : "Issue SAR call"}
          </button>
        </div>
      </Panel>

      <Panel
        title="SAR judgment"
        subtitle="Verdict + narrative + sent user_message"
      >
        <div className="p-4">
          {busy && <Spinner label="Drafting SAR…" />}
          {err && <ErrorBox text={err} />}
          {!busy && !err && out == null && (
            <p className="text-xs text-muted">
              The trained model's response will render here.
            </p>
          )}
          {out != null && <ResponseRender data={out as Record<string, unknown>} />}
        </div>
      </Panel>
    </div>
  );
}

