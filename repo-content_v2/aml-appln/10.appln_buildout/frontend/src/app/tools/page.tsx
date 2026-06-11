"use client";

import { useState } from "react";
import useSWR from "swr";
import ReactMarkdown from "react-markdown";
import {
  searchPolicy,
  policySources,
  listSops,
  getSop,
  screenSanctions,
} from "@/lib/api";
import { PageHeader, Panel, Spinner, EmptyState } from "@/components/ui";
import { titleCase } from "@/lib/format";
import {
  BookText,
  ClipboardList,
  ShieldQuestion,
  Search,
  Play,
} from "lucide-react";
import clsx from "clsx";

const TABS = [
  { key: "policy", label: "Policy RAG", icon: BookText },
  { key: "sops", label: "SOP browser", icon: ClipboardList },
  { key: "sanctions", label: "Sanctions screen", icon: ShieldQuestion },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export default function ToolsPage() {
  const [tab, setTab] = useState<TabKey>("policy");
  return (
    <div className="pb-10">
      <PageHeader
        title="Compliance Tools"
        subtitle="Standalone views into the same retrievers and screens the workflow uses internally."
      />
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
      <div className="px-6">
        {tab === "policy" && <PolicyTab />}
        {tab === "sops" && <SopTab />}
        {tab === "sanctions" && <SanctionsTab />}
      </div>
    </div>
  );
}

/* -------------------- Policy RAG -------------------- */

const TYPOLOGIES = [
  "structuring",
  "smurfing",
  "layering",
  "shell_company",
  "trade_based_ml",
  "human_trafficking",
  "terrorist_financing",
  "elder_exploitation",
];

function PolicyTab() {
  const { data: sources } = useSWR("policy-sources", policySources);
  const [typology, setTypology] = useState("structuring");
  const [q, setQ] = useState("");
  const [k, setK] = useState(4);
  const [items, setItems] = useState<
    | {
        source: string;
        section: string;
        url: string;
        text: string;
        match_offset?: number;
      }[]
    | null
  >(null);
  const [busy, setBusy] = useState(false);

  async function go() {
    setBusy(true);
    try {
      const r = await searchPolicy({ typology, q: q || undefined, k });
      setItems(r.items);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <Panel title="Policy corpus" subtitle="The agent's grounded reference shelf" className="lg:col-span-1">
        <div className="p-4 space-y-3 text-xs">
          {!sources ? (
            <Spinner />
          ) : (
            <>
              <div className="text-muted">
                {(
                  sources.n_chunks ??
                  Object.values(sources.sources ?? {}).reduce(
                    (s, v) => s + v,
                    0,
                  )
                ).toLocaleString()}{" "}
                chunks
              </div>
              {Object.entries(sources.sources ?? {}).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between">
                  <span className="font-medium">{k}</span>
                  <span className="font-mono">{v.toLocaleString()}</span>
                </div>
              ))}
            </>
          )}
        </div>
      </Panel>

      <Panel
        title="Stratified top-k search"
        subtitle="Same retriever the workflow uses (POST /api/policy/search)"
        className="lg:col-span-2"
        actions={
          <button onClick={go} disabled={busy} className="btn btn-primary">
            <Play size={12} />
            {busy ? "Searching…" : "Search"}
          </button>
        }
      >
        <div className="p-4 space-y-3">
          <div className="grid grid-cols-12 gap-2">
            <select
              value={typology}
              onChange={(e) => setTypology(e.target.value)}
              className="col-span-4 text-xs border divider rounded-md surface-muted py-2 px-2"
            >
              {TYPOLOGIES.map((t) => (
                <option key={t} value={t}>
                  {titleCase(t)}
                </option>
              ))}
            </select>
            <div className="relative col-span-6">
              <Search
                size={14}
                className="absolute left-2 top-1/2 -translate-y-1/2 text-muted pointer-events-none"
              />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Optional keyword (e.g. $10,000 threshold)"
                className="w-full pl-7 pr-2 py-2 text-xs border divider rounded-md surface-muted focus:outline-none focus:border-nv-green/60"
              />
            </div>
            <input
              type="number"
              min={1}
              max={10}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="col-span-2 text-xs border divider rounded-md surface-muted py-2 px-2"
              title="k"
            />
          </div>

          {items === null ? (
            <p className="text-xs text-muted">
              Pick a typology, optionally add a keyword, then run.
            </p>
          ) : items.length === 0 ? (
            <EmptyState title="No excerpts" />
          ) : (
            <div className="space-y-2">
              {items.map((it, i) => (
                <div
                  key={i}
                  className="surface-muted rounded p-3 border-l-2 border-nv-green/40"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium">{it.source}</span>
                    <span className="text-muted">{it.section}</span>
                  </div>
                  {it.url && (
                    <a
                      href={it.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[11px] text-nv-green hover:underline mt-0.5 block truncate"
                    >
                      {it.url}
                    </a>
                  )}
                  <p className="mt-1 text-xs leading-relaxed whitespace-pre-wrap">
                    {it.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}

/* -------------------- SOP browser -------------------- */

function SopTab() {
  const { data: index } = useSWR("sops-index", listSops);
  const [picked, setPicked] = useState<string | null>(null);
  const { data: sop } = useSWR(
    picked ? ["sop", picked] : null,
    picked ? () => getSop(picked) : null,
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <Panel title="Typology playbooks" className="lg:col-span-1">
        <div className="p-3 space-y-1 text-xs">
          {!index ? (
            <Spinner />
          ) : (
            (index.sops ?? index.sop_ids ?? []).map((id) => (
              <button
                key={id}
                onClick={() => setPicked(id)}
                className={clsx(
                  "w-full text-left px-3 py-2 rounded-md border",
                  picked === id
                    ? "border-nv-green/40 bg-nv-green/10 text-nv-green"
                    : "border-transparent hover:bg-[rgb(var(--line))]",
                )}
              >
                <div className="font-mono">{id}</div>
                {index.sections_per_sop?.[id] && (
                  <div className="text-[10px] text-muted mt-0.5">
                    {index.sections_per_sop[id].slice(0, 2).join(" · ")}
                    {index.sections_per_sop[id].length > 2 ? "…" : ""}
                  </div>
                )}
              </button>
            ))
          )}
        </div>
      </Panel>
      <Panel
        title={picked || "Pick a playbook"}
        subtitle="Investigation steps, escalation criteria, filing decision"
        className="lg:col-span-2"
      >
        <div className="p-4 sop-md text-sm overflow-auto max-h-[70vh]">
          {!picked ? (
            <p className="text-xs text-muted">
              Select a playbook on the left.
            </p>
          ) : !sop ? (
            <Spinner />
          ) : sop.body_markdown ? (
            <ReactMarkdown>{sop.body_markdown}</ReactMarkdown>
          ) : sop.sections ? (
            <div className="space-y-3">
              {sop.sections.map((s, i) => (
                <section key={i}>
                  <h2 className="text-sm font-semibold mt-3 mb-1.5 text-nv-green">
                    {s.title}
                  </h2>
                  <pre className="text-xs whitespace-pre-wrap font-sans leading-relaxed">
                    {s.text}
                  </pre>
                </section>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted">No body returned.</p>
          )}
        </div>
      </Panel>
    </div>
  );
}

/* -------------------- Sanctions screen -------------------- */

function SanctionsTab() {
  const [name, setName] = useState("ACME Trading LLC");
  const [country, setCountry] = useState("");
  const [minScore, setMinScore] = useState(0.55);
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<
    | {
        name: string;
        n_hits: number;
        hits: { name: string; list: string; match_score: number }[];
      }
    | null
  >(null);
  const [err, setErr] = useState<string | null>(null);

  async function go() {
    setBusy(true);
    setErr(null);
    setOut(null);
    try {
      const r = await screenSanctions({
        name,
        country: country || undefined,
        min_score: minScore,
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
        title="OFAC + PEP fuzzy screen"
        subtitle="POST /api/sanctions/screen — RapidFuzz token_set_ratio over the two lists"
      >
        <div className="p-4 space-y-3">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
              Name
            </div>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full text-sm border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
                Country (optional)
              </div>
              <input
                value={country}
                onChange={(e) => setCountry(e.target.value)}
                placeholder="CY, RU, …"
                className="w-full text-sm border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
              />
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-muted mb-1">
                Min match score
              </div>
              <input
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={minScore}
                onChange={(e) => setMinScore(Number(e.target.value))}
                className="w-full text-sm border divider rounded-md surface-muted p-2 focus:outline-none focus:border-nv-green/60"
              />
            </div>
          </div>
          <button onClick={go} disabled={busy || !name} className="btn btn-primary">
            <Play size={14} /> {busy ? "Screening…" : "Screen"}
          </button>
          {err && <div className="text-xs text-rose-500">{err}</div>}
        </div>
      </Panel>

      <Panel title="Hits" subtitle={out ? `${out.n_hits} match${out.n_hits === 1 ? "" : "es"}` : ""}>
        <div className="p-4">
          {!out ? (
            <p className="text-xs text-muted">Screen a name to see hits.</p>
          ) : out.hits.length === 0 ? (
            <EmptyState title="No hits above threshold" />
          ) : (
            <table className="w-full text-xs">
              <thead className="text-muted">
                <tr className="text-left border-b divider">
                  <th className="py-2 px-3">Match</th>
                  <th className="py-2 px-3">List</th>
                  <th className="py-2 px-3 text-right">Score</th>
                </tr>
              </thead>
              <tbody>
                {out.hits.map((h, i) => (
                  <tr key={i} className="t-row border-b">
                    <td className="py-2 px-3">{h.name}</td>
                    <td className="py-2 px-3">
                      <span className="chip chip-neutral">{h.list}</span>
                    </td>
                    <td
                      className={clsx(
                        "py-2 px-3 text-right font-mono",
                        h.match_score >= 0.85
                          ? "text-rose-500"
                          : h.match_score >= 0.7
                            ? "text-amber-500"
                            : "text-muted",
                      )}
                    >
                      {(h.match_score * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </Panel>
    </div>
  );
}
