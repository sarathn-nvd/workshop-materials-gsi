/**
 * Lightweight API client.
 * - All requests go to /api/* and are proxied by next.config to NAT (port 8010).
 * - NAT wraps custom-function responses in {"value": ...}; we unwrap it transparently.
 */
export type Json = unknown;

async function request<T = Json>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text.slice(0, 200)}`);
  }
  const body = await res.json();
  // NAT envelope: { "value": <actual payload> }. Unwrap when present.
  if (
    body &&
    typeof body === "object" &&
    !Array.isArray(body) &&
    "value" in body &&
    Object.keys(body).length === 1
  ) {
    return (body as { value: T }).value;
  }
  return body as T;
}

export const api = {
  get: <T = Json>(path: string) => request<T>(path),
  post: <T = Json>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      body: JSON.stringify(body ?? {}),
    }),
};

/* ----------------- Typed wrappers ----------------- */
export const health = () => api.get<{ ok: boolean; n_transactions: number; n_entities: number }>("/api/health");

export const listAlerts = (params: {
  status?: string;
  q?: string;
  limit?: number;
  offset?: number;
}) => {
  const qs = new URLSearchParams();
  if (params.status) qs.set("status", params.status);
  if (params.q) qs.set("q", params.q);
  qs.set("limit", String(params.limit ?? 50));
  qs.set("offset", String(params.offset ?? 0));
  return api.get<{
    total: number;
    limit: number;
    offset: number;
    items: AlertRow[];
  }>(`/api/alerts?${qs}`);
};

export const alertStats = () =>
  api.get<{
    total: number;
    by_status: Record<string, number>;
    by_typology: Record<string, number>;
  }>("/api/alerts/stats");

export const getAlert = (id: string) =>
  api.post<{
    alert: AlertRow;
    status: string;
    kyc_snippet: Record<string, string> | null;
    trace: Trace | null;
    disposition: Disposition | null;
  }>(`/api/alerts/get`, { alert_id: id });

export const postDisposition = (
  id: string,
  body: { verdict: string; note: string },
) =>
  api.post<{ ok: boolean; path: string }>(`/api/alerts/${id}/disposition`, {
    alert_id: id,
    ...body,
  });

export const getTrace = (caseId: string) =>
  api.post<Trace>(`/api/investigation/get`, { case_id: caseId });

export const runInvestigation = (caseId: string) =>
  api.post<Trace>("/api/investigation/run", { case_id: caseId });

export const listEntities = (params: {
  risk_rating?: string;
  entity_type?: string;
  jurisdiction?: string;
  q?: string;
  limit?: number;
  offset?: number;
}) => {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => v != null && qs.set(k, String(v)));
  qs.set("limit", String(params.limit ?? 50));
  qs.set("offset", String(params.offset ?? 0));
  return api.get<{
    total: number;
    items: KYC[];
  }>(`/api/entities?${qs}`);
};

export const getEntity = (id: string) =>
  api.post<{
    kyc: KYC;
    n_tx_total: number;
    n_unique_counterparties: number;
    channel_mix: Record<string, number>;
    n_related_alerts: number;
    related_alerts: AlertRow[];
  }>(`/api/entities/get`, { entity_id: id });

export const getEntityTx = (id: string, limit = 200) =>
  api.post<{ total: number; items: Tx[] }>(`/api/entities/transactions`, {
    entity_id: id,
    limit,
  });

export const getEntityBehavioral = (id: string) =>
  api.post<{
    entity_id: string;
    n_transactions: number;
    metrics: BehavioralMetrics;
  }>(`/api/entities/behavioral_summary`, { entity_id: id });

export const getEntityRisk = (id: string) =>
  api.post<{
    entity_id: string;
    score: number;
    components: Record<string, unknown>;
  }>(`/api/entities/risk_score`, { entity_id: id });

export const getEntityNetwork = (id: string, depth = 2) =>
  api.post<EntityNetwork>(`/api/entities/network`, { entity_id: id, depth });

export const getEntityTimeline = (id: string) =>
  api.post<{
    entity_id: string;
    daily: { date: string; n_tx: number; total_usd: number }[];
  }>(`/api/entities/timeline`, { entity_id: id });

/* analytics */
export const analyticsOverview = () =>
  api.get<{
    n_alerts_total: number;
    n_alerts_open: number;
    n_alerts_in_progress: number;
    n_alerts_closed: number;
    n_entities: number;
    n_transactions: number;
    n_sars_drafted: number;
    avg_case_latency_ms: number;
  }>("/api/analytics/overview");

export const analyticsTypology = () =>
  api.get<{
    seeded: Record<string, number>;
    from_traces: Record<string, number>;
  }>("/api/analytics/typology_distribution");

export const analyticsRiskHeatmap = () =>
  api.get<{
    alerts_by_jurisdiction: Record<string, number>;
    alerts_by_risk_rating: Record<string, number>;
  }>("/api/analytics/risk_heatmap");

export const analyticsTimeline = () =>
  api.get<{ daily_tx_count: { date: string; n_tx: number }[] }>(
    "/api/analytics/timeline",
  );

export const analyticsChannelMix = () =>
  api.get<{ by_typology: Record<string, Record<string, number>> }>(
    "/api/analytics/channel_mix",
  );

export const analyticsTopCp = () =>
  api.get<{
    top_by_volume: {
      counterparty: string;
      n_tx: number;
      total_usd: number;
    }[];
  }>("/api/analytics/top_counterparties");

export const analyticsAuxUsage = () =>
  api.get<{
    n_cases: number;
    used: Record<string, number>;
    dropped: Record<string, number>;
  }>("/api/analytics/aux_usage");

export const analyticsAgentPerf = () =>
  api.get<{
    per_typology: Record<
      string,
      {
        tp: number;
        fp: number;
        fn: number;
        tn: number;
        recall: number;
        precision: number;
      }
    >;
    n_traces: number;
  }>("/api/analytics/agent_performance");

/* eval / compare */
export const evalRuns = () =>
  api.get<{
    n_runs: number;
    items: { name: string; n_traces: number; is_active: boolean }[];
  }>("/api/demo/eval/runs");

export const evalCompare = (body: {
  run_a: string;
  run_b: string;
  label_a?: string;
  label_b?: string;
  token?: string;
}) => api.post<EvalCompareResp>("/api/demo/eval/compare", body);

/* network */
export const globalNetwork = () =>
  api.get<{
    n_nodes: number;
    n_edges: number;
    top_hubs: { id: string; pagerank: number }[];
  }>("/api/network/global");

export const networkPatterns = () =>
  api.get<{
    n_cycles: number;
    cycles: { length: number; nodes: string[] }[];
  }>("/api/network/patterns");

/* skills */
export const skillBehavioral = (b: { passage: string; question?: string }) =>
  api.post<BehavioralFinding | SkillError>("/api/skills/behavioral", {
    question: "",
    ...b,
  });
export const skillNumeric = (b: { passage: string; question?: string }) =>
  api.post<NumericFinding | SkillError>("/api/skills/numeric", {
    question: "",
    ...b,
  });
export const skillCitation = (b: { passage: string; question?: string }) =>
  api.post<CitationFinding | SkillError>("/api/skills/citation", {
    question: "",
    ...b,
  });
export const skillStatutory = (b: {
  statute: string;
  fact_pattern: string;
  question?: string;
}) =>
  api.post<StatutoryFinding | SkillError>("/api/skills/statutory", {
    question: "",
    ...b,
  });
export const skillSar = (body: unknown) =>
  api.post<SarOutput | SkillError>("/api/skills/sar", body);

/* policy / sops / sanctions */
export const searchPolicy = (b: { typology: string; q?: string; k?: number }) =>
  api.post<{
    typology: string;
    n: number;
    items: {
      source: string;
      section: string;
      url: string;
      text: string;
      match_offset?: number;
    }[];
  }>("/api/policy/search", b);

export const policySources = () =>
  api.get<{ sources: Record<string, number>; n_chunks?: number }>(
    "/api/policy/sources",
  );

export const listSops = () =>
  api.get<{
    // Newer backends return `{sops: string[]}`; older return `{sop_ids, sections_per_sop}`.
    sops?: string[];
    sop_ids?: string[];
    sections_per_sop?: Record<string, string[]>;
  }>("/api/sops");

export const getSop = (id: string) =>
  api.post<{
    sop_id: string;
    body_markdown?: string;
    sections?: { title: string; text: string }[];
  }>(`/api/sops/get`, { sop_id: id });

export interface SanctionsHit {
  name: string;
  list: string;
  match_score: number;
}

export const screenSanctions = async (b: {
  name: string;
  country?: string;
  min_score?: number;
}): Promise<{ name: string; n_hits: number; hits: SanctionsHit[] }> => {
  // Backend returns {name, n, items} on current NAT; older spec said {name, n_hits, hits}.
  // Normalise client-side so the UI is unaffected by the rename.
  const r = await api.post<{
    name: string;
    n?: number;
    n_hits?: number;
    items?: SanctionsHit[];
    hits?: SanctionsHit[];
  }>("/api/sanctions/screen", b);
  const hits = r.hits ?? r.items ?? [];
  return {
    name: r.name,
    n_hits: r.n_hits ?? r.n ?? hits.length,
    hits,
  };
};

/* network */
export const networkPath = (b: { source: string; target: string }) =>
  api.post<
    | { found: true; length: number; path: string[] }
    | { found: false; reason: string }
  >("/api/network/path", b);

/* model comparison (pre-compiled report) */
export const modelComparison = (report = "latest", token?: string) =>
  api.post<ModelComparisonReport | { error: string }>(
    "/api/demo/eval/model_comparison",
    { report, token },
  );

export interface ModelComparisonReport {
  report_file: string;
  ran_at: string;
  demo_size?: number;
  demo_version?: string;
  notes?: string;
  endpoints?: { label: string; n_total_keys?: number }[];
  headline_metrics: {
    metric: string;
    higher_is_better: boolean;
    values: Record<string, number>;
    winner_label: string;
  }[];
  confusion: {
    label: string;
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  }[];
  per_typology_recall?: {
    typology: string;
    values: Record<string, number>;
    winner_label: string;
  }[];
  wall_clock_ms?: { field: string; values: Record<string, number> }[];
}

/* ----------------- shared types ----------------- */
export interface AlertRow {
  case_id: string;
  alert_id: string;
  entity_id: string;
  investigation_window_start: string;
  investigation_window_end: string;
  trigger_summary: string;
  status?: string;
}

export interface KYC {
  entity_id: string;
  entity_type?: string;
  expected_monthly_volume?: number;
  business_purpose?: string;
  risk_rating?: string;
  incorporation_jurisdiction?: string;
}

export interface Tx {
  transaction_id?: string;
  date: string;
  amount: number;
  currency: string;
  counterparty: string;
  channel: string;
  notes?: string;
  entity_id?: string;
}

export interface BehavioralMetrics {
  tx_count: number;
  tx_total_usd: number;
  channel_mix: Record<string, number>;
  velocity_24h_max: number;
  velocity_24h_avg_30d: number;
  unique_counterparties_7d: number;
  amount_z_score_max: number;
  country_risk_max: number;
  loop_detected: boolean;
  vs_declared_volume_ratio: number;
}

export interface Disposition {
  case_id: string;
  alert_id: string;
  verdict: string;
  note: string;
  ts: number;
}

export interface GateDecision {
  task: string;
  used: boolean;
  reason: string;
  reviewer_verdict?: string;
  reviewer_explain?: string;
}

export interface Trace {
  case_id: string;
  alert_id: string;
  entity_id: string;
  started_at: string;
  finished_at: string;
  wall_clock_ms: number;
  transactions: Tx[];
  kyc_profile: KYC | null;
  sanctions_pep_hits: { name: string; list: string; match_score: number }[];
  policy_excerpts: { source: string; section: string; url: string; text: string }[];
  sop_excerpts: { sop_id: string; section: string; text: string }[];
  semantic_profile: Record<string, unknown> | null;
  typology_hypothesis: string;
  activity_descriptor: string;
  orchestrator_calls: { tool: string; args: Json; result_summary: string }[];
  aux_responses_raw: Record<string, Json>;
  aux_gate_decisions: GateDecision[];
  auxiliary_findings: {
    behavioral?: BehavioralFinding[];
    numeric?: NumericFinding[];
    citation?: CitationFinding[];
    statutory?: StatutoryFinding[];
  };
  sar_user_message: string;
  sar_raw_text: string;
  sar_output: SarOutput | null;
  sar_parse_error: string | null;
  sar_is_suspicious: boolean;
  sar_narrative: string;
  judge_enabled: boolean;
  error: string | null;
}

export interface BehavioralFinding {
  question: string;
  summary: string;
  metrics: BehavioralMetrics;
  evidence: string;
}
export interface NumericFinding {
  question: string;
  answer: string;
  calculation: string;
  evidence: string;
}
export interface CitationFinding {
  question: string;
  answer: string;
  evidence_span: string;
}
export interface StatutoryFinding {
  question: string;
  answer: string;
  label: "entailment" | "contradiction" | "neutral";
  reasoning: string;
}
export interface SarOutput {
  is_suspicious: boolean;
  suspicious_activity_report: string;
}

export interface SkillError {
  error: string;
  raw?: string;
  detail?: string;
}

export interface EntityNetwork {
  center: string;
  depth: number;
  n_nodes: number;
  n_edges: number;
  nodes: {
    id: string;
    type: "entity" | "counterparty";
    in_degree: number;
    out_degree: number;
  }[];
  edges: { source: string; target: string; n_tx: number; total_usd: number }[];
}

export interface EvalCompareResp {
  ground_truth: { n_total: number; n_sar: number; n_no_sar: number; n_near_miss: number };
  run_a: RunScore;
  run_b: RunScore;
  diff: {
    metrics: Record<string, { absolute: number; relative_pct: number }>;
    confusion: { tp: number; fp: number; tn: number; fn: number };
    fp_breakdown: { near_miss: number; clean: number };
    latency_ms: number;
  };
}

export interface RunScore {
  name: string;
  label: string;
  n_predictions: number;
  n_missing: number;
  parse_errors: number;
  confusion: { tp: number; fp: number; tn: number; fn: number };
  fp_breakdown: { near_miss: number; clean: number };
  metrics: Record<string, number>;
  latency: { avg_case_ms: number; n_with_timing: number };
}
