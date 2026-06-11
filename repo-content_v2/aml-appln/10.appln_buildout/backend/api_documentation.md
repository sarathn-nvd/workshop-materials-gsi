# AML Investigation Backend — API Reference

This is the complete API surface served by `nat serve --config_file src/configs/workflow.yaml`. The backend exposes **43 endpoints** total:

- **1 root workflow endpoint** (`POST /api/investigation/run`) → drives the deterministic 7-phase `investigate_case` workflow against one alert.
- **42 custom endpoints** under `/api/*` for tools, entity views, network graph, analytics, skill playgrounds, alert queue, eval, model comparison, and system health.

## Conventions

| Item | Convention |
|---|---|
| Base URL | `http://${NAT_AML_HOST}:${NAT_AML_PORT}` (defaults `0.0.0.0:8000`) |
| Body format | All POST requests take a single JSON object; all GETs accept query params named identically to the body fields below. |
| Path params | Written as `{name}` in the route. |
| Authentication | Workshop demo has no global auth. The three `/api/demo/eval*` routes are gated by a bearer-style `token` field that must match `NAT_AML_EVAL_TOKEN` from the env when that env var is set. |
| Internal column stripping | Every route that surfaces transactions or KYC strips the ground-truth sidecars (`source_pool`, `typology_tag`, `_archetype`) at the response boundary. |
| Errors | Returned as `{"error": "<message>"}` with HTTP 200 in most handlers (NAT idiom). Token-gate failures return `{"error": "unauthorized"}`. |
| Auto-generated docs | OpenAPI / Swagger UI is available at `/docs` once the server is running. |

## Section index

1. [Core investigation](#1-core-investigation)
2. [Alert queue & case management](#2-alert-queue--case-management)
3. [Entity 360](#3-entity-360)
4. [Network / graph analysis](#4-network--graph-analysis)
5. [Skill playgrounds (interactive)](#5-skill-playgrounds-interactive)
6. [Policy / SOP / sanctions tooling](#6-policy--sop--sanctions-tooling)
7. [Analytics dashboard](#7-analytics-dashboard)
8. [Evaluation against ground truth (gated)](#8-evaluation-against-ground-truth-gated)
9. [Demo orchestration](#9-demo-orchestration)
10. [System & health](#10-system--health)

---

## 1. Core investigation

The agent's primary job. One alert in → one persisted `CaseTrace` out.

### `POST /api/investigation/run`

End-to-end investigation. A deterministic 7-phase workflow walks the case through data-fetch tools (Phase 1), an internal typology guess (Phase 2, routing-only — never sent to the LLM), retrieval tools (Phase 3), aux skills (Phase 4 — three LLM calls plus a Python-computed behavioral block), the aux gate (Phase 5), the final SAR call with the **7-key bundle** (Phase 6), and trace persistence (Phase 7). See `backend.md` §4 for the full diagram.

**Request body**

```json
{
  "case_id": "DEMO_0042"
}
```

OR a full alert payload:

```json
{
  "alert_id": "ALT_12345",
  "entity_id": "SYN_22da3c8f",
  "investigation_window_start": "2026-02-01",
  "investigation_window_end": "2026-04-30",
  "trigger_summary": "Wire velocity flagged by monitoring rule"
}
```

**Response** — same shape as the persisted trace (see `GET /api/investigation/{case_id}` below).

The trace JSON is also written to `./data/traces/{case_id}.json` for later inspection.

---

### `GET /api/investigation/{case_id}`

Backing function: `get_trace`. Retrieve the persisted `CaseTrace` for a previously-run case.

**Path params** — `case_id` (string, e.g. `DEMO_0042`).

**Response (200 — trace found)**

The trace records both (a) the 7-key bundle that was sent to the SAR LLM (`sar_user_message`), and (b) the internal-only routing artifacts (`semantic_profile`, `typology_hypothesis`, `activity_descriptor`) that drove Phase 2/3 retrieval. The routing artifacts are persisted for the cockpit UI to display — they are **never** part of the SAR user message.

```json
{
  "case_id": "DEMO_0042",
  "alert_id": "ALT_2461263",
  "entity_id": "SYN_22da3c8f",
  "started_at": "2026-05-23T07:31:12.418Z",
  "finished_at": "2026-05-23T07:31:15.840Z",
  "wall_clock_ms": 3421.2,

  "transactions": [ {"date": "...", "amount": ..., "currency": "...",
                     "counterparty": "...", "channel": "...", "notes": "..."}, ... ],
  "kyc_profile": { "entity_id": "...", "entity_type": "individual|business",
                   "expected_monthly_volume": 50000.0,
                   "business_purpose": "...", "risk_rating": "low|medium|high|enhanced|prohibited",
                   "incorporation_jurisdiction": "..." },
  "sanctions_pep_hits": [ {"name": "...", "list": "OFAC|OpenSanctions|EU|UN",
                            "match_score": 0.78} ],
  "policy_excerpts": [ {"source": "FFIEC|FATF|FinCEN|OFAC",
                        "section": "...", "url": "...", "text": "..."} ],
  "sop_excerpts": [ {"sop_id": "SOP-STRUCTURING-01", "section": "...", "text": "..."} ],

  "semantic_profile": { "channel_mix": {...}, "cash_present": false,
                         "declared_volume_band": "...", "geo_risk": "..." },
  "typology_hypothesis": "structuring",
  "activity_descriptor": "6 sub-$10K cash deposits across 4 branches in 8 days",

  "orchestrator_calls": [
    { "tool": "aml_data_tools.get_transactions", "args": {...}, "result_summary": "..." },
    ...
  ],

  "aux_responses_raw": { "behavioral": {...}, "numeric": {...},
                          "citation": {...}, "statutory": {...} },
  "aux_gate_decisions": [
    { "task": "behavioral", "used": true, "reason": "deterministic_python",
      "reviewer_verdict": null, "reviewer_explain": null },
    { "task": "numeric",    "used": true, "reason": "schema_ok" },
    ...
  ],
  "auxiliary_findings": { "behavioral": [{...}], "numeric": [{...}],
                           "citation": [{...}], "statutory": [{...}] },

  "sar_user_message": "<the exact 7-key JSON sent to the trained model>",
  "sar_raw_text":     "<the model's raw output>",
  "sar_output":       { "is_suspicious": true, "suspicious_activity_report": "..." },
  "sar_parse_error":  null,
  "sar_is_suspicious": true,
  "sar_narrative":     "Suspicious activity is identified for ...",

  "judge_enabled": true,
  "error": null
}
```

**Response (200 — trace missing)** — `{"error": "trace not found: <case_id>"}`.

---

## 2. Alert queue & case management

### `GET /api/alerts`

Backing function: `list_alerts`. List alerts from `./data/demo/manifest.jsonl` with filters and pagination. Status is derived from whether a trace / disposition exists.

**Query params**

| Field | Type | Default | Notes |
|---|---|---|---|
| `status` | `"open"` \| `"in_progress"` \| `"closed"` | — | `open` = no trace, `in_progress` = trace exists, `closed` = disposition exists. |
| `typology_hypothesis` | string | — | Filter by the trace's predicted typology. Requires a trace to exist. |
| `q` | string | — | Case-insensitive substring match against `trigger_summary`, `entity_id`, `alert_id`. |
| `limit` | int | 50 | 1–500 |
| `offset` | int | 0 | |

**Response**

```json
{
  "total": 194,
  "limit": 50,
  "offset": 0,
  "items": [
    {
      "case_id": "DEMO_0001",
      "alert_id": "ALT_2461263",
      "entity_id": "SYN_22da3c8f",
      "investigation_window_start": "2026-02-01",
      "investigation_window_end": "2026-04-30",
      "trigger_summary": "Wire velocity flagged by monitoring rule",
      "status": "open"
    },
    ...
  ]
}
```

---

### `GET /api/alerts/{alert_id}`

Backing function: `get_alert`. One alert + KYC snippet + latest trace + disposition (if any).

**Path params** — `alert_id` (accepts either the `alert_id` OR the `case_id`).

**Response**

```json
{
  "alert": { "case_id": "...", "alert_id": "...", "entity_id": "...",
             "investigation_window_start": "...", "investigation_window_end": "...",
             "trigger_summary": "..." },
  "status": "open|in_progress|closed",
  "kyc_snippet": { "entity_id": "...", "entity_type": "...",
                    "business_purpose": "...", "risk_rating": "...",
                    "incorporation_jurisdiction": "..." },
  "trace": { ... full CaseTrace ... } | null,
  "disposition": { "case_id": "...", "verdict": "file_sar|dismiss|escalate",
                    "note": "...", "ts": 1779522210.4 } | null
}
```

---

### `POST /api/alerts/{alert_id}/disposition`

Backing function: `post_disposition`. Record an analyst verdict. Persists to `./data/dispositions/{case_id}.json`.

**Request body**

```json
{
  "alert_id": "ALT_2461263",
  "verdict": "file_sar | dismiss | escalate",
  "note": "Free-text rationale for the verdict."
}
```

**Response**

```json
{ "ok": true, "path": "./data/dispositions/DEMO_0001.json" }
```

---

### `GET /api/alerts/stats`

Backing function: `alerts_stats`. Summary counts for the queue.

**Response**

```json
{
  "total": 194,
  "by_status":    { "open": 165, "in_progress": 20, "closed": 9 },
  "by_typology":  { "structuring": 9, "layering": 10, "none": 140, "unknown": 174, ... }
}
```

---

## 3. Entity 360

### `GET /api/entities`

Backing function: `list_entities`. Search KYC entities.

**Query params**

| Field | Type | Notes |
|---|---|---|
| `risk_rating` | `low|medium|high|enhanced|prohibited` | |
| `entity_type` | `individual|business` | |
| `jurisdiction` | string | e.g. `US-NY`, `BVI`, `Cayman` |
| `q` | string | Case-insensitive substring match against `entity_id` and `business_purpose`. |
| `limit` | int (default 50, max 500) | |
| `offset` | int | |

**Response**

```json
{
  "total": 2072, "limit": 50, "offset": 0,
  "items": [
    { "entity_id": "...", "entity_type": "business",
      "expected_monthly_volume": 35486.0,
      "business_purpose": "...", "risk_rating": "low",
      "incorporation_jurisdiction": "US-CA" },
    ...
  ]
}
```

Internal sidecars (`source_pool`, `_archetype`) are stripped.

---

### `GET /api/entities/{entity_id}`

Backing function: `get_entity`. Full profile + activity counts.

**Response**

```json
{
  "kyc": { ...same shape as list items above... },
  "n_tx_total": 27,
  "n_unique_counterparties": 14,
  "channel_mix": { "wire": 12, "ach": 8, "cash": 5, "card": 2 },
  "n_related_alerts": 1,
  "related_alerts": [ { "case_id": "...", ... } ]
}
```

---

### `GET /api/entities/{entity_id}/transactions`

Backing function: `get_entity_tx`. Paginated tx history with optional window filter.

**Query params** — `window_start`, `window_end` (YYYY-MM-DD), `limit` (default 200, max 2000), `offset`.

**Response**

```json
{
  "total": 27, "limit": 200, "offset": 0,
  "items": [
    { "transaction_id": "TX_...", "date": "2026-02-04", "amount": 6255.32,
      "currency": "USD", "counterparty": "TRX Holdings LLC", "channel": "ach",
      "notes": "", "entity_id": "..." },
    ...
  ]
}
```

Internal sidecars stripped.

---

### `GET /api/entities/{entity_id}/behavioral_summary`

Backing function: `get_entity_behavioral`. **Pure-Python** deterministic behavioral metrics over the entity's full transaction history (no LLM). Same `BehavioralMetrics` schema the trained model's `auxiliary_behavioral` task produces.

**Response**

```json
{
  "entity_id": "SYN_22da3c8f",
  "n_transactions": 27,
  "metrics": {
    "tx_count": 27,
    "tx_total_usd": 145890.42,
    "channel_mix": { "ach": 0.44, "wire": 0.33, "card": 0.15, "cash": 0.08 },
    "velocity_24h_max": 4,
    "velocity_24h_avg_30d": 0.9,
    "unique_counterparties_7d": 5,
    "amount_z_score_max": 2.3,
    "country_risk_max": 0.1,
    "loop_detected": false,
    "vs_declared_volume_ratio": 1.47
  }
}
```

---

### `GET /api/entities/{entity_id}/risk_score`

Backing function: `get_entity_risk`. Hand-tuned weighted blend of (KYC risk rating, country risk, volume ratio, amount z-score, sanctions hits).

**Response**

```json
{
  "entity_id": "SYN_22da3c8f",
  "score": 8,
  "components": {
    "kyc_risk_rating": "low",
    "country_risk_max": 0.1,
    "vs_declared_volume_ratio": 1.47,
    "amount_z_score_max": 2.3
  }
}
```

`score` is in 0–100.

---

### `GET /api/entities/{entity_id}/network`

Backing function: `get_entity_network`. N-hop counterparty graph centred on the entity. NetworkX on the backend; frontend renders force-directed.

**Query params** — `depth` (default 2, 1–3), `window_start`, `window_end`.

**Response**

```json
{
  "center": "SYN_22da3c8f",
  "depth": 2,
  "n_nodes": 28,
  "n_edges": 27,
  "nodes": [
    { "id": "SYN_22da3c8f", "type": "entity",       "in_degree": 0, "out_degree": 14 },
    { "id": "TRX Holdings LLC", "type": "counterparty", "in_degree": 1, "out_degree": 0 },
    ...
  ],
  "edges": [
    { "source": "SYN_22da3c8f", "target": "TRX Holdings LLC",
      "n_tx": 3, "total_usd": 18765.96 },
    ...
  ]
}
```

---

### `GET /api/entities/{entity_id}/timeline`

Backing function: `get_entity_timeline`. Daily transaction volume.

**Response**

```json
{
  "entity_id": "SYN_22da3c8f",
  "daily": [
    { "date": "2026-02-04", "n_tx": 1, "total_usd": 6255.32 },
    { "date": "2026-02-12", "n_tx": 2, "total_usd": 8410.7 },
    ...
  ]
}
```

---

## 4. Network / graph analysis

### `GET /api/network/global`

Backing function: `get_global_network`. Top-line counterparty graph stats across the whole corpus.

**Response**

```json
{
  "n_nodes": 4321,
  "n_edges": 5678,
  "top_hubs": [
    { "id": "TRX Holdings LLC", "pagerank": 0.0142 },
    ...
  ]
}
```

---

### `GET /api/network/patterns`

Backing function: `get_network_patterns`. Pre-computed simple cycles (length 2–6) in the global transaction graph.

**Response**

```json
{
  "n_cycles": 12,
  "cycles": [
    { "length": 3, "nodes": ["A", "B", "C"] },
    ...
  ]
}
```

---

### `POST /api/network/path`

Backing function: `get_network_path`. Shortest directed path between two entities/counterparties.

**Request body**

```json
{ "source": "SYN_22da3c8f", "target": "Some Counterparty" }
```

**Response**

```json
{ "found": true, "length": 3, "path": ["SYN_22da3c8f", "X", "Some Counterparty"] }
```

Or `{ "found": false, "reason": "no path" | "<node missing>" }`.

---

## 5. Skill playgrounds (interactive)

These five endpoints let the workshop attendee invoke a single auxiliary skill or the final SAR call directly against the trained model — no orchestrator, no agent loop. They reuse the **exact** registered functions the workflow uses (`aux_*_call`, `sar_judgment_caller`), so prompts / schema / parsing are identical.

### `POST /api/skills/behavioral`

Backing function: `skill_behavioral` → wraps `aux_behavioral_call`. Sends a passage to the trained model with `task_type=auxiliary_behavioral`.

**Request body**

```json
{
  "passage": "## KYC profile\nentity_id: X\n...\n\n## Transactions\ndate,amount,...",
  "question": ""
}
```

**Response** — a validated `BehavioralFinding`:

```json
{
  "question": "",
  "summary": "On 2026-04-01 entity ... executed 8 wires totalling 75,314 USD ...",
  "metrics": {
    "tx_count": 8, "tx_total_usd": 75314.0,
    "channel_mix": {"wire": 1.0},
    "velocity_24h_max": 8, "velocity_24h_avg_30d": 1.5,
    "unique_counterparties_7d": 1, "amount_z_score_max": 2.3,
    "country_risk_max": 0.0, "loop_detected": false,
    "vs_declared_volume_ratio": 1.29
  },
  "evidence": "transactions[0..7]; kyc_profile.expected_monthly_volume"
}
```

Errors: `{"error": "no_json", "raw": "..."}` if the model didn't return JSON; `{"error": "schema", "detail": "..."}` if validation failed.

---

### `POST /api/skills/numeric`

Backing function: `skill_numeric` → `aux_numeric_call`. Same shape as behavioral.

**Request body**

```json
{
  "passage": "## Source data ...",
  "question": "Sum the cash deposits in April and compare to declared monthly volume."
}
```

**Response** — `NumericFinding`:

```json
{
  "question": "...",
  "answer": "$57,300 over 8 days, 4.3× declared",
  "calculation": "1. Sum cash deposits Apr 1–8: ... = 57,300\n2. 8-day rate × (30/8) = 214,875 ...",
  "evidence": "transactions[0..5], kyc_profile.expected_monthly_volume"
}
```

---

### `POST /api/skills/citation`

Backing function: `skill_citation` → `aux_citation_call`.

**Request body**

```json
{
  "passage": "## Policy excerpt\nsource: FinCEN\nsection: Advisory FIN-2014-A005\n...",
  "question": "What does this advisory say about structuring?"
}
```

**Response** — `CitationFinding`:

```json
{
  "question": "...",
  "answer": "FinCEN Advisory FIN-2014-A005 governs sub-threshold cash deposit patterns by cash-intensive businesses.",
  "evidence_span": "Multiple cash deposits structured below the $10,000 ... may constitute structuring under 31 U.S.C. § 5324."
}
```

---

### `POST /api/skills/statutory`

Backing function: `skill_statutory` → `aux_statutory_call`.

**Request body**

```json
{
  "passage": "## Statute\n31 U.S.C. § 5324 ...\n\n## Fact pattern\nEntity made 6 sub-$10K ...",
  "question": "Does this conduct fall within 31 U.S.C. § 5324(a)(3)?"
}
```

**Response** — `StatutoryFinding`:

```json
{
  "question": "...",
  "answer": "Yes, the conduct described falls within 31 U.S.C. § 5324(a)(3).",
  "label": "entailment | contradiction | neutral",
  "reasoning": "31 U.S.C. § 5324(a)(3) prohibits structuring transactions with a domestic financial institution ..."
}
```

---

### `POST /api/skills/sar`

Backing function: `skill_sar` → `sar_judgment_caller`. Full hand-built **7-key bundle** → trained model → typed SAR.

**Request body** — exactly the 7 keys; the input schema uses Pydantic `extra="forbid"`, so any additional field is rejected.

```json
{
  "transactions":       [ { "date": "...", "amount": ..., "currency": "...",
                            "counterparty": "...", "channel": "wire|ach|cash|card|cheque|crypto",
                            "notes": "" }, ... ],
  "kyc_profile":        { "entity_id": "...", "entity_type": "individual|business",
                          "expected_monthly_volume": 50000.0, "business_purpose": "...",
                          "risk_rating": "low|medium|high|enhanced|prohibited",
                          "incorporation_jurisdiction": "..." },
  "sanctions_pep_hits": [ { "name": "...", "list": "OFAC|OpenSanctions|EU|UN",
                            "match_score": 0.78 } ],
  "policy_excerpts":    [ { "source": "FFIEC|FATF|FinCEN|OFAC", "section": "...",
                            "url": "...", "text": "..." } ],
  "sop_excerpts":       [ { "sop_id": "SOP-STRUCTURING-01", "section": "...", "text": "..." } ],
  "auxiliary_findings": { "behavioral": [...], "numeric": [...],
                          "citation": [...], "statutory": [...] } | null
}
```

> **Note.** No `regulatory_frame`, `typology_inferred`, or `decision_target` field is accepted — the SFT contract is evidence-only. The trained model derives the verdict from the evidence keys alone. The model's response is exactly two fields.

**Response**

```json
{
  "is_suspicious": true,
  "suspicious_activity_report": "Suspicious activity is identified for entity ...",
  "raw_text":     "<model's raw output>",
  "parse_error":  null,
  "user_message": "<exact 7-key JSON that was sent to the trained model>"
}
```

---

## 6. Policy / SOP / sanctions tooling

### `POST /api/policy/search`

Backing function: `search_policy`. Stratified top-k retrieval over the policy corpus, filtered by typology. If `q` is provided, the `match_offset` of the first hit per excerpt is included.

**Request body**

```json
{
  "typology": "structuring",
  "q": "$10,000 threshold",
  "k": 4
}
```

**Response**

```json
{
  "typology": "structuring",
  "n": 4,
  "items": [
    { "source": "FinCEN", "section": "Advisory FIN-2014-A005", "url": "...",
      "text": "Multiple cash deposits structured below the $10,000 ...",
      "match_offset": 134 },
    ...
  ]
}
```

---

### `GET /api/policy/sources`

Backing function: `list_policy_sources`. Distribution of the policy corpus by source.

**Response**

```json
{
  "sources": { "FinCEN": 23794, "OFAC": 16157, "FFIEC": 2357, "FATF": 248 },
  "n_chunks": 42556
}
```

---

### `GET /api/sops`

Backing function: `list_sops`. List of all SOPs and their section headings.

**Response**

```json
{
  "sop_ids": [
    "SOP-ELDER-EXPLOITATION-01", "SOP-HUMAN-TRAFFICKING-01",
    "SOP-LAYERING-01", "SOP-SHELL-COMPANY-01",
    "SOP-SMURFING-01", "SOP-STRUCTURING-01",
    "SOP-TERRORIST-FINANCING-01", "SOP-TRADE-BASED-ML-01"
  ],
  "sections_per_sop": {
    "SOP-STRUCTURING-01": [
      "Investigation Steps", "Escalation Criteria",
      "Documentation Requirements", "Filing Decision",
      "Tools and Systems", "References"
    ],
    ...
  }
}
```

---

### `GET /api/sops/{sop_id}`

Backing function: `get_sop_body`. Render the SOP as Markdown.

**Response**

```json
{
  "sop_id": "SOP-STRUCTURING-01",
  "body_markdown": "## Investigation Steps\n\n1. ...\n\n## Escalation Criteria\n\n..."
}
```

---

### `POST /api/sanctions/screen`

Backing function: `screen_name`. Free-form fuzzy screen against OFAC + PEP.

**Request body**

```json
{
  "name": "ACME Trading LLC",
  "country": "CY",
  "min_score": 0.55
}
```

**Response**

```json
{
  "name": "ACME Trading LLC",
  "n_hits": 2,
  "hits": [
    { "name": "ACME TRADING LIMITED", "list": "OFAC",         "match_score": 0.82 },
    { "name": "Acme Trading",         "list": "OpenSanctions","match_score": 0.71 }
  ]
}
```

---

## 7. Analytics dashboard

All endpoints under `/api/analytics/*` are GET, take no body, and aggregate across the entire corpus + persisted traces + dispositions. Built for the frontend dashboard's charts.

### `GET /api/analytics/overview`

Backing function: `analytics_overview`. Top-line cards.

```json
{
  "n_alerts_total": 194,
  "n_alerts_open": 165,
  "n_alerts_in_progress": 20,
  "n_alerts_closed": 9,
  "n_entities": 2072,
  "n_transactions": 71601,
  "n_sars_drafted": 28,
  "avg_case_latency_ms": 3421
}
```

---

### `GET /api/analytics/typology_distribution`

Backing function: `analytics_typology`. Donut data.

```json
{
  "seeded":     { "structuring": 9, "layering": 10, "none": 140, ... },
  "from_traces": { "structuring": 8, "layering": 9,  "none": 142, ... }
}
```

---

### `GET /api/analytics/risk_heatmap`

Backing function: `analytics_risk_heatmap`. Per-jurisdiction + per-risk-rating alert counts.

```json
{
  "alerts_by_jurisdiction": { "US-CA": 67, "US-NY": 38, "BVI": 4, ... },
  "alerts_by_risk_rating":  { "low": 96, "medium": 65, "high": 28, "enhanced": 5 }
}
```

---

### `GET /api/analytics/timeline`

Backing function: `analytics_timeline`. Daily tx counts.

```json
{
  "daily_tx_count": [
    { "date": "2026-02-01", "n_tx": 187 },
    { "date": "2026-02-02", "n_tx": 203 },
    ...
  ]
}
```

---

### `GET /api/analytics/channel_mix`

Backing function: `analytics_channel_mix`. Per-typology channel breakdown (from the internal `typology_tag` column — analytics-only).

```json
{
  "by_typology": {
    "structuring":      { "cash": 50, "wire": 2, "ach": 0, ... },
    "layering":         { "wire": 38, "ach": 4, ... },
    ...
  }
}
```

---

### `GET /api/analytics/top_counterparties`

Backing function: `analytics_top_cp`. Top 20 counterparties by total volume.

```json
{
  "top_by_volume": [
    { "counterparty": "TRX Holdings LLC", "n_tx": 142, "total_usd": 4521098.0 },
    ...
  ]
}
```

---

### `GET /api/analytics/aux_usage`

Backing function: `analytics_aux_usage`. How often each aux finding was USED vs DROPPED across all persisted traces.

```json
{
  "n_cases": 38,
  "used":    { "behavioral": 38, "numeric": 30, "citation": 25, "statutory": 18 },
  "dropped": {                   "numeric": 8,  "citation": 13, "statutory": 20 }
}
```

---

### `GET /api/analytics/agent_performance`

Backing function: `analytics_agent_perf`. Per-typology recall / precision against `eval_keys.jsonl`. **Aggregate-only — no per-case labels leaked.**

```json
{
  "per_typology": {
    "structuring":     { "tp": 8, "fp": 1, "fn": 1, "tn": 184,
                          "recall": 0.889, "precision": 0.889 },
    "layering":        { "tp": 9, "fp": 0, "fn": 1, "tn": 184,
                          "recall": 0.9,   "precision": 1.0   },
    ...
  },
  "n_traces": 194
}
```

---

### `GET /api/analytics/profile`

Backing function: `analytics_profile`. NAT profiler artifact pointer (populated when the workflow was run under `nat eval --profile`).

```json
{
  "n_profiler_files": 4,
  "files": ["./.tmp/nat/profiler/run_001/usage_stats.json", ...],
  "note":  "NAT profiler data is captured when the workflow is run via `nat eval --profile=...`; raw artifacts are listed here."
}
```

---

## 8. Evaluation against ground truth (gated)

These are the **only** three endpoints that touch `./data/demo/eval_keys.jsonl`. The agent never reads it at runtime. All three are gated by an optional `token` field that must match the env var `NAT_AML_EVAL_TOKEN` when that variable is set.

### `POST /api/demo/eval`

Backing function: `demo_eval`. Aggregate scorecard joining `./data/traces/*.json` against ground truth.

**Request body**

```json
{ "token": "<NAT_AML_EVAL_TOKEN>" }
```

**Response**

```json
{
  "n_total_in_ground_truth": 194,
  "n_predictions": 188,
  "n_missing_predictions": 6,

  "confusion": { "tp": 28, "fp": 5, "tn": 153, "fn": 2 },

  "metrics": {
    "accuracy":                       0.9628,
    "precision":                      0.8485,
    "recall":                         0.9333,
    "f1":                             0.8889,
    "macro_f1":                       0.7421,
    "fpr":                            0.0316,
    "false_positive_rate_clean":      0.0214,
    "false_positive_rate_near_miss":  0.125,
    "near_miss_specificity":          0.875,
    "typology_accuracy_on_tp":        0.9286,
    "narrative_grounding_rate":       0.9697
  },

  "per_typology": {
    "structuring":     { "tp": 8, "fp": 1, "tn": 184, "fn": 1,
                         "recall": 0.889, "precision": 0.889, "f1": 0.889 },
    "layering":        { "tp": 9, "fp": 0, "tn": 184, "fn": 1,
                         "recall": 0.9,   "precision": 1.0,   "f1": 0.9474 },
    "none":            { "tp": 0, "fp": 4, "tn": 136, "fn": 0,
                         "recall": null,  "precision": 0.0,   "f1": null },
    ...
  }
}
```

Errors: `{"error": "unauthorized"}` if the token gate fails.

---

### `POST /api/demo/eval/cases`

Backing function: `demo_eval_cases`. Per-case prediction vs ground truth list, filterable.

**Request body**

```json
{
  "token":     "<NAT_AML_EVAL_TOKEN>",
  "outcome":   "TP | FP | TN | FN | no_prediction",
  "correct":   true,
  "typology":  "structuring",
  "limit":     200,
  "offset":    0
}
```

All filter fields are optional.

**Response**

```json
{
  "total": 30,
  "limit": 200,
  "offset": 0,
  "items": [
    {
      "case_id":     "DEMO_0042",
      "entity_id":   "SYN_a3f9e1b2",
      "ground_truth": {
        "label":    true,
        "typology": "structuring",
        "near_miss": false
      },
      "prediction": {
        "label":              true,
        "typology":           "structuring",
        "narrative_excerpt":  "Suspicious activity is identified for entity ..."
      },
      "is_correct":         true,
      "typology_correct":   true,
      "outcome":            "TP",
      "narrative_grounded": true,
      "wall_clock_ms":      3421.2
    },
    ...
  ]
}
```

When `outcome == "no_prediction"` the row carries `prediction: null`, `is_correct: null`.

---

### `POST /api/demo/eval/case/{case_id}`

Backing function: `demo_eval_case`. Deep-dive on a single case — comparison + every tool output + agentic audit + SAR text.

**Path params** — `case_id`.

**Request body**

```json
{
  "case_id": "DEMO_0042",
  "token":   "<NAT_AML_EVAL_TOKEN>",
  "include_full_tool_outputs": true
}
```

(`case_id` may be in the path OR the body; both are accepted.)

**Response**

```json
{
  "comparison": {
    "case_id":      "DEMO_0042",
    "entity_id":    "SYN_a3f9e1b2",
    "ground_truth": { "label": true, "typology": "structuring", "near_miss": false },
    "prediction":   { "label": true, "typology": "structuring",
                      "narrative_excerpt": "Suspicious activity ..." },
    "is_correct":         true,
    "typology_correct":   true,
    "outcome":            "TP",
    "narrative_grounded": true,
    "wall_clock_ms":      3421.2
  },

  "tool_outputs": {
    "transactions":      [ ... ],
    "kyc_profile":       { ... },
    "sanctions_pep_hits": [ ... ],
    "policy_excerpts":   [ ... ],
    "sop_excerpts":      [ ... ],
    "semantic_profile":  { ... },
    "compute_hints": {
      "typology_hypothesis": "structuring",
      "activity_descriptor": "6 sub-$10K cash deposits in 8 days"
    }
  },

  "agentic": {
    "orchestrator_calls": [ { "tool": "...", "args": {...}, "result_summary": "..." }, ... ]
  },

  "aux": {
    "responses_raw":      { "behavioral": {...}, "numeric": {...},
                            "citation": {...}, "statutory": {...} },
    "gate_decisions":     [
      { "task": "behavioral", "used": true,  "reason": "judge: PASS",
        "reviewer_verdict": "PASS",          "reviewer_explain": "..." },
      { "task": "citation",   "used": false, "reason": "judge: ISSUES_FOUND",
        "reviewer_verdict": "ISSUES_FOUND",  "reviewer_explain": "..." },
      ...
    ],
    "auxiliary_findings": { "behavioral": [...], "numeric": [...],
                            "citation":   [...], "statutory": [...] }
  },

  "sar": {
    "user_message":  "<exact 7-key JSON sent to the trained model>",
    "raw_text":      "<model's raw output>",
    "parsed_output": { "is_suspicious": true, "suspicious_activity_report": "..." },
    "parse_error":   null
  },

  "timing": {
    "started_at":     "2026-05-23T07:31:12.418Z",
    "finished_at":    "2026-05-23T07:31:15.840Z",
    "wall_clock_ms":  3421.2
  },

  "error": null
}
```

If `include_full_tool_outputs=false`, `transactions` / `policy_excerpts` / `sop_excerpts` are replaced with their counts (`n_transactions`, `n_policy_excerpts`, `n_sop_excerpts`) and `sar.user_message` / `sar.raw_text` are replaced with `"(omitted in compact mode)"`.

If no trace exists for the case yet, the response is:

```json
{
  "comparison": { "outcome": "no_prediction", "is_correct": null, ... },
  "trace": null,
  "note": "No persisted trace; run /api/investigation/run on this case first."
}
```

If the case_id isn't in `eval_keys.jsonl` at all: `{"error": "case_id not found in ground truth: <case_id>"}`.

---

### `GET /api/demo/eval/runs`

Backing function: `demo_eval_runs`. Lists all trace-snapshot directories under `./data/` that are eligible for scoring or for use in `/api/demo/eval/compare`.

**Discovery convention** — the active write directory is always `./data/traces/`. Any sibling directory whose name starts with `traces_` and contains at least one `*.json` file is treated as an archived snapshot. The backend uses this convention to keep multiple model-run snapshots side-by-side (e.g. one for the custom SFT/RL run, one for a base-model comparison run).

**Request** — none (GET, no query params).

**Response**

```json
{
  "n_runs": 3,
  "items": [
    { "name": "traces",                "n_traces": 194, "is_active": true  },
    { "name": "traces_base_uncapped",  "n_traces": 194, "is_active": false },
    { "name": "traces_custom_run8",    "n_traces": 194, "is_active": false }
  ]
}
```

The frontend uses this to populate the two run-pickers in the "Model comparison" view.

---

### `POST /api/demo/eval/compare`

Backing function: `demo_eval_compare`. Scores two trace-snapshot directories against the same ground truth (`./data/demo/eval_keys.jsonl`) and returns a complete side-by-side scorecard with per-metric deltas. This is the endpoint backing the **Custom vs Base** comparison card in the demo UI.

**Authorization** — gated by `NAT_AML_EVAL_TOKEN` when set (same rules as `/api/demo/eval`).

**Request body**

```json
{
  "token":   "secret-xyz",
  "run_a":   "traces_custom_run8",
  "run_b":   "traces",
  "label_a": "Custom Task NIM (CPT+SFT+RL)",
  "label_b": "Base Nemotron-3-Nano (max_thinking_tokens=50)"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `token`   | string  | conditional | Required when `NAT_AML_EVAL_TOKEN` is set in the server env. |
| `run_a`   | string  | no (default `"traces"`) | Snapshot dir name. Must equal `"traces"` or start with `"traces_"`. Path-traversal is rejected. |
| `run_b`   | string  | yes | Second snapshot dir name; same validation as `run_a`. |
| `label_a` | string? | no | Display label for `run_a` (falls back to its dir name). |
| `label_b` | string? | no | Display label for `run_b`. |

**Response shape**

```json
{
  "ground_truth": {
    "n_total": 194,
    "n_sar":    30,
    "n_no_sar": 164,
    "n_near_miss": 24
  },
  "run_a": {
    "name": "traces_custom_run8",
    "label": "Custom Task NIM (CPT+SFT+RL)",
    "n_predictions": 194,
    "n_missing": 0,
    "parse_errors": 0,
    "confusion":    { "tp": 27, "fp": 27, "tn": 137, "fn": 3 },
    "fp_breakdown": { "near_miss": 12, "clean": 15 },
    "metrics": {
      "accuracy":                       0.8454,
      "precision":                      0.5,
      "recall":                         0.9,
      "f1":                             0.6429,
      "macro_f1":                       0.7721,
      "false_positive_rate_clean":      0.1071,
      "false_positive_rate_near_miss":  0.5,
      "near_miss_specificity":          0.5,
      "narrative_grounding_rate":       0.3889
    },
    "latency": { "avg_case_ms": 3738.3, "n_with_timing": 194 }
  },
  "run_b": {
    "name": "traces",
    "label": "Base Nemotron-3-Nano (max_thinking_tokens=50)",
    "n_predictions": 194,
    "n_missing": 0,
    "parse_errors": 29,
    "confusion":    { "tp": 10, "fp": 4, "tn": 160, "fn": 20 },
    "fp_breakdown": { "near_miss": 1, "clean": 3 },
    "metrics": {
      "accuracy":  0.8763, "precision": 0.7143, "recall": 0.3333,
      "f1":        0.4545, "macro_f1":  0.6924,
      "false_positive_rate_clean":      0.0214,
      "false_positive_rate_near_miss":  0.0417,
      "near_miss_specificity":          0.9583,
      "narrative_grounding_rate":       0.5
    },
    "latency": { "avg_case_ms": 9772.3, "n_with_timing": 194 }
  },
  "diff": {
    "metrics": {
      "f1":        { "absolute":  0.1884, "relative_pct":  41.5 },
      "recall":    { "absolute":  0.5667, "relative_pct": 170.0 },
      "precision": { "absolute": -0.2143, "relative_pct": -30.0 },
      "macro_f1":  { "absolute":  0.0797, "relative_pct":  11.5 }
    },
    "confusion":    { "tp":  17, "fp":  23, "tn": -23, "fn": -17 },
    "fp_breakdown": { "near_miss": 11, "clean": 12 },
    "latency_ms":  -6034.0
  }
}
```

**Diff direction.** All deltas are computed as `run_a − run_b`. A positive `recall.absolute` of `+0.5667` means `run_a` has 0.5667 higher recall than `run_b`. The metric name tells you which direction is better; the consumer (UI) is responsible for the green/red colouring.

**Error responses**

| Status | Body | When |
|---|---|---|
| 200 | `{"error": "unauthorized"}` | Token gate failed. |
| 200 | `{"error": "run_a_not_found", "run_a": "..."}` | `run_a` doesn't resolve to a valid snapshot dir. |
| 200 | `{"error": "run_b_not_found", "run_b": "..."}` | Same, for `run_b`. |
| 200 | `{"error": "run_a_and_run_b_are_same_directory"}` | Same directory passed for both runs. |

Path-traversal attempts (e.g. `"run_b": "../../etc"`) and unrelated directory names (e.g. anything not starting with `traces_`) return `*_not_found` — the validator never reads files outside `./data/traces*`.

**Operational note.** Snapshots are produced by renaming `./data/traces/` after a batch finishes (e.g. `mv data/traces data/traces_custom_run8 && mkdir data/traces`). The `scripts/run_batch.py` runner writes to `./data/traces/`; archive between runs to retain history for comparison.

---

### `POST /api/demo/eval/model_comparison`

Backing function: `demo_eval_model_comparison_report`. Returns a **pre-compiled N-way comparison report** of multiple model endpoints scored on the same eval set. Read-only — does not invoke any LLM at request time, so the response is instant. This is the endpoint that powers the workshop's headline "Model Picker" / leaderboard tile.

The pre-compiled reports live under `./data/benchmarks/` and are produced offline by:

```bash
# 1. Run the same investigation workflow against each endpoint, isolating traces:
python -m scripts.compare_endpoints --concurrency 6 ...
# 2. Score each endpoint's traces against the demo eval keys:
python -m scripts.score_traces --traces data/traces_custom_pathA --eval-keys ... --out data/eval_pathA.json
python -m scripts.score_traces --traces data/traces_base_pathA   --eval-keys ... --out data/eval_base_pathA.json
# ... one per endpoint ...
# 3. Fuse the per-endpoint eval JSONs into one N-way report and update the pointer:
python -m scripts.build_model_comparison_report \
  --eval data/eval_pathA.json:"aml-custom-task-nim" \
  --eval data/eval_base_pathA.json:"nemotron-3-nano (base)" \
  --eval data/eval_gemma_pathA.json:"gemma-4-31b-it (frontier)" \
  --eval data/eval_gpt52.json:"gpt-5.2 (frontier)" \
  --update-latest \
  --out data/benchmarks/four_way_<timestamp>.json
```

**Authorization** — gated by `NAT_AML_EVAL_TOKEN` when set (same rules as `/api/demo/eval`).

**Request body**

```json
{
  "report": "latest",
  "token":  "<NAT_AML_EVAL_TOKEN>"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `report` | string | no (default `"latest"`) | Either the literal `"latest"` (follows `data/benchmarks/latest.json` to the most recent full report), a bare filename like `"four_way_464case_recovered_20260531T064030Z.json"`, or an absolute path under `data/benchmarks/`. |
| `token`  | string | conditional | Required when `NAT_AML_EVAL_TOKEN` is set in the server env. |

**Response** — the body of the resolved report, plus a tiny provenance envelope. The schema below mirrors what `scripts/build_model_comparison_report.py` writes. Numbers shown are from the current default report (464-case intersection of clean parses, filtered from a 500-case prod-mimic v2 eval set).

```json
{
  "report_file": "four_way_464case_recovered_20260531T064030Z.json",
  "report_path": "/data/.../benchmarks/four_way_464case_recovered_<ts>.json",
  "served_from": "/data/.../benchmarks",

  "ran_at":       "2026-05-31T06:40:30.565473+00:00",
  "demo_size":    500,
  "demo_version": "v2",
  "notes":        "Clean intersection 464/500 cases — apples-to-apples 4-way comparison over the cases where all four endpoints produced a parseable response. Excludes 36 cases where at least one endpoint had a parse failure or hard error.",

  "endpoints": [
    { "label": "nemotron-3-nano (base)",            "eval_path": "...", "traces_dir": "...", "n_total_keys": 464 },
    { "label": "gemma-4-31b-it (frontier)",         "...": "..." },
    { "label": "openai/gpt-5.2 (frontier)",         "...": "..." },
    { "label": "aml-custom-task-nim (SFT, custom)", "...": "..." }
  ],

  "coverage": [
    { "field": "n_scored",         "values": { "aml-custom-task-nim (SFT, custom)": 464, ... } },
    { "field": "n_parse_failures", "values": { "aml-custom-task-nim (SFT, custom)": 0,   ... } },
    { "field": "n_errors",         "values": { "aml-custom-task-nim (SFT, custom)": 0,   ... } }
  ],

  "headline_metrics": [
    { "metric": "f1",                     "higher_is_better": true,
      "values": { "aml-custom-task-nim (SFT, custom)": 0.698,
                  "nemotron-3-nano (base)":            0.465,
                  "gemma-4-31b-it (frontier)":         0.398,
                  "openai/gpt-5.2 (frontier)":         0.240 },
      "winner_label": "aml-custom-task-nim (SFT, custom)" },
    { "metric": "precision",              "higher_is_better": true,
      "values": { "aml-custom-task-nim (SFT, custom)": 0.587, "nemotron-3-nano (base)": 0.321,
                  "gemma-4-31b-it (frontier)":         0.257, "openai/gpt-5.2 (frontier)": 0.142 },
      "winner_label": "aml-custom-task-nim (SFT, custom)" },
    { "metric": "recall",                 "higher_is_better": true,
      "values": { "aml-custom-task-nim (SFT, custom)": 0.860, "nemotron-3-nano (base)": 0.837,
                  "gemma-4-31b-it (frontier)":         0.884, "openai/gpt-5.2 (frontier)": 0.791 },
      "winner_label": "gemma-4-31b-it (frontier)" },
    { "metric": "near_miss_specificity",  "higher_is_better": true,
      "values": { "aml-custom-task-nim (SFT, custom)": 0.743, "nemotron-3-nano (base)": 0.371,
                  "gemma-4-31b-it (frontier)":         0.486, "openai/gpt-5.2 (frontier)": 0.200 },
      "winner_label": "aml-custom-task-nim (SFT, custom)" },
    { "metric": "clean_fpr",              "higher_is_better": false,
      "values": { "aml-custom-task-nim (SFT, custom)": 0.044, "nemotron-3-nano (base)": 0.140,
                  "gemma-4-31b-it (frontier)":         0.238, "openai/gpt-5.2 (frontier)": 0.461 },
      "winner_label": "aml-custom-task-nim (SFT, custom)" }
  ],

  "confusion": [
    { "label": "nemotron-3-nano (base)",            "tp": 36, "fp":  76, "tn": 345, "fn": 7 },
    { "label": "gemma-4-31b-it (frontier)",         "tp": 38, "fp": 110, "tn": 311, "fn": 5 },
    { "label": "openai/gpt-5.2 (frontier)",         "tp": 34, "fp": 206, "tn": 215, "fn": 9 },
    { "label": "aml-custom-task-nim (SFT, custom)", "tp": 37, "fp":  26, "tn": 395, "fn": 6 }
  ],

  "per_typology_recall": [
    { "typology": "structuring",
      "values":         { "aml-custom-task-nim (SFT, custom)": 1.0, "nemotron-3-nano (base)": 0.667, ... },
      "n_per_endpoint": { "aml-custom-task-nim (SFT, custom)":   6, ... },
      "winner_label":   "aml-custom-task-nim (SFT, custom)" },
    ...
  ],

  "per_typology_nm_specificity": [
    { "typology": "structuring", "values": { "...": ... }, "winner_label": "..." },
    ...
  ],

  "narrative_stats": [
    { "field": "n_non_empty", "values": { "aml-custom-task-nim (SFT, custom)": 464,   ... } },
    { "field": "mean_chars",  "values": { "aml-custom-task-nim (SFT, custom)": 690.7, ... } },
    { "field": "median_chars","values": { "...": ... } },
    { "field": "min_chars",   "values": { "...": ... } },
    { "field": "max_chars",   "values": { "...": ... } }
  ],

  "wall_clock_ms": [
    { "field": "mean",   "values": { "aml-custom-task-nim (SFT, custom)": 7736.5,
                                      "gemma-4-31b-it (frontier)":         27246.1, ... } },
    { "field": "median", "values": { "...": ... } }
  ]
}
```

**Error responses**

| Status | Body | When |
|---|---|---|
| 200 | `{"error": "unauthorized"}` | Token gate failed. |
| 200 | `{"error": "...benchmarks directory not found..."}` | `data/benchmarks/` doesn't exist; run the offline pipeline first. |
| 200 | `{"error": "Report not found: foo.json. Available reports: [...]"}` | The requested filename isn't under `data/benchmarks/`. |

**Why this endpoint exists vs `/api/demo/eval/compare`.** `/compare` scores **two trace directories at request time** — used for ad-hoc "I just produced these two snapshots, compare them" questions. This endpoint serves a **pre-built N-way report** so the workshop's headline tile loads instantly and is reproducible across audiences. The two endpoints are complementary.

---

## 9. Demo orchestration

### `POST /api/demo/seed_traces`

Backing function: `demo_seed_traces`. Pre-loads the bundled baseline rollout (one full pass of the reference agent over the 194 demo cases — shipped at `./data/seed_traces/`) into `./data/traces/` so the analytics dashboard has content before the trained model has produced any predictions.

**This is NOT how you make predictions.** Predictions come from `POST /api/investigation/run`. Use this only if you want a baseline reference visible on the dashboard.

**Request body** — empty `{}`.

**Response**

```json
{ "n_seeds": 194, "n_written": 187 }
```

`n_written` < `n_seeds` when traces already exist for some `case_id`s (existing files are not overwritten).

---

## 10. System & health

### `GET /api/health`

Backing function: `health`. Liveness probe + data plane sanity check.

```json
{ "ok": true, "n_transactions": 71601, "n_entities": 2072, "ts": 1779522210.4 }
```

On failure: `{"ok": false, "error": "..."}`.

---

### `GET /api/system/config`

Backing function: `system_config`. Shows what's wired — useful for workshop attendees to confirm what data + which models are plugged in.

```json
{
  "data_dir":            "data",
  "transactions_schema": { ... contents of tool_1_transactions/schema.json ... },
  "transactions_stats":  { ... contents of tool_1_transactions/stats.json ... },
  "kyc_schema":          { ... },
  "kyc_stats":           { ... },
  "policy_sources":      { "FinCEN": 23794, "OFAC": 16157, "FFIEC": 2357, "FATF": 248 },
  "sop_count":           8
}
```

---

### `GET /api/system/components`

Backing function: `system_components`. Human-readable summary of every registered NAT component in this workflow.

```json
{
  "registered_groups":    ["aml_data_tools"],
  "registered_functions": [
    "compute_hints", "aux_call (x4 specialist task types)",
    "aux_gate", "sar_judgment_caller",
    "investigate_case (workflow root — deterministic 7-phase orchestrator)",
    "list_alerts, get_alert, post_disposition, alerts_stats",
    "list_entities, get_entity, get_entity_tx, get_entity_behavioral, get_entity_risk, get_entity_network, get_entity_timeline",
    "get_global_network, get_network_patterns, get_network_path",
    "skill_behavioral, skill_numeric, skill_citation, skill_statutory, skill_sar",
    "search_policy, list_policy_sources, list_sops, get_sop_body, screen_name",
    "analytics_overview, analytics_typology, analytics_risk_heatmap, analytics_timeline, analytics_channel_mix, analytics_top_cp, analytics_aux_usage, analytics_agent_perf, analytics_profile",
    "demo_eval, demo_eval_cases, demo_eval_case, demo_eval_runs, demo_eval_compare, demo_eval_model_comparison_report, demo_seed_traces, get_trace, health, system_config"
  ]
}
```

For the live machine-readable inventory, also see `nat info components -t function` on the CLI or `/docs` (Swagger UI).

---

## Quick reference — route summary

| Method | Route | Purpose |
|---|---|---|
| POST  | `/api/investigation/run`                              | End-to-end SAR pipeline (root workflow) |
| GET   | `/api/investigation/{case_id}`                        | Fetch persisted trace |
| GET   | `/api/alerts`                                         | Alert queue (filters + pagination) |
| GET   | `/api/alerts/{alert_id}`                              | One alert + trace + disposition |
| POST  | `/api/alerts/{alert_id}/disposition`                  | Record analyst verdict |
| GET   | `/api/alerts/stats`                                   | Alert-queue summary |
| GET   | `/api/entities`                                       | KYC search |
| GET   | `/api/entities/{entity_id}`                           | Entity 360 profile |
| GET   | `/api/entities/{entity_id}/transactions`              | Tx history |
| GET   | `/api/entities/{entity_id}/behavioral_summary`        | Deterministic behavioral metrics |
| GET   | `/api/entities/{entity_id}/risk_score`                | 0–100 risk score |
| GET   | `/api/entities/{entity_id}/network`                   | N-hop counterparty graph |
| GET   | `/api/entities/{entity_id}/timeline`                  | Daily volume timeline |
| GET   | `/api/network/global`                                 | Global graph stats |
| GET   | `/api/network/patterns`                               | Loop / cycle detections |
| POST  | `/api/network/path`                                   | Shortest path between two nodes |
| POST  | `/api/skills/behavioral`                              | Trained model: behavioral task |
| POST  | `/api/skills/numeric`                                 | Trained model: numeric task |
| POST  | `/api/skills/citation`                                | Trained model: citation task |
| POST  | `/api/skills/statutory`                               | Trained model: statutory task |
| POST  | `/api/skills/sar`                                     | Trained model: final SAR call |
| POST  | `/api/policy/search`                                  | Policy RAG retrieval |
| GET   | `/api/policy/sources`                                 | Policy corpus distribution |
| GET   | `/api/sops`                                           | List SOPs |
| GET   | `/api/sops/{sop_id}`                                  | Render SOP body |
| POST  | `/api/sanctions/screen`                               | Fuzzy sanctions screen |
| GET   | `/api/analytics/overview`                             | Top-line cards |
| GET   | `/api/analytics/typology_distribution`                | Per-typology donut |
| GET   | `/api/analytics/risk_heatmap`                         | Per-jurisdiction heatmap |
| GET   | `/api/analytics/timeline`                             | Daily tx counts |
| GET   | `/api/analytics/channel_mix`                          | Channel mix per typology |
| GET   | `/api/analytics/top_counterparties`                   | Top counterparties by volume |
| GET   | `/api/analytics/aux_usage`                            | Aux gate USED/DROPPED stats |
| GET   | `/api/analytics/agent_performance`                    | Per-typology recall / precision |
| GET   | `/api/analytics/profile`                              | NAT profiler artifacts |
| POST  | `/api/demo/eval`                                      | **Aggregate scorecard** (gated) |
| POST  | `/api/demo/eval/cases`                                | **Per-case correct/incorrect list** (gated) |
| POST  | `/api/demo/eval/case/{case_id}`                       | **Deep-dive on one case** (gated) |
| GET   | `/api/demo/eval/runs`                                 | List trace snapshots available for comparison |
| POST  | `/api/demo/eval/compare`                              | **Side-by-side scorecard for two runs** (gated) |
| POST  | `/api/demo/eval/model_comparison`                     | **Pre-compiled N-way leaderboard** (gated, read-only) |
| POST  | `/api/demo/seed_traces`                               | Pre-load baseline rollout for dashboard |
| GET   | `/api/health`                                         | Liveness + data plane sanity |
| GET   | `/api/system/config`                                  | Show wired data + models |
| GET   | `/api/system/components`                              | NAT-component inventory |

---

## Known limitations

### NAT 1.5 GET handler doesn't bind path or query parameters

Routes declared as `method: GET` with a Pydantic input schema that has any non-default fields receive `args = None` instead of a populated model. NAT 1.5's `front_ends/fastapi/routes/common_utils.py::get_single_endpoint` hard-codes `None` for the input argument.

| Route | Status | Effect |
|---|---|---|
| `GET /api/alerts/{alert_id}` | broken on GET | Alert detail view |
| `GET /api/entities/{entity_id}` | broken on GET | Entity 360 profile |
| `GET /api/entities/{entity_id}/transactions` | broken on GET | Tx history tab |
| `GET /api/entities/{entity_id}/behavioral_summary` | broken on GET | Behavioral tab |
| `GET /api/entities/{entity_id}/risk_score` | broken on GET | Risk score |
| `GET /api/entities/{entity_id}/network` | broken on GET | Entity network graph |
| `GET /api/entities/{entity_id}/timeline` | broken on GET | Entity timeline |
| `GET /api/sops/{sop_id}` | broken on GET | SOP detail |
| `GET /api/investigation/{case_id}` | broken on GET | Trace fetch (workshop demo uses `/api/demo/eval/case/{case_id}` POST instead) |

**Why it happens** — NAT's GET handler is intentionally side-effect-free and treats the function as a "no-input" generate call. Path/query params from FastAPI are visible in `request.path_params` inside NAT but not surfaced to the registered function.

**Workaround** — convert affected routes to `method: POST` and pass the identifier in the JSON body. POST handlers receive a properly populated input model. Example fix in workflow.yaml:

```yaml
- path: /api/entities/{entity_id}     # original
  method: GET
  function_name: get_entity
```

becomes

```yaml
- path: /api/entities/get              # POST workaround
  method: POST
  function_name: get_entity
```

Frontend then calls `POST /api/entities/get` with body `{"entity_id": "SYN_..."}`.

**Demo impact** — the workshop's headline flow (`POST /api/investigation/run`, the seven `POST /api/demo/eval/*` routes including `model_comparison`, all skill playgrounds, and all 9 `analytics_*` GETs) is unaffected. Only the Entity-360 and Alert-Detail UI tiles need the POST workaround to function.

### Listing routes work via defensive None handling

`GET /api/alerts`, `GET /api/entities`, `GET /api/alerts/stats`, and `GET /api/analytics/*` succeed because their input schemas have only `Optional` fields with defaults. The handler body defaults the args object on `None`:

```python
async def _run(args: ListAlertsInput) -> dict:
    if args is None:
        args = ListAlertsInput()
    ...
```

This is the recommended pattern for any new GET route with a non-Empty schema.

### Route ordering in workflow.yaml

`/api/alerts/stats` must appear **before** `/api/alerts/{alert_id}` in the `endpoints:` list; otherwise FastAPI matches `stats` as a value for `{alert_id}` and routes to the wrong function. The current YAML has the correct order, but any future addition of static-path routes under `/api/alerts/` or similar prefixes must respect this constraint.

### Skill playgrounds require the local Custom-Task NIM

The five `/api/skills/*` routes invoke `custom_task_nim` (default `http://localhost:8088/v1`). When the NIM container is down, all five routes return `"Cannot connect to host localhost:8088"`. The deterministic workflow at `POST /api/investigation/run` and the read-only eval endpoints (`/api/demo/eval/model_comparison` etc.) are unaffected — only the live skill playgrounds need the NIM running.
