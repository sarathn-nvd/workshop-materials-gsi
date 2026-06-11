# AML Investigation Agent — Backend Strategy

This document is the internal backend strategy for the end-to-end AML
Investigation application. The application is the deployment surface
for the model produced by the CPT → SFT pipeline in this repository;
the trained model is consumed as a Custom Task NIM endpoint.

The contract the backend implements is the one specified in
[`5.sdg_corpus_mimic/AGENT_USAGE_GUIDE.md`](../../5.sdg_corpus_mimic/AGENT_USAGE_GUIDE.md).

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Data Layer](#2-data-layer)
3. [Tool Catalog (NAT Functions)](#3-tool-catalog-nat-functions)
4. [Core Investigation Workflow](#4-core-investigation-workflow)
5. [FastAPI Surface — Routes for the Demo UI](#5-fastapi-surface--routes-for-the-demo-ui)
6. [Demo-Enrichment Functionalities](#6-demo-enrichment-functionalities)
7. [Directory Layout](#7-directory-layout)
8. [Tech Stack and Dependencies](#8-tech-stack-and-dependencies)
9. [Explainability, Observability, and Profiling](#9-explainability-observability-and-profiling)
10. [Configuration and Environment](#10-configuration-and-environment)

---

> **Note on agent shape.** The trained Custom Task NIM was
> post-trained for the five `task_type` skills only — it does not
> emit tool-calling JSON. We therefore use a **deterministic NAT
> workflow function** (`investigate_case`) as the workflow root.
> The leaves (data tools, aux skill calls, gate, SAR caller) are
> still first-class NAT components, the LLM is still NAT's `nim`
> provider, observability/profiling are still NAT-native, and the
> FastAPI front-end is still `nat serve`. The §4 "Per-case execution"
> section is the deployed reality.

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                  FRONTEND (React / Next.js)                       │
│  Investigation Cockpit • Entity 360 • Network Graph • Analytics Dashboard         │
│  Alert Queue & Case Mgmt • Skill Playgrounds • Live Agent Trace Viewer            │
│  Model Comparison Tile (custom NIM vs base Nemotron vs frontier)                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                   REST + WebSocket (SSE)
                                          │
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Front-End (nat serve, custom routes)             │
│                                                                                   │
│  /api/investigation/*      /api/alerts/*          /api/entities/*                 │
│  /api/network/*            /api/analytics/*       /api/skills/*                   │
│  /api/policy/*             /api/sops/*            /api/sanctions/*                │
│  /api/demo/* (eval)        /api/system/* (health) /ws/investigation/{case_id}     │
└──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              NeMo Agent Toolkit Runtime (workflow_builder)                        │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  WORKFLOW ROOT — investigate_case (deterministic Python orchestration)    │  │
│  │                                                                            │  │
│  │   Phase 1 — Data fetch     (get_transactions, get_kyc, screen_sanctions)   │  │
│  │   Phase 2 — Internal       (classify_typology + semantic_profile,          │  │
│  │             typology guess  routing-only; never sent to the SAR LLM)       │  │
│  │   Phase 3 — Retrieval      (retrieve_policy focused+broad, get_sop)        │  │
│  │   Phase 4 — Aux skills     (3 LLM calls + 1 deterministic Python calc):    │  │
│  │              • aux_numeric_call                                            │  │
│  │              • aux_citation_call                                           │  │
│  │              • aux_statutory_call                                          │  │
│  │              • behavioral — Python-computed; NO LLM call                   │  │
│  │   Phase 5 — Aux gate       (input guard → schema → optional LLM judge)     │  │
│  │   Phase 6 — SAR judgment   (sar_judgment_caller → custom_task_nim:         │  │
│  │              7-key user message, 2-field response)                         │  │
│  │   Phase 7 — Trace persist  (./data/traces/{case_id}.json)                  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  LLM:  custom_task_nim   (vLLM / NIM serving the trained AML model)               │
│  LLM:  judge_llm         (optional — used inside aux_gate only)                   │
│                                                                                   │
│  Instrumentation (always-on):                                                     │
│    • Explainability — per-case CaseTrace JSON → ./data/traces/                    │
│    • Observability  — NAT OpenTelemetry → Arize Phoenix (every span)              │
│    • Profiling      — NAT profiler → per-invocation token + latency stats         │
└──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              Local Data Plane  (./data)                            │
│                                                                                    │
│  ./data/tool_1_transactions/transactions.parquet                                   │
│  ./data/tool_2_kyc/entities.parquet                                                │
│  ./data/tool_3_sanctions/{ofac.csv, pep.csv}                                       │
│  ./data/tool_4_policy/policy_chunks.parquet                                        │
│  ./data/tool_5_sop/*.md                                                            │
│  ./data/demo/{manifest.jsonl, eval_keys.jsonl, stratification_report.json}         │
│  ./data/seeded_subpopulations/*.parquet                                            │
│  ./data/traces/{case_id}.json            (write — persisted reasoning chains)      │
│  ./data/dispositions/{case_id}.json      (write — analyst verdicts)                │
│  ./data/benchmarks/*.json                (pre-compiled multi-model comparisons)    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Five deliberate choices in the above:**

1. **Deterministic Python orchestration at the root.** A plain async
   function (`investigate_case`) walks Phases 1–7 in fixed order. The
   trained checkpoint is asked one job per call — produce a SAR
   verdict from the 7-key bundle, or produce an aux-skill finding —
   never to decide what tool to call next. This matches what the SFT
   corpus taught it and keeps the runtime behavior reproducible.

2. **Two hard rules drive the bundle assembly.** The user message
   into the SAR call carries exactly 7 evidence keys —
   `task_type, transactions, kyc_profile, sanctions_pep_hits,
   policy_excerpts, sop_excerpts, auxiliary_findings`. No
   pre-computed typology, regulatory frame, or decision hint is in
   the user message. `SARCallerInput` uses Pydantic's
   `extra="forbid"` so an accidental extra key is rejected before
   the LLM is hit. The assistant response is exactly two fields —
   `{is_suspicious, suspicious_activity_report}` — and the SAR
   narrative is non-empty for both polarities (negatives produce a
   disposition rationale; positives produce a regulator-grade SAR).

3. **Behavioral finding is computed deterministically in Python at
   runtime.** Channel mix, velocity, counterparty concentration,
   declared-vs-observed volume ratio, and loop / pass-through
   indicators are computed by a Python feature computer that is
   byte-identical with the SFT Stage-7 helper that produced gold
   `behavioral.metrics` blocks at training time. The other three
   aux skills (`numeric`, `citation`, `statutory`) are still
   `task_type`-keyed LLM calls to the trained model. This matches
   the train-time contract for the behavioral block exactly.

4. **Internal typology guess is routing-only.** The rule-layer
   typology classification (Phase 2) selects which question /
   passage / statute to feed the aux skills (Phase 4) and which
   typology to filter the policy / SOP retrievers on (Phase 3).
   The guess is **never** passed into the SAR-judgment user
   message. The model derives the verdict from raw evidence alone.

5. **Two LLM bindings, distinct roles.** `custom_task_nim` is the
   trained AML model, called inside each aux-skill leaf and inside
   `sar_judgment_caller`. `judge_llm` is the (optional) reviewer
   inside `aux_gate` — kept as a separate binding so the model
   never grades its own output. For the workshop default, the
   judge stage is disabled (`NAT_AML_ENABLE_JUDGE=false`); the
   binding is wired up for completeness.

---

## 2. Data Layer

The data plane lives at `./data/` under the backend root. Everything
the agent reads at runtime is on local disk — no remote services. The
loaders are written so swapping any one of them for a service-backed
reader later is a single-file change.

| Path under `./data/` | Format | Contents |
|---|---|---|
| `tool_1_transactions/transactions.parquet` | Parquet | Money-movement events for every entity over the demo window. Columns: `date, amount, currency, counterparty, channel, notes, transaction_id, entity_id`, plus internal sidecars (`source_pool`, `typology_tag`) that are ground-truth-only and stripped at the tool's response boundary. Drives Tool 1. |
| `tool_1_transactions/{schema,stats}.json` | JSON | Schema and summary statistics for the transactions Parquet — surfaced verbatim by `/api/system/config`. |
| `tool_2_kyc/entities.parquet` | Parquet | One row per entity. Columns: `entity_id, entity_type, expected_monthly_volume, business_purpose, risk_rating, incorporation_jurisdiction`, plus internal `source_pool` and `_archetype` (stripped at boundary). Covers archetypes from clean retail businesses to shell-holding offshore and crypto-VASPs. Drives Tool 2. |
| `tool_2_kyc/{schema,stats}.json` | JSON | Schema + stats for the KYC Parquet. |
| `tool_3_sanctions/ofac.csv` | CSV | OFAC enforcement target list with `name, aliases, countries, addresses, program_ids`. Fuzzy-matched against transaction counterparties. Drives Tool 3 (OFAC half). |
| `tool_3_sanctions/pep.csv` | CSV | Politically Exposed Persons starter list, same schema as the OFAC CSV. Drives Tool 3 (PEP half). |
| `tool_4_policy/policy_chunks.parquet` | Parquet | Pre-chunked regulatory corpus. Columns: `chunk_id, source ∈ {FFIEC, FATF, FinCEN, OFAC}, section, url, text, typology_tags`. The agent retrieves stratified top-k by `typology_tags` for the current case. Drives Tool 4. |
| `tool_5_sop/{typology}_v{n}.md` | Markdown | One file per canonical typology — investigation playbook with sections for *Investigation Steps*, *Escalation Criteria*, *Documentation Requirements*, *Filing Decision*, *Tools and Systems*, *References*. Tool 5 returns the highest-priority section matching the current typology. |
| `demo/manifest.jsonl` | JSONL | The alert queue. One alert per line: `case_id, alert_id, entity_id, investigation_window_start, investigation_window_end, trigger_summary`. Powers `/api/alerts` and the batch runner. |
| `demo/eval_keys.jsonl` | JSONL | Ground truth per case (`expected_typology, expected_label, near_miss, expected_evidence`). **Gated**: the only routes that may read it are the `/api/demo/eval/*` family, behind a bearer token. |
| `demo/stratification_report.json` | JSON | Per-typology distribution of the demo manifest. Backs the analytics dashboard's typology-mix donut. |
| `seeded_subpopulations/{suspicious,near_miss}_entities.parquet` | Parquet | Pre-curated entity lists — seeded suspicious cases and seeded "looks suspicious but isn't" near-miss cases. Used by the analytics dashboard's "highlights" and the per-typology guided tour. |
| `seeded_subpopulations/seed_manifest.json` | JSON | Metadata for the above (counts per typology, seed config). |
| `traces/{case_id}.json` | JSON (write) | Per-case full reasoning chain, written at end of every run. The cockpit, replay, and eval views all read from this. |
| `dispositions/{case_id}.json` | JSON (write) | Analyst verdict (`file_sar / dismiss / escalate`) + note + timestamp. Drives the disposition workflow and agent-vs-analyst agreement chart. |
| `benchmarks/*.json` | JSON | Pre-compiled multi-model comparison reports produced offline by `scripts/build_model_comparison_report.py`. `latest.json` is a pointer file to the most recent full report. Served verbatim by `/api/demo/eval/model_comparison`. |

**Internal-column stripping.** Two columns on
`tool_1_transactions/transactions.parquet` (`source_pool`,
`typology_tag`) and two on `tool_2_kyc/entities.parquet`
(`source_pool`, `_archetype`) are ground-truth sidecars. They exist
on disk so the analytics routes can aggregate over them, but they
are stripped at the tool's response boundary — the agent never sees
them.

---

## 3. Tool Catalog (NAT Functions)

All entries below are NAT components registered with
`@register_function` or `@register_function_group` and listed in
`configs/workflow.yaml`. They fall into three layers:

- **Layer A — Leaf tools** (§3.1, §3.2, §3.4, §3.5): pure-Python
  or single-LLM-call functions.
- **Layer B — Skill playgrounds** (§3.3): per-task-type wrappers
  for the interactive `/api/skills/*` routes.
- **Layer C — Deterministic workflow** (§3.6, §4): the
  `investigate_case` function that walks Phases 1–7.

### 3.1 Data tools (function group `aml_data_tools`)

Shared resources: Parquet handles (lazy-loaded once per process),
`country_risk` lookup table. All five tools return Pydantic models
that match the canonical schema.

| `_type` (registered name) | Input → Output | How the data is leveraged |
|---|---|---|
| `aml_data_tools.get_transactions` | `(entity_id, window_start, window_end) → list[Transaction]` | Loads `./data/tool_1_transactions/transactions.parquet` once into a Pandas dataframe; filters rows on `entity_id` AND `date ∈ [window_start, window_end]`. Strips `source_pool` and `typology_tag` before returning. |
| `aml_data_tools.get_kyc` | `entity_id → KYCProfile` | Loads `./data/tool_2_kyc/entities.parquet` once into an `entity_id → row` dict; returns the row's canonical fields only (`source_pool`, `_archetype` stripped). Raises `EntityNotFound` on miss. |
| `aml_data_tools.screen_sanctions` | `(name, country=None, min_score=0.55) → list[SanctionsHit]` | Loads `./data/tool_3_sanctions/{ofac,pep}.csv`; runs RapidFuzz `token_set_ratio(name, candidate)` across both pools; boosts score by 5% when `country` matches the candidate's listed country. Returns top-5 hits above `min_score`, tagged with `list ∈ {"OFAC", "OpenSanctions"}`. |
| `aml_data_tools.retrieve_policy` | `(typology, k=4, activity_descriptor=None) → list[PolicyExcerpt]` | Loads `./data/tool_4_policy/policy_chunks.parquet`; filters rows where `typology in typology_tags`; **stratifies** the top-k pick across the four `source` enums (FinCEN, FFIEC, FATF, OFAC) so the bundle's evidence isn't dominated by a single source. |
| `aml_data_tools.get_sop` | `(typology, variant=1, section=None) → list[SOPExcerpt]` | Reads `./data/tool_5_sop/{typology}_v{variant}.md`, splits on `##` headings, and returns the highest-priority section. |

### 3.2 Internal typology hint (single function `compute_hints`)

| `_type` | Input → Output | Notes |
|---|---|---|
| `compute_hints` | `(transactions, kyc, sanctions_pep_hits, trigger_summary) → {typology_inferred, regulatory_frame, activity_descriptor, semantic_profile}` | Pure Python, no LLM. Wraps `classify_typology` + `compute_semantic_profile`. The result is used **internally only** to (a) select the typology-keyed aux-skill questions / passages / statutes, and (b) filter the policy / SOP retrievers. It is **not** part of the SAR-judgment user message. |

### 3.3 Auxiliary skill calls

The four aux skills are exposed both internally (called from
`investigate_case`) and as `/api/skills/*` playgrounds. Each leaf
call wraps a single LLM call to `custom_task_nim` with a specific
`task_type` and Pydantic-validates the response.

| Leaf function | Task type | Returns |
|---|---|---|
| `aux_behavioral_call` | `auxiliary_behavioral` | `BehavioralFinding` — `{summary, metrics, evidence}` |
| `aux_numeric_call`    | `auxiliary_numeric`    | `NumericFinding` — `{question, answer, calculation, evidence}` |
| `aux_citation_call`   | `auxiliary_citation`   | `CitationFinding` — `{question, answer, evidence_span}` |
| `aux_statutory_call`  | `auxiliary_statutory`  | `StatutoryFinding` — `{question, answer, label, reasoning}` |

> **Runtime note on `auxiliary_behavioral`.** Inside the workflow
> (`investigate_case`), the behavioral finding is **not** produced
> by an LLM call. Instead, it is computed deterministically in
> Python (the same logic the SFT Stage-7 helper used to anchor gold
> training `metrics`), then overlaid onto the post-gate aux
> findings as if it had been an LLM response. The `aux_behavioral_call`
> leaf still exists and is wired up — but it is invoked only via
> the `/api/skills/behavioral` playground, never inside the main
> investigation workflow. This is controlled by env var
> `NAT_AML_BEHAVIORAL_MODE=python_only` (the default).

The aux LLM-call leaves (`aux_numeric_call`, `aux_citation_call`,
`aux_statutory_call`) are thin Pydantic-typed wrappers that
encapsulate the SFT-time system prompt verbatim, the JSON-parse,
and the `{Numeric|Citation|Statutory}Finding` validation. The user
messages they send to the trained model are the task-type schema
**only** — no typology / frame / label hint in any of them.

### 3.4 Gating function

| `_type` | Input → Output | Notes |
|---|---|---|
| `aux_gate` | `(findings, transactions, kyc, policy_excerpts, typology) → (AuxiliaryFindings, list[GateDecision])` | Two-or-three-stage per finding: input-availability guard → schema validity (Pydantic) → optional LLM-as-Judge (one `judge_llm` call per surviving finding). Returns the filtered `AuxiliaryFindings` ready to inline into the SAR bundle. |

The judge stage is opt-in (`NAT_AML_ENABLE_JUDGE=true`) and uses a
second LLM binding (`judge_llm`) so the model never grades its own
output. Workshop default: judge disabled.

### 3.5 Final SAR call

| `_type` | Input → Output | Notes |
|---|---|---|
| `sar_judgment_caller` | `(transactions, kyc_profile, sanctions_pep_hits, policy_excerpts, sop_excerpts, auxiliary_findings) → SARJudgmentOutput` | Builds the **7-key user message in the exact order** required by the SFT contract, posts to `custom_task_nim`, parses to `{is_suspicious, suspicious_activity_report}`. The Pydantic input schema is `extra="forbid"` so any caller passing a hint field is rejected before the LLM is invoked. |

The 7 keys are exactly:

```text
1. task_type            ("sar_judgment")
2. transactions         []
3. kyc_profile          {}
4. sanctions_pep_hits   []
5. policy_excerpts      []
6. sop_excerpts         []
7. auxiliary_findings   {behavioral, numeric, citation, statutory} | null
```

This shape is byte-identical to what the SFT corpus trained the
model on. The training-distribution parity is the lever that
keeps the deployed model accurate.

### 3.6 Investigation workflow function

| Field | Value |
|---|---|
| `_type` | `investigate_case` |
| Implementation | Deterministic Python async function in `aml_app/workflow/investigate_case.py` |
| Sequence | Fixed Phases 1–7 (see §4) |
| LLM count per case | 4 calls — 3 aux LLM calls (numeric, citation, statutory) + 1 SAR call. Behavioral aux is Python; judge is opt-in. |

The workflow function is registered as a NAT `workflow` and bound
to the FastAPI root POST path `/api/investigation/run`. Its output
is a fully-populated `CaseTrace` (see §9.1).

---

## 4. Core Investigation Workflow

### 4.1 Per-case execution (the seven phases)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│   PHASE 1 — DATA FETCH (3 deterministic tools, in parallel)                  │
│                                                                              │
│     get_transactions(entity, window)        → transactions[]                 │
│     get_kyc(entity)                         → kyc_profile                    │
│     screen_sanctions(entity, counterparts)  → sanctions_pep_hits[]           │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│   PHASE 2 — INTERNAL TYPOLOGY GUESS (rule layer; never reaches the LLM)      │
│                                                                              │
│     guess_typology = classify_typology(tx, kyc, hits, trigger)               │
│                                                                              │
│   Used internally only to:                                                   │
│     • choose the typology-specific question/passage/statute for aux skills   │
│     • filter policy + SOP retrieval                                          │
│                                                                              │
│   The value is recorded in the trace for audit, but never serialised into    │
│   the SAR-judgment user message.                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│   PHASE 3 — RETRIEVAL (2 more deterministic tools)                           │
│                                                                              │
│     policy_focused = retrieve_policy(typology=guess_typology, k=3)           │
│     policy_broad   = retrieve_policy(no_filter, k=2)        ← safety net     │
│     sop_excerpts   = get_sop(typology=guess_typology)                        │
│                                                                              │
│   The broad-search safety net is mandatory: if Phase 2's guess is wrong,     │
│   the model still has alternative regulatory context to fall back on.        │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│   PHASE 4 — AUXILIARY SKILLS (3 parallel LLM calls + 1 deterministic calc)   │
│                                                                              │
│     numeric_agent   : aux_numeric_call(passage, NUMERIC_Q[guess_typ])        │
│     citation_agent  : aux_citation_call(policy_focused[0], CITATION_Q)       │
│     statutory_agent : aux_statutory_call(STATUTE[guess_typ], facts)          │
│     behavioral      : computed_in_python(tx, kyc)  ← NO LLM call             │
│                                                                              │
│   The user message into each aux LLM call is the task-type schema only —    │
│   no typology / frame / decision hint is in any of them.                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│   PHASE 5 — AUX GATE                                                         │
│                                                                              │
│   Each finding survives only if:                                             │
│     1. The input passage was present and well-formed                         │
│     2. The output matches its typed Pydantic schema                          │
│     3. (Optional, off by default) An independent judge LLM confirms          │
│        consistency with the underlying data                                  │
│                                                                              │
│   The deterministic behavioral finding is overlaid onto the surviving        │
│   findings AFTER the gate, so it is always present and trusted.              │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│   PHASE 6 — SAR JUDGMENT (one LLM call to custom_task_nim)                   │
│                                                                              │
│   USER MESSAGE — exactly 7 keys, evidence only:                              │
│     { task_type:"sar_judgment",                                              │
│       transactions:[…], kyc_profile:{…},                                     │
│       sanctions_pep_hits:[…], policy_excerpts:[…],                           │
│       sop_excerpts:[…], auxiliary_findings:{…}|null }                        │
│                                                                              │
│   MODEL OUTPUT — exactly 2 fields:                                           │
│     { is_suspicious: <bool>,                                                 │
│       suspicious_activity_report: "<400–800 char grounded narrative>" }      │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│   PHASE 7 — TRACE PERSISTENCE                                                │
│                                                                              │
│   A CaseTrace JSON is written to ./data/traces/{case_id}.json with every     │
│   phase's inputs, the rule-layer typology guess (audit-only), all aux        │
│   findings (raw + gated), the SAR call's user message + raw text + parsed    │
│   output, and per-phase latencies. This is the audit trail consumed by       │
│   analytics + the /api/demo/eval/* family of endpoints.                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Why a deterministic workflow (not an agentic stack)

- The trained Custom Task NIM was post-trained for the five
  `task_type` skills. It is not a tool-calling model. Asking it
  to plan or select tools would push it off-distribution.
- Every phase boundary is a place where the SFT contract has to
  hold (e.g. the 7-key bundle into Phase 6). A deterministic
  walker makes the contract explicit at every boundary and easy
  to test.
- The "agentic" flexibility we still get is at the **leaves**:
  three of the four aux skills are real LLM calls with prompt
  templating, parsing, and gating around them.

### 4.3 Failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| Aux LLM returns malformed JSON | Pydantic validation in the leaf rejects; the gate records `schema_failure` and drops the finding; the SAR call still proceeds with whatever survived. |
| Aux gate drops everything | The SAR call still gets the deterministic behavioral finding (overlaid after the gate) — never empty. |
| The trained model emits non-JSON for the SAR call | The wrapper records `sar_parse_error` in the trace and returns `is_suspicious=null` so downstream UI flags the case for analyst review. |
| Phase 1 data missing for the case | The data tool raises `EntityNotFound`; the workflow records the error in the trace and returns early. |

### 4.4 Standalone component invocation

Because every component is NAT-registered, each of the following
is independently invokable from the `nat` CLI by setting the entry
function: any single data tool, any single aux-skill leaf call, the
aux gate on a hand-built findings bundle, the SAR caller on a
hand-built 7-key bundle, or the full workflow. This is what powers
the `/api/skills/*` playground routes, the `/api/system/components`
self-discovery view, and the workshop's "let me show you this one
tool in isolation" demo moves.

---

## 5. FastAPI Surface — Routes for the Demo UI

All routes below are declared in
`configs/workflow.yaml` → `general.front_end.endpoints` and served
by NAT's built-in FastAPI front-end (`nat serve` → uvicorn on
`:8000`). Each `function_name` resolves to a NAT-registered
function.

### 5.1 Core investigation routes (the SAR pipeline)

| Route | Method | Backing function | Purpose |
|---|---|---|---|
| `/api/investigation/run` | POST | (root workflow) | Run end-to-end on one alert (body: `{case_id}` OR full alert payload). Returns the final trace. |
| `/api/investigation/{case_id}` | GET | `get_trace` | Retrieve persisted trace from `./data/traces/{case_id}.json`. |

### 5.2 Alert queue / case management

| Route | Method | Purpose |
|---|---|---|
| `/api/alerts` | GET | List alerts from `./data/demo/manifest.jsonl` with filters and pagination. Status is derived from `./data/traces/` + `./data/dispositions/`. |
| `/api/alerts/{alert_id}` | GET | One alert + linked entity snippet + latest trace if present. |
| `/api/alerts/{alert_id}/disposition` | POST | Analyst verdict: `{verdict: "file_sar" | "dismiss" | "escalate", note}`. Persists to `./data/dispositions/{case_id}.json`. |
| `/api/alerts/stats` | GET | Counts: open / in-progress / closed; by-typology breakdown; SLA buckets. |

### 5.3 Entity 360

| Route | Method | Purpose |
|---|---|---|
| `/api/entities` | GET | List/search entities with filters (`risk_rating`, `entity_type`, `jurisdiction`, free text). |
| `/api/entities/{entity_id}` | GET | Full KYC profile + counts (tx_n in last 90d, sanctions_hits_n, related alerts_n). |
| `/api/entities/{entity_id}/transactions` | GET | Paginated tx history with window filter. Strips internal columns. |
| `/api/entities/{entity_id}/behavioral_summary` | GET | Returns the deterministic behavioral metrics block computed for the entity's last 90 days. |
| `/api/entities/{entity_id}/risk_score` | GET | Derived risk score: weighted blend of (KYC `risk_rating`, behavioral metric outliers, sanctions hit count, country risk). |
| `/api/entities/{entity_id}/network` | GET | N-hop counterparty graph (`depth=2`, `window=90d`). NetworkX-built. |
| `/api/entities/{entity_id}/timeline` | GET | Daily tx volume + alert markers over the configured window. |

### 5.4 Network / graph analysis (multi-entity)

| Route | Method | Purpose |
|---|---|---|
| `/api/network/global` | GET | Global counterparty graph summary stats (n_nodes, n_edges, top hubs by PageRank). |
| `/api/network/patterns` | GET | Pre-computed loop / pass-through / fan-out detections across the corpus. |
| `/api/network/path` | POST | Shortest-path between two entities. |

### 5.5 Skill playgrounds (interactive)

| Route | Method | Purpose |
|---|---|---|
| `/api/skills/behavioral` | POST | Invoke `aux_behavioral_call` on a raw `{passage}` payload (LLM-driven path; the workflow itself uses the Python computer). Useful for showing the trained model's behavioral skill in isolation. |
| `/api/skills/numeric` | POST | Same, for numeric. |
| `/api/skills/citation` | POST | Same, for citation (paste a policy excerpt). |
| `/api/skills/statutory` | POST | Same, for statutory (paste statute + fact pattern). |
| `/api/skills/sar` | POST | Run only the final SAR call on a hand-assembled 7-key bundle. Lets the workshop attendee see the model's behavior on edge cases. |

### 5.6 Policy / SOP / sanctions tooling (standalone)

| Route | Method | Purpose |
|---|---|---|
| `/api/policy/search` | POST | Query the policy RAG with `{typology, q, k}`. Returns `list[PolicyExcerpt]` with highlights. |
| `/api/policy/sources` | GET | Distribution of the corpus by `source` enum (FinCEN, OFAC, FFIEC, FATF). |
| `/api/sops` | GET | List of typology SOPs. |
| `/api/sops/{sop_id}` | GET | Render SOP body as Markdown. |
| `/api/sanctions/screen` | POST | Free-form sanctions screen: `{name, country?}`. |

### 5.7 Analytics dashboard (chart data)

| Route | Method | Purpose |
|---|---|---|
| `/api/analytics/overview` | GET | Top-line cards: total alerts, % open, n entities, n transactions, n SARs filed, avg case latency. |
| `/api/analytics/typology_distribution` | GET | Donut: alerts per typology. |
| `/api/analytics/risk_heatmap` | GET | Choropleth-ready: alerts × jurisdiction. |
| `/api/analytics/timeline` | GET | Alerts and SAR filings over time (daily bins). |
| `/api/analytics/channel_mix` | GET | Stacked bar: tx counts per channel per typology. |
| `/api/analytics/top_counterparties` | GET | Top-N counterparties by sanctions hit count / risk-weighted volume. |
| `/api/analytics/aux_usage` | GET | How often each aux finding was USED vs DROPPED by the gate. |
| `/api/analytics/agent_performance` | GET | Per-typology recall / precision (if `/api/demo/eval` has been run). |
| `/api/analytics/profile` | GET | NAT-profiler summary: per-component latency, token-efficiency per LLM binding, bottleneck and concurrency analysis. |

### 5.8 Demo orchestration + evaluation

| Route | Method | Purpose |
|---|---|---|
| `/api/demo/run_batch` | POST | Run the workflow over the manifest (params: `limit`, `concurrency`). SSE-streams per-case completions. |
| `/api/demo/eval` | POST | Score the persisted traces against `./data/demo/eval_keys.jsonl`. Returns confusion matrix + per-typology recall + grounding score. **Token-gated.** |
| `/api/demo/eval/cases` | POST | Per-case prediction-vs-ground-truth list, filterable by outcome / correctness / typology. Same token gate. |
| `/api/demo/eval/case/{case_id}` | POST | Deep-dive for a single case: ground truth + prediction + every tool output + the full audit + the SAR text. Same token gate. |
| `/api/demo/eval/runs` | GET | Lists trace-snapshot directories under `./data/` available for scoring (`traces/` plus any sibling whose name starts with `traces_`). Populates the run-pickers in the model-comparison UI. |
| `/api/demo/eval/compare` | POST | **Side-by-side scorecard for any two trace snapshots.** Scores both directories against the same ground truth and returns each run's confusion matrix + headline metrics + per-metric deltas. Token-gated. |
| `/api/demo/eval/model_comparison` | POST | **Pre-compiled N-way scorecard** — returns the most recent `data/benchmarks/*.json` report comparing the custom NIM against base Nemotron + frontier models (Gemma, GPT-5.2). Read-only — does not invoke any LLM at request time. Optional `{"report": "<filename>"}` body to pick a specific report; defaults to the `latest.json` pointer. Token-gate honored. |
| `/api/demo/seed_traces` | POST | Pre-loads a bundled baseline run so the dashboard has content before the user runs anything. |

### 5.9 System / health

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness + downstream LLM endpoint reachability. |
| `/api/system/config` | GET | Show the wired workflow YAML (model names, tool paths) — useful so workshop attendees can see what's plugged in. |
| `/api/system/components` | GET | Wrap `nat info components` — list every registered function / function-group / LLM. |

---

## 6. Demo-Enrichment Functionalities

These are the demo-facing capabilities that ride on top of the core
SAR pipeline. Each one is wired in the backend and exposed via the
API surface in §5.

| # | Functionality | What it does |
|---|---|---|
| 1 | **Live Investigation Cockpit** | Streams the per-case workflow live — Phase 1 tool calls, the rule-layer typology guess, Phase 3 retrievals, Phase 4 aux skills (3 LLM + 1 Python), the gate's verdicts, and the final SAR rendering. |
| 2 | **Entity 360** profile page | One screen per entity with tabs for KYC, Transactions, Behavioral Summary (Python-computed), Network, Alerts, and Risk Score. |
| 3 | **Network Graph viewer** | Force-directed N-hop counterparty graph. Surfaces money-mule chains, pass-through accounts, shared counterparties. |
| 4 | **Skill Playgrounds** | One playground per `task_type`: paste a passage / bundle into `/api/skills/{behavioral, numeric, citation, statutory, sar}` and see the trained model emit a typed finding. |
| 5 | **Analytics Dashboard** | Open-alert counters, typology distribution, jurisdiction risk heatmap, channel-mix bars, alerts/SAR timeline, top counterparties, aux-usage breakdown (USED vs DROPPED), per-typology agent performance. |
| 6 | **Side-by-side SAR comparison** — same case with (a) bare bundle, (b) augmented findings | Two parallel workflow runs per case. Surfaces the value of the auxiliary skills directly. |
| 6a | **Model-vs-Model live scorecard** | The frontend pulls available runs from `/api/demo/eval/runs` and posts both selections to `/api/demo/eval/compare`. The endpoint scores each run against the same ground truth and returns both scorecards with a `diff` block (every metric carries `{absolute, relative_pct}`). For ad-hoc comparisons across trace snapshots. |
| 6b | **Pre-compiled multi-model leaderboard** | `/api/demo/eval/model_comparison` returns the latest pre-built N-way report comparing the custom-task NIM against base Nemotron + frontier models. Drives the workshop's headline "what did fine-tuning actually buy us?" tile — no LLM is invoked at request time, so the panel loads instantly. |
| 7 | **Policy RAG explorer** | `/api/policy/search` exposes the same stratified retriever the workflow uses, so attendees can grep the regulatory corpus the same way the agent does. |
| 8 | **"What if?" simulator** | Edit a transaction bundle in-place and re-run the SAR call via `/api/skills/sar`. Shows the model's sensitivity to evidence. |
| 9 | **Disposition workflow with audit trail** | Analyst marks each case "File SAR / Dismiss / Escalate" with a note; persisted to `./data/dispositions/{case_id}.json`. The analytics dashboard surfaces agent-vs-analyst agreement from this store. |
| 10 | **Aux-gate inspector** | Per case, show all aux findings + which were dropped at which gate stage + the judge's rationale (when enabled). Showcases the grounding-quality story. |
| 11 | **Trace JSON download** | Single-click gzipped download of the full `CaseTrace` for any case. |
| 12 | **Loop / cycle pre-detection** | Pre-computed NetworkX cycle detection over the transaction graph, surfaced via `/api/network/patterns`. |
| 13 | **Cross-entity behavioral comparison** | Pick two entities, render the two behavioral metrics blocks side by side. |
| 14 | **Per-typology guided tour** | Clicking a typology badge auto-walks through 3 representative cases for that typology. |

### 6.1 Cross-cutting demo "moves"

Scripted demo motions enabled by the functionalities above:

- **The 30-second pitch:** open the dashboard → click a structuring
  alert → cockpit shows the seven phases firing → behavioral summary
  surfaces `vs_declared_volume_ratio: 4.3` → gate passes → SAR
  narrative cites that exact ratio + the 31 USC § 5324 statute →
  analyst dispositions "File SAR".
- **The "look how it grounds" move:** open the behavioral skill
  playground → paste 8 wires of 9,500 USD → model returns metrics
  → switch to the SAR playground → paste the same bundle → SAR
  narrative cites the velocity spike exactly as computed.
- **The leaderboard move:** open the dashboard's Model Comparison
  tile → it loads the pre-compiled four-way report from
  `/api/demo/eval/model_comparison` → audience sees the custom NIM
  win on F1, precision, near-miss specificity, and clean-FPR while
  matching the frontier models on recall.
- **The network move:** click any entity → switch to the Network
  tab → 2-hop graph shows a pass-through (three accounts in series
  with single beneficiary) → click the center node → entity 360
  says shell-company archetype → click "investigate" → end-to-end.

---

## 7. Directory Layout

```
10.appln_buildout/backend/
├── backend.md                              # this file
├── application.md                          # public application overview
├── api_documentation.md                    # full API reference
├── pyproject.toml                          # package metadata, NAT entry point
├── README.md                               # quickstart
├── .env.example                            # NIM endpoint + model name placeholders
│
├── src/                                    # installable Python package + NAT config
│   ├── aml_app/
│   │   ├── __init__.py
│   │   ├── register.py                     # imports tool/skill modules → triggers @register_function
│   │   │
│   │   ├── common/                         # pure-Python helpers (no LLM, no I/O)
│   │   │   ├── schemas.py                  # Pydantic models (single source of truth)
│   │   │   ├── semantic_profile.py
│   │   │   ├── typology_classifier.py
│   │   │   └── behavioral_features.py      # Python computer used at runtime + at SFT time
│   │   │
│   │   ├── tools/                          # leaf tools — data + hints
│   │   │   ├── data_tools.py               # function group aml_data_tools (5 tools)
│   │   │   └── hints.py                    # compute_hints (internal routing only)
│   │   │
│   │   ├── skills/                         # leaf skill calls (one custom_task_nim call each)
│   │   │   ├── aux_call.py                 # aux_*_call leaves (task_type-parameterized)
│   │   │   ├── sar_caller.py               # sar_judgment_caller (7-key build, extra="forbid")
│   │   │   └── prompts.py                  # verbatim SFT-time system prompts
│   │   │
│   │   ├── gating/                         # aux gate
│   │   │   └── aux_gate.py                 # input guard + schema + optional LLM-as-Judge
│   │   │
│   │   ├── workflow/                       # the deterministic orchestrator + trace persistence
│   │   │   ├── investigate_case.py         # the 7-phase async workflow function
│   │   │   └── trace.py                    # CaseTrace dataclass + write_trace()
│   │   │
│   │   ├── api/                            # custom-route handler functions (NAT-registered)
│   │   │   ├── alerts.py                   # /api/alerts/*
│   │   │   ├── entities.py                 # /api/entities/*
│   │   │   ├── network.py                  # /api/network/*
│   │   │   ├── analytics.py                # /api/analytics/*
│   │   │   ├── eval_comparison.py          # /api/demo/eval/model_comparison
│   │   │   ├── misc.py                     # /api/policy/*, /api/sops/*, /api/sanctions/*,
│   │   │   │                               # /api/demo/eval/*, /api/system/*, /api/health
│   │   │   └── skills.py                   # /api/skills/*  (playgrounds)
│   │   │
│   │   └── utils/
│   │       ├── data_loader.py              # singleton Parquet handles, country_risk lookup
│   │       ├── network_graph.py            # NetworkX helpers for graph routes
│   │       └── risk_score.py               # derived risk score formula
│   │
│   ├── configs/
│   │   └── workflow.yaml                   # the NAT workflow YAML (LLMs + functions + front_end)
│   │
│   ├── scripts/
│   │   ├── run_batch.py                    # batch runner over the demo manifest
│   │   ├── score_traces.py                 # offline scorer (traces vs eval_keys)
│   │   ├── compare_endpoints.py            # orchestrate side-by-side runs against N endpoints
│   │   ├── build_model_comparison_report.py
│   │   │                                   # fuse per-endpoint eval JSONs into one N-way report
│   │   ├── seed_baseline_run.sh
│   │   └── start_demo.sh
│   │
│   └── tests/
│
├── data/                                   # local data plane — see §2
│   ├── tool_1_transactions/                ├── tool_4_policy/
│   ├── tool_2_kyc/                         ├── tool_5_sop/
│   ├── tool_3_sanctions/                   ├── demo/
│   ├── seeded_subpopulations/              ├── seed_traces/
│   ├── traces/                             ├── dispositions/
│   └── benchmarks/                         (pre-compiled multi-model reports)
│
└── bkp/                                    # historical backup (do not edit)
```

---

## 8. Tech Stack and Dependencies

| Layer | Choice |
|---|---|
| Orchestration framework | NVIDIA NeMo Agent Toolkit `nvidia-nat` ≥ 1.5.0 |
| Framework extras | `nvidia-nat[langchain, phoenix, opentelemetry, profiler]` |
| LLM serving (trained model) | NVIDIA NIM (vLLM) exposing OpenAI-compatible `/v1/chat/completions` |
| LLM serving (judge, optional) | NIM, default model `nvidia/nemotron-3-nano` |
| Web framework | FastAPI (via NAT's `front_end._type: fastapi`) |
| Server | uvicorn (NAT spawns it) |
| Data handling | pandas + pyarrow for Parquet, rapidfuzz for fuzzy match |
| Graph analytics | networkx |
| Schemas | pydantic v2 (with `extra="forbid"` on the SAR caller's input) |
| Observability backend | Arize Phoenix (local instance, OTLP exporter declared under `general.telemetry`) |
| Profiling backend | NAT profiler module — runtime callbacks + offline forecasting / sizing |
| Testing | pytest + httpx (TestClient) |
| Frontend (separate track) | React/Next.js + Cytoscape.js + Recharts/ECharts + Tailwind |

The backend is packaged as a standard `pyproject.toml` project with
a NAT entry point so all registered components auto-discover at
`nat serve` startup.

---

## 9. Explainability, Observability, and Profiling

| Layer | Question it answers | Source |
|---|---|---|
| **Explainability** | "Why did the agent say this about THIS case?" | Custom — our `CaseTrace` JSON |
| **Observability** | "What did the runtime actually do, when, and how long did each step take?" | NAT-native — OpenTelemetry + Arize Phoenix |
| **Profiling** | "Where is time and where are tokens being spent across runs?" | NAT-native — `nvidia-nat-profiler` |

### 9.1 Explainability — per-case `CaseTrace`

At the end of every investigation the workflow writes one
`CaseTrace` record to `./data/traces/{case_id}.json`. The trace is
the **single audit artifact** the Investigation Cockpit, replay
view, aux-gate inspector, eval scorer, and disposition workflow
all read from. It carries:

- **Identity**: `case_id`, `alert_id`, `entity_id`, `started_at`,
  `finished_at`, `wall_clock_ms`.
- **Phase-1 inputs**: `transactions`, `kyc_profile`,
  `sanctions_pep_hits`.
- **Phase-2 internal routing artifacts** (audit-only): the
  rule-layer typology guess + the computed `semantic_profile` +
  `activity_descriptor` derived by `compute_hints`. These are
  recorded for the cockpit to display but are NOT passed into the
  SAR call.
- **Phase-3 retrieval outputs**: `policy_excerpts` (focused +
  broad), `sop_excerpts`.
- **Phase-4 aux outputs**: raw per-skill responses
  (`aux_responses_raw`) plus the deterministic Python-computed
  behavioral metrics.
- **Phase-5 gate audit**: every `aux_gate` decision with its reason
  (input-guard / schema / judge verdict + judge explain) and the
  final `auxiliary_findings` that were inlined into the SAR bundle.
- **Phase-6 output**: the exact 7-key user-message JSON sent to
  `sar_judgment_caller`, the model's raw text, the parsed
  `{is_suspicious, suspicious_activity_report}`, and any parse
  error.
- **Metadata**: whether the judge was enabled, any top-level error.

Two things are deliberately NOT persisted: the policy_chunks
corpus body verbatim (only the per-case excerpts actually
retrieved), and the judge's raw text (only `{verdict, explain}`).

The trace is what the frontend renders in:

- **Investigation Cockpit** — full reasoning chain timeline
- **Aux-gate inspector** — per-finding USE/DROP table with judge rationale
- **Trace JSON download** — gzipped one-click export
- **Side-by-side SAR comparison** — two traces rendered as two columns
- **`/api/demo/eval`** — scored against `eval_keys.jsonl`

### 9.2 Observability — NAT-native distributed tracing

The workflow has NAT's OpenTelemetry instrumentation enabled with
an **Arize Phoenix** exporter pointed at a local Phoenix instance,
declared under `general.telemetry` in the workflow YAML. NAT emits
a span for every NAT primitive automatically — no custom
instrumentation in our function bodies. Spans cover the
`investigate_case` invocation, every leaf data-tool call, every
aux-skill LLM call, the gate, the SAR call, and the judge calls
(when enabled).

Spans are linked into a single trace per case via the
NAT-propagated context, so Phoenix renders the entire
investigation as a hierarchical call tree. The same OTel pipeline
can also export to any other OTLP-compatible collector
(LangSmith, Langfuse, Dynatrace, W&B Weave) by adding another
exporter under `general.telemetry.exporters` — no code changes
required.

### 9.3 Profiling — NAT profiler

The backend installs `nvidia-nat[profiler]` and enables the
profiler module on the workflow, so every workflow run
automatically collects per-invocation usage statistics: tokens-in /
tokens-out per LLM call, latency per leaf, time gaps between
calls, prompt-prefix repetition for cache-friendliness, concurrency
spikes, and tail-latency outliers. The profiler stores raw runtime
statistics with the case trace metadata so they can be re-analyzed
offline.

What the profiler gives us for the workshop:

- **Per-component latency breakdown** — surfaced through the
  analytics dashboard's "case-latency breakdown" view.
- **Token-efficiency view** — prompt vs. completion tokens per LLM
  binding (`custom_task_nim`, `judge_llm`).
- **Workflow forecasting** — fits time-series models on the
  collected statistics to predict latency / token budget for
  unseen cases.
- **Sizing guidance** — `nat sizing-calc` consumes the profiler
  output to estimate concurrent-user capacity for a given hardware
  budget.

Profiling output is rendered into the Analytics Dashboard's
"agent performance" panel and is available raw via
`/api/analytics/profile`.

---

## 10. Configuration and Environment

The backend is configured through two artifacts: an `.env` file
that holds endpoint locations and secrets, and a workflow YAML
that wires up LLMs, functions, and the workflow.

### 10.1 Environment variables

| Variable | Purpose |
|---|---|
| `NAT_AML_NIM_BASE_URL` | Base URL of the custom-task NIM (e.g. `http://localhost:8088/v1`). |
| `NAT_AML_NIM_MODEL` | Model name on that NIM (e.g. `aml-custom-task-nim-1`). |
| `NAT_AML_NIM_API_KEY` | Bearer token if the NIM enforces auth (default `EMPTY`). |
| `NAT_AML_JUDGE_BASE_URL` / `_MODEL` / `_API_KEY` | Same three knobs for the optional judge LLM. |
| `NAT_AML_ENABLE_JUDGE` | `true` / `false` (default `false`). Toggles the LLM-as-Judge stage in `aux_gate`. |
| `NAT_AML_BEHAVIORAL_MODE` | `python_only` (default). Reserved knob — alternative modes that route the behavioral block through the LLM are off in production. |
| `NAT_AML_NO_THINK` | `1` / `0` (default `0`). When `1`, prepends `/no_think` to system prompts so reasoning-capable Nemotron variants emit JSON directly without verbose chain-of-thought. |
| `NAT_AML_SAR_MAX_TOKENS` / `NAT_AML_AUX_MAX_TOKENS` | Per-call output budgets. Defaults are conservative; raise when targeting reasoning-heavy endpoints (e.g. frontier models with native thinking). |
| `NAT_AML_DATA_DIR` | Absolute path to the local data plane (default `./data`). |
| `NAT_AML_HOST` / `NAT_AML_PORT` | uvicorn binding (defaults `0.0.0.0:8000`). |
| `NAT_AML_EVAL_TOKEN` | Bearer token required by every `/api/demo/eval*` route when set. Leave empty in dev to disable the gate. |

### 10.2 Workflow YAML wiring

`configs/workflow.yaml` declares, in this order:

- **`general.front_end`** — `_type: fastapi` with streaming + CORS
  enabled, the root POST path bound to `/api/investigation/run`,
  and the full custom-endpoint list from §5 declared under
  `endpoints:`. Each endpoint maps a route to a registered
  function name.
- **`general.telemetry`** — the Arize Phoenix OTLP exporter (§9.2).
- **`general.profiler`** — the NAT profiler module (§9.3).
- **`llms`** — two entries pinned to `temperature: 0.0`:
  - `custom_task_nim` — the trained checkpoint, used by every
    auxiliary skill call and by the final SAR call.
  - `judge_llm` — used inside `aux_gate` when the LLM-as-Judge
    stage is enabled.
- **`function_groups.aml_data_tools`** — bundles the five data
  tools under one config block with a shared `data_dir`.
- **`functions`** — declares the leaf tools (`compute_hints`, the
  four `aux_*_call` task-typed wrappers, `aux_gate`,
  `sar_judgment_caller`), the handler functions for every
  `/api/*` route from §5 (including
  `demo_eval_model_comparison_report` for the pre-compiled N-way
  comparison endpoint), and the skill playgrounds.
- **`workflow`** — root `_type: investigate_case` — the
  deterministic 7-phase workflow.

---

## 11. Benchmark Snapshot

The current pre-compiled comparison served by
`/api/demo/eval/model_comparison` is the **464-case intersection
report** (cases where all four endpoints produced a clean parse,
filtered from a 500-case prod-mimic v2 eval set). Path A backend on
every endpoint (Python-deterministic `auxiliary_behavioral`, 7-key
SAR bundle).

| Metric | aml-custom-task-nim (SFT) | nemotron-3-nano (base) | gemma-4-31b-it (frontier) | openai/gpt-5.2 (frontier) |
|---|---:|---:|---:|---:|
| F1 | **0.698** | 0.465 | 0.398 | 0.240 |
| Precision | **0.587** | 0.321 | 0.257 | 0.142 |
| Recall | 0.860 | 0.837 | **0.884** | 0.791 |
| Near-miss specificity | **0.743** | 0.371 | 0.486 | 0.200 |
| Clean-cohort FPR | **0.044** | 0.140 | 0.238 | 0.461 |
| Confusion (TP/FP/TN/FN) | 37 / **26** / **395** / 6 | 36 / 76 / 345 / 7 | 38 / 110 / 311 / 5 | 34 / **206** / 215 / 9 |

The custom-trained NIM wins every metric except recall (where Gemma
nudges ahead at 0.884 vs 0.860 — but at 4× the false-positive cost,
110 FP vs 26 FP). Most striking: GPT-5.2 flags 52% of all cases as
suspicious, vs the ground-truth positive rate of 9.3% — a frontier
general-purpose model is uniformly mis-calibrated for AML triage.
The custom NIM flags 14% of cases, the closest of any endpoint to
the true rate.

The full report (per-typology breakdowns, per-endpoint narrative
length, latency stats, full confusion matrices) is served verbatim
by the endpoint and stored at:

```
data/benchmarks/four_way_464case_recovered_<timestamp>.json
data/benchmarks/latest.json   (pointer)
```

---

## 12. Known Issues

### 12.1 NAT 1.5 GET handler doesn't bind path / query params

Endpoints declared as `method: GET` with a non-`Empty` Pydantic input
schema receive `args = None` instead of a populated input model. This
affects 9 routes:

| Route | Impact |
|---|---|
| `GET /api/alerts/{alert_id}` | Alert detail view broken |
| `GET /api/entities/{entity_id}` | Entity 360 broken |
| `GET /api/entities/{entity_id}/transactions` | Entity tx tab broken |
| `GET /api/entities/{entity_id}/behavioral_summary` | Entity behavioral tab |
| `GET /api/entities/{entity_id}/risk_score` | Entity risk score |
| `GET /api/entities/{entity_id}/network` | Entity network graph |
| `GET /api/entities/{entity_id}/timeline` | Entity timeline |
| `GET /api/sops/{sop_id}` | SOP detail |
| `GET /api/investigation/{case_id}` | Trace retrieval |

**Workaround patterns** (any one works):

- *Server-side*: convert the route to `method: POST` and pass the
  identifier in the JSON body (works because POST routes parse the
  body into the input model normally).
- *Server-side*: replace path-param routes with query-string routes,
  but the underlying NAT bug affects query params too — so the
  function body would also need to read `os.environ.get(...)` or pull
  from a context the frontend pre-populates. Less clean.
- *Wait for upstream fix*: NAT 1.5's
  `front_ends/fastapi/routes/common_utils.py:get_single_endpoint`
  hard-codes `None` for input. A future NAT version is likely to bind
  path/query params; until then, GET routes with non-Empty schemas
  silently fail.

Listing routes (e.g. `GET /api/alerts?status=open`) work because
their input schemas use only `Optional` fields with defaults — the
function bodies defensively default-initialise on `None`. See
`alerts.py::list_alerts` for the pattern.

**Demo impact**: the workshop's headline flow
(`POST /api/investigation/run`, `POST /api/demo/eval/model_comparison`)
is unaffected. The Entity 360 / Alert Detail UI tiles need the
workaround above to function.

### 12.2 Skill playgrounds require the custom NIM at port 8088

The five `/api/skills/*` playground routes invoke `custom_task_nim`
(default `http://localhost:8088/v1`). If that NIM is not running,
all five routes return `"Cannot connect to host localhost:8088"`.
Spin up the NIM container before testing the playgrounds.

### 12.3 Route ordering in workflow.yaml

`/api/alerts/stats` must appear before `/api/alerts/{alert_id}` in
the `endpoints:` list; otherwise FastAPI matches `stats` as a value
for `{alert_id}`. The current YAML has the correct order, but any
future addition of routes under `/api/alerts/` must respect this
constraint.

