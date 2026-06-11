# AML Investigator — Frontend Functionality Guide

What the UI does, screen by screen, widget by widget. Aimed at attendees who'll click around the running app rather than read code.

The frontend is **Next.js 14 (App Router)**, served at `localhost:3000`. All `/api/*` traffic is proxied to the **NAT backend at `localhost:8010`**, which orchestrates the **Custom Task NIM at `localhost:8088`**. The model the NIM serves is the AML SFT/RL checkpoint produced by the upstream training pipeline.

---

## Table of contents

1. [Global UI scaffolding](#1-global-ui-scaffolding)
2. [Dashboard — Analytics overview](#2-dashboard--analytics-overview)
3. [Alert Queue — Triage list](#3-alert-queue--triage-list)
4. [Investigation Cockpit — Per-case deep dive](#4-investigation-cockpit--per-case-deep-dive)
5. [Entity 360 — KYC + behavior + network](#5-entity-360--kyc--behavior--network)
6. [Compliance Tools — Policy / SOP / Sanctions](#6-compliance-tools--policy--sop--sanctions)
7. [Skill Playgrounds — Single-call model probes](#7-skill-playgrounds--single-call-model-probes)
8. [Model Comparison — Ad-hoc two-run scorecard](#8-model-comparison--ad-hoc-two-run-scorecard)
9. [Leaderboard — Pre-compiled N-way benchmark](#9-leaderboard--pre-compiled-n-way-benchmark)
10. [End-to-end demo flow](#10-end-to-end-demo-flow)

Appendices: [page → API map](#appendix--page--api-map), [colour key](#appendix--colour-key), [graceful-degradation behaviour](#appendix--graceful-degradation).

---

## 1. Global UI scaffolding

Persistent on every page.

### 1.1 Left sidebar (8 entries)

| Entry | Purpose |
|---|---|
| **Dashboard** | Top-of-funnel analytics — typologies, transactions, jurisdictions, aux-gate stats, per-typology accuracy. |
| **Alert Queue** | The analyst's worklist. Every alert from `manifest.jsonl`, with status filters and a one-click "Run investigation" trigger per row. |
| **Investigation Cockpit** | Per-case deep dive: 7-phase trace, all 4 auxiliary findings, gate verdicts, SAR narrative, disposition form. |
| **Entity 360** | Customer-book search. Per-entity profile with KYC, transactions, behavioral metrics, counterparty graph. |
| **Compliance Tools** | Standalone access to the same policy RAG, SOP library, and sanctions screen the workflow uses internally. |
| **Skill Playgrounds** | Call one auxiliary skill (or the final SAR call) directly — no orchestrator. Showcases the trained model in isolation. |
| **Model Comparison** | Ad-hoc scorecard for any two trace snapshots against the same ground truth. |
| **Leaderboard** | Pre-compiled N-way benchmark — custom NIM vs base Nemotron vs frontier models. |

Footer pill shows backend status and the model name wired through the proxy.

### 1.2 Top bar

Polls `/api/health` every 30 s. Pill flips red if the backend stops responding. The live `n_entities · n_transactions` counter confirms the data plane is healthy. Theme toggle on the right.

### 1.3 Visual language

- **NVIDIA-green `#76b900`** = primary / selected / winning / "USED" outcomes.
- **Risk colours** — `low` emerald, `medium` amber, `high` orange, `enhanced` red, `prohibited` deep red.
- **Status chips** — `Open` sky, `In progress` amber, `Closed` emerald.
- **Typology hues** — every typology has a fixed colour so cross-screen recognition is consistent: structuring=`#76b900`, layering=`#06b6d4`, smurfing=`#22c55e`, shell_company=`#f97316`, trade_based_ml=`#a855f7`, terrorist_financing=`#b91c1c`, human_trafficking=`#ef4444`, elder_exploitation=`#eab308`.

---

## 2. Dashboard — Analytics overview

**Route:** `/dashboard` · **Backend:** `/api/analytics/*`

The "open the app and look smart" landing page. Aggregate-only — no per-case detail, safe to leave open during a demo.

### 2.1 KPI strip — four headline tiles

| Tile | Question it answers | What it tells you |
|---|---|---|
| **Total Alerts** | "How big is the queue?" | Live total from `manifest.jsonl`. Sub-line breaks it into open / in-progress / closed (status derived from the presence of a trace / disposition file). |
| **Entities under monitoring** | "How big is the customer book?" | Distinct entity_ids loaded into the KYC store. |
| **Transactions ingested** | "How much money-movement data is the agent reasoning over?" | Total tx rows on disk, plus a sub-line with the count of SARs the agent has drafted. |
| **Avg. case latency** | "Is the pipeline fast enough to deploy?" | Mean wall-clock per case across all persisted traces. ~7 s on the demo box. |

### 2.2 Typology distribution (donut)

**What it shows:** alerts grouped by typology — coloured by the canonical typology hues.

**Graceful degradation:** if fewer than 5 traces have been persisted, the donut falls back to the **ground-truth manifest** so the slice is meaningful out-of-the-box. Subtitle changes to "Ground-truth manifest" so the reader knows they're not yet looking at the agent's own inferences. Once at least 5 traces exist, the subtitle flips to "Inferred from persisted case traces" and the data switches to the agent's typology hypotheses.

**Reading it:** the dominant slice is the typology you should expect the agent to be best at. On a clean install you see 8 well-balanced slices plus a "none" (clean negatives) bucket — that's the demo manifest's stratification.

### 2.3 Transaction volume (daily line)

Per-day count of transactions ingested. Spikes correspond to the demo corpus's seeded patterns; a flat line means the data plane is healthy.

### 2.4 Alerts by risk rating (bar)

Distribution of monitored entities by KYC risk band. A normal book is heavy on `low`; a high `enhanced`/`prohibited` count means the workshop has loaded a stressed cohort.

### 2.5 Channel mix per typology (stacked bar)

For each canonical typology, the percentage of activity through each channel (`cash / wire / ach / card / cheque / crypto`). Internal `near_miss::*` sidecar tags from the ground-truth corpus are filtered out — they're labelling artefacts, not real typologies, and used to be the visual noise you saw before.

**Reading it:**
- `structuring` → cash-heavy (sub-CTR deposits).
- `layering`, `trade_based_ml` → wire-heavy.
- `shell_company` → mixed wire + ach.
- `smurfing` → distributed across small-value channels.

This is the most analyst-actionable chart on the dashboard — operational fingerprint per typology.

### 2.6 Top counterparties (table)

Top 10 counterparties by aggregate USD flow across the whole corpus. Hubs in the global money-flow graph. A counterparty that appears here **and** in a flagged case is a strong network-overlap signal worth investigating.

### 2.7 Aux-gate decisions (horizontal stacked bar)

For each auxiliary specialist (behavioral / numeric / citation / statutory), how often the finding was **USED** (kept) vs **DROPPED** (rejected) by the gate across all persisted traces.

**Reading it:**
- `behavioral` is always USED — it's deterministic Python and never gated out.
- `numeric` and `citation` have moderate USED rates.
- `statutory` is the strictest — many findings fail the entailment/contradiction Pydantic schema and get dropped.
- A high DROPPED rate for a skill is a regression signal — the trained model is producing low-quality output for that task type.

### 2.8 Per-typology performance (table)

TP / FP / FN / TN per typology against `eval_keys.jsonl`, with recall and precision.

**Graceful degradation:** if fewer than 5 traces exist, the table renders an empty-state message with a link to `/alerts` and a hint to run `scripts/run_batch.py` for bulk population. Once enough cases are scored, the full table appears with one row per typology and the "none" bucket filtered out (it's not a real typology).

**Reading it:** this is the numeric proof of fine-tuning value — read structuring's recall (typically high) and statutory's clean-FPR (the hard metric for off-the-shelf models).

---

## 3. Alert Queue — Triage list

**Route:** `/alerts` · **Backend:** `/api/alerts`, `/api/alerts/stats`, `/api/investigation/run`

The analyst's daily workbench.

### 3.1 KPI strip

Total / Open / In progress / Closed. Status is derived live:
- **Open** — no trace file exists yet for this case.
- **In progress** — trace exists but no analyst disposition.
- **Closed** — disposition file exists (`file_sar / dismiss / escalate`).

### 3.2 Search + filter row

- Free-text `q` matches `trigger_summary`, `entity_id`, and `alert_id` (case-insensitive substring).
- Status dropdown filters to open / in-progress / closed.

### 3.3 Alerts table

| Column | Meaning |
|---|---|
| **Case** | `DEMO_NNNN` id — clicks through to the cockpit. |
| **Alert** | Upstream monitoring system's alert id. |
| **Entity** | Subject entity — clicks through to Entity 360. |
| **Trigger** | The rule summary that generated the alert. |
| **Window** | Investigation window — date range the agent must scope to. |
| **Status** | Open / In progress / Closed chip. |
| **Actions** | For Open alerts: a green **Run** button. One click → `POST /api/investigation/run` for that case. After ~10–30 s the row flips to "In progress" and you can drill in. |

### 3.4 Pagination + typology mini-breakdown

Bottom row shows page-N-of-M and a colour-coded chip list of typology counts from persisted traces. Quick at-a-glance view of what's been investigated.

---

## 4. Investigation Cockpit — Per-case deep dive

**Route:** `/cockpit/{caseId}` · **Backend:** POST workaround routes for path-param GETs (`/api/alerts/get`, `/api/investigation/get`), plus `/api/alerts/{id}/disposition` and `/api/investigation/run`.

Single most important screen — it defends the agent's verdict and lets the analyst close the case. Two columns: left = reasoning chain, right = evidence + disposition.

### 4.1 Page header

Case id in mono green · typology chip (the rule-layer hypothesis) · **SUSPICIOUS / NOT SUSPICIOUS** chip (the agent's final verdict) · **Trace JSON** download button (gzipped audit export) · trigger summary, alert id, entity link, investigation window.

### 4.2 No-trace state

If the alert has never been investigated, the cockpit shows a single panel with the trigger and a **Run investigation** button. One click fires the workflow; the page auto-refreshes when the trace is written. Same effect as the green Run button on the queue, but inline.

### 4.3 Left column — the reasoning chain

#### Trigger header (3 mini-tiles)
- **Activity descriptor** — one-line Python summary of the entity's behaviour (e.g. "6 sub-$10K cash deposits in 8 days").
- **Typology (internal routing)** — rule-layer Phase 2 hypothesis. Note the "internal routing" label — this is NOT sent to the LLM.
- **Wall clock** — total end-to-end latency for this case.

#### Deterministic 7-phase workflow (timeline)
A vertical green rail listing the seven phases and one detail per phase: Data fetch → Typology guess → Retrieval → Aux skills → Aux gate → SAR judgment → Trace persisted. Each phase shows a concrete number (n_tx, n_excerpts, n_findings, etc.) so you can verify each phase actually produced something.

#### Orchestrator tool calls
For deterministic workflows (the workshop default), this panel explains why no LLM-driven tool calls were emitted. If you switch to an agentic workflow later, this is where the per-tool call sequence renders.

#### Auxiliary specialist findings (tabbed)
Four tabs — Behavioral, Numeric, Citation, Statutory.

- **Behavioral** — text summary + the typed `BehavioralMetrics` block (channel_mix, velocity, z-score, vs_declared_volume_ratio, loop_detected). Highlights if velocity / volume-ratio is anomalous.
- **Numeric** — the question, the answer, and the step-by-step calculation. Verifiable by hand against the transactions panel.
- **Citation** — the question, the answer, and an italicised evidence span lifted verbatim from a retrieved policy excerpt.
- **Statutory** — `entailment / contradiction / neutral` chip + reasoning. Tells the analyst whether the conduct meets the statute's elements.

This is the cockpit's headline moment — the model showed its work, with grounded evidence per skill.

#### Aux-gate inspector (table)
For each specialist: USED / DROPPED chip, reason (`schema_ok`, `judge: PASS`, `schema_failure`, `input_missing`), reviewer verdict + explanation. Tells the analyst which findings made it into the final SAR bundle and why others were dropped.

#### SAR narrative
The agent's final output. Subtitle distinguishes SUSPICIOUS_ACTIVITY_REPORT vs VERDICT framing. The narrative is 250-800 chars, grounded in evidence, with regulatory framing. **"View raw bundle"** toggle reveals:
- `user_message` — the exact 7-key JSON sent to the NIM.
- `raw_text` — the model's raw completion before parsing.

If parsing failed an amber "parse error" chip appears in the header.

### 4.4 Right column — evidence + disposition

#### Case summary
- KYC snippet: type / risk rating (colour-coded) / jurisdiction / business purpose. Entity id links to Entity 360.
- Sanctions / PEP hits with match-score percentages.
- Transaction snapshot: count + total USD.

#### Policy & SOP excerpts
What the agent grounded its narrative against — up to 4 policy excerpts (source + section + 3-line preview) and 2 SOP excerpts.

#### Analyst disposition
Three buttons — **File SAR / Dismiss / Escalate** — plus a rationale textarea. Pressing **File** posts to `/api/alerts/{id}/disposition` and persists to `./data/dispositions/{case_id}.json`. The alert flips to **Closed**. If a disposition already exists the form is pre-filled with a footer line showing when and what was previously recorded.

---

## 5. Entity 360 — KYC + behavior + network

**Route:** `/entities` (list) and `/entities/{entityId}` (profile).

### 5.1 List page

Filters: free-text (`entity_id` / `business_purpose`), risk rating, entity type. 25 rows per page. Hover any row to highlight.

### 5.2 Profile — KPI strip

| Tile | Meaning |
|---|---|
| **Transactions** | Total tx count + unique counterparties. |
| **Risk score** | 0–100 weighted blend (KYC × tx × sanctions × country). Red ≥70, amber 50–70, green <50. |
| **Expected monthly** | Declared volume from KYC. |
| **Related alerts** | Alerts where this entity is the subject. |

### 5.3 Profile — Tabs

**Overview** — KYC profile card, channel-mix bar, daily-volume line, related-alerts table. Each related alert clicks through to its cockpit.

**Transactions** — latest 200 tx (Date / Counterparty / Channel chip / Amount / Notes). The backend strips internal ground-truth columns before responding — analysts never see typology labels.

**Behavioral** — the deterministic Python-computed `BehavioralMetrics` block — same schema the trained model's `auxiliary_behavioral` task produces:
- `tx_count`, `tx_total_usd`
- `velocity_24h_max`, `velocity_24h_avg_30d`
- `unique_counterparties_7d`
- `amount_z_score_max`
- `country_risk_max`
- `vs_declared_volume_ratio` — **highlighted ring** when >2× declared (key signal for structuring / layering)
- `loop_detected` — **highlighted** when true (round-tripping flag)
- `channel_mix` — pill breakdown.

Below: Risk-score components — the actual weighted inputs that produced the 0-100 score above.

**Network** — force-style SVG counterparty graph at depth=2. Central green node = the entity. Counterparties fan around it in a ring. Heavy edges (top 5 by USD volume) are bright green; light edges dimmed. Footer shows `n_nodes · n_edges`. Use it to spot pass-through chains and shared-counterparty rings.

---

## 6. Compliance Tools — Policy / SOP / Sanctions

**Route:** `/tools` · Three tabs.

### 6.1 Policy RAG tab

Left panel: corpus distribution (FinCEN / OFAC / FFIEC / FATF chunk counts).
Right panel: stratified top-k search.

Inputs: typology dropdown · keyword input · k (default 4). Returns up to k excerpts **stratified across the four source enums** so a single source can't dominate. Each excerpt shows source, section, optional URL, and the chunk body verbatim.

Use case: analyst's instinct says "this looks like layering" — they pull the same regulatory shelf the agent uses, in the same order, before reading the cockpit.

### 6.2 SOP browser tab

Left panel: every typology SOP (`SOP-STRUCTURING-01`, `SOP-LAYERING-01`, etc.) with section-heading previews.
Right panel: full SOP body rendered as structured Markdown when clicked, with sections (`Investigation Steps`, `Escalation Criteria`, `Filing Decision`, …) in NVIDIA green.

Use case: explain what the agent was instructed to do for each typology, before showing what it actually did in the cockpit.

### 6.3 Sanctions screen tab

Free-form OFAC + PEP fuzzy screen. Inputs: name (required), country (optional — boosts score 5 % on match), min match score slider (default 0.55). Returns hits with list (`OFAC` / `OpenSanctions`) and match score, colour-coded ≥85 red, ≥70 amber, below grey.

Use case: ad-hoc "is this name on a list?" check outside the per-case flow.

---

## 7. Skill Playgrounds — Single-call model probes

**Route:** `/skills` · Five tabs, one per `task_type` the model was trained on.

### 7.1 Behavioral / Numeric / Citation

Two-column layout. Input column has an optional question (numeric / citation only) and a pre-filled passage textarea. **Run skill** posts to `/api/skills/{tab}`.
Output column renders the typed finding:
- **Behavioral** — summary text + the metrics JSON block.
- **Numeric** — answer + step-by-step calculation.
- **Citation** — answer + italicised evidence span.

### 7.2 Statutory

**Two-field input** — Statute textarea + Fact-pattern textarea + question. The backend's `SkillStatutoryInput` requires them as separate fields. Output: entailment/contradiction/neutral chip + reasoning.

### 7.3 Final SAR

A JSON textarea pre-filled with the 6-key evidence bundle (`transactions, kyc_profile, sanctions_pep_hits, policy_excerpts, sop_excerpts, auxiliary_findings`). Pydantic `extra="forbid"` rejects anything else; the backend prepends `task_type="sar_judgment"` server-side. Run → renders `{is_suspicious, suspicious_activity_report}` plus the raw text and the exact user_message that was sent.

Use case: edit a bundle in-place and re-run to see the model's sensitivity to a single piece of evidence (drop the citation excerpt and watch the narrative degrade).

---

## 8. Model Comparison — Ad-hoc two-run scorecard

**Route:** `/compare` · **Backend:** `/api/demo/eval/runs`, `/api/demo/eval/compare`.

Pick any two trace-snapshot directories (anything starting with `traces_` under `backend/data/`); the page scores both against the same `eval_keys.jsonl`.

**Ground truth bar** — total cases / SAR positives / no-SAR negatives / near-miss negatives.

**Per-run cards (side-by-side)** — confusion matrix (TP/FP/TN/FN, with FP broken into near_miss / clean), the full metrics table (accuracy, precision, recall, F1, macro F1, FPR clean, FPR near-miss, near-miss specificity, grounding rate), and the latency footer.

**Diff matrix (run A − run B)** — F1 / Precision / Recall / Macro F1 deltas with up/down arrows. Green-ringed if the change favours run A on that metric's preferred direction. Bottom row carries ΔTP / ΔFP / ΔFN / ΔTN / Δlatency / ΔFP-by-type.

Use case: snapshot the active traces, run a fine-tune, rerun the manifest, and quantify the new model.

---

## 9. Leaderboard — Pre-compiled N-way benchmark

**Route:** `/leaderboard` · **Backend:** `/api/demo/eval/model_comparison`.

The workshop's headline tile.

| Section | What's in it |
|---|---|
| **Report header** | Filename, ran-at timestamp, demo size + version, explanatory note. |
| **Headline metrics** | F1 / precision / recall / near-miss specificity / clean-FPR per endpoint. Winning column per row in bold green. Each metric label carries `↑ better` / `↓ better`. |
| **Confusion matrix** | TP / FP / TN / FN per endpoint (custom NIM, base Nemotron, frontier models). |
| **Latency** | Mean and median wall-clock ms — proves the custom NIM is faster *and* more accurate. |
| **Per-typology recall** | One row per typology, endpoints as columns. Per-row winner highlighted. |

Loads instantly because the report is a flat JSON file under `data/benchmarks/` — no LLM at request time.

**Not-available state:** if `data/benchmarks/latest.json` doesn't exist yet, the page renders the three-command offline-pipeline recipe instead of a blank panel — the demo never breaks.

---

## 10. End-to-end demo flow

Scripted ~5-minute walkthrough using only the UI:

1. **`/dashboard`** — set the scene. "Here's the volume: 194 alerts, 2K entities, 71K transactions. The donut shows the typology mix; structuring and layering dominate. Per-typology table at the bottom shows the agent's recall against ground truth."

2. **`/alerts`** — pick an open structuring alert (search "wire velocity" or "cash deposit"). Click the green **Run** button on its row.

3. **Wait 10–30 s** — the row flips Open → In progress.

4. **Click into the cockpit** — walk down the left column:
   - 7-phase rail (workflow actually ran)
   - Behavioral metrics — point at `vs_declared_volume_ratio: 4.3` (highlighted ring)
   - Numeric tab — the model summed the deposits and computed monthly rate
   - Citation tab — verbatim FinCEN advisory text
   - Statutory tab — `entailment` chip on 31 USC §5324
   - Aux-gate inspector — all four findings USED
   - SAR narrative — cites the velocity spike and the statute exactly.

5. **Toggle "View raw bundle"** — show the 7-key user message and the 2-field response.

6. **File the disposition** — File SAR, one-line rationale, click File. Status flips to **Closed**.

7. **`/entities/SYN_...`** — same entity. Behavioral + Network tabs prove what the model was reasoning over.

8. **`/leaderboard`** — the punch line: custom NIM wins F1, precision, near-miss specificity, clean-FPR vs base Nemotron + Gemma + GPT-5.2, at ~3× lower latency.

9. **`/skills`** — open Numeric, edit the passage (drop one deposit), rerun. The model is grounded in the data, not pattern-matching the prompt.

10. **`/tools`** — show the SOP and policy RAG. "These are the exact retrievers the agent uses — analyst can probe them by hand."

---

## Appendix — Page → API map

| Page | Backend endpoints |
|---|---|
| `/dashboard` | `/api/analytics/{overview, typology_distribution, risk_heatmap, timeline, channel_mix, top_counterparties, aux_usage, agent_performance}` |
| `/alerts` | `/api/alerts`, `/api/alerts/stats`, `/api/investigation/run` |
| `/cockpit` | `/api/alerts?limit=12` |
| `/cockpit/{caseId}` | `POST /api/alerts/get`, `POST /api/investigation/get`, `POST /api/alerts/{id}/disposition`, `POST /api/investigation/run` |
| `/entities` | `/api/entities` (filters: q, risk_rating, entity_type, jurisdiction) |
| `/entities/{entityId}` | `POST /api/entities/{get, transactions, behavioral_summary, risk_score, network, timeline}` |
| `/tools` — Policy | `/api/policy/sources`, `POST /api/policy/search` |
| `/tools` — SOPs | `/api/sops`, `POST /api/sops/get` |
| `/tools` — Sanctions | `POST /api/sanctions/screen` |
| `/skills` | `POST /api/skills/{behavioral, numeric, citation, statutory, sar}` |
| `/compare` | `/api/demo/eval/runs`, `POST /api/demo/eval/compare` |
| `/leaderboard` | `POST /api/demo/eval/model_comparison` |
| top bar | `/api/health` (polled every 30 s) |

> **Why POSTs for what used to be GETs:** the documented NAT 1.5 GET handler doesn't bind path or query parameters to the function input (`backend.md §12.1`). The backend exposes both — `GET /api/entities/{id}` and `POST /api/entities/get` — and the frontend uses the POST variant for reliability. The NAT framework itself is unchanged; only the workflow.yaml carries a few extra route entries.

---

## Appendix — Colour key

| Colour | Meaning |
|---|---|
| **`#76b900` (NVIDIA green)** | Primary / selected / winning / USED / SUSPICIOUS_ACTIVITY_REPORT / `entailment` |
| **Rose** | Bad / DROPPED / parse error / contradiction / failure |
| **Amber** | Warning / `near-miss` / in-progress / parse-issue chip |
| **Sky** | Informational / open alert |
| **Emerald** | Good — closed alert / low risk / `entailment` chip |
| **Risk band** | `low` emerald · `medium` amber · `high` orange · `enhanced` red · `prohibited` deep red |
| **Typology hues** | Stable across all screens — see §1.3 |

---

## Appendix — Graceful degradation

Behaviours when the system has limited data:

| State | What the UI does |
|---|---|
| **Backend down** | Top-bar pill turns red ("backend offline"). All SWR fetches fail quickly; pages render their loading spinners. |
| **NIM (:8088) down** | Skill calls and `/api/investigation/run` return a body-level `ClientConnectorError`. Pages render the error in their output panels rather than crashing. |
| **No traces written yet (<5)** | Dashboard donut falls back to the seeded ground-truth distribution. Per-typology performance table shows an empty-state CTA pointing at `/alerts`. |
| **No benchmarks file** | `/leaderboard` renders the offline-pipeline recipe instead of blank. |
| **No second snapshot dir** | `/compare` shows the run-pickers but the **Compare** button is disabled. |
| **Trace JSON parse error** | Cockpit shows an amber "parse error" chip in the SAR header; the narrative panel still renders best-effort text. |
