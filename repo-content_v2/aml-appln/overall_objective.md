# AML Investigation Agent — Workshop Walk-through

> An end-to-end, functional walk-through of the demo application: what it
> does, how it was trained, and how to read its outputs at the workshop.
>
> Each section answers one question. Code paths are pointers, not the
> point — the goal is *what* and *why*, not *how every byte is wired*.
>
> **Companion documents** (read these only if you want to go deeper):
>
> - [`training_strategy.md`](./training_strategy.md) — the full plan-of-record
> - [`1.data_download/`](./1.data_download/), [`2.data_processing/`](./2.data_processing/) — data acquisition + extraction
> - [`3.cpt/`](./3.cpt/) — Continued Pre-Training
> - [`4.sdg_sft/`](./4.sdg_sft/), [`7.run_sft/`](./7.run_sft/) — SFT data + training
> - [`10.appln_buildout/backend/`](./10.appln_buildout/backend/) — the agentic application

---

## Table of Contents

**WHAT**
1. [Problem Statement](#1-problem-statement)
2. [Tools and Custom Skills — the *Why*](#2-tools-and-custom-skills--the-why)
3. [Target Application — Trigger → Tools → Skills → SAR](#3-target-application--trigger--tools--skills--sar)

**HOW**
4. [Data Strategy — sources and significance](#4-data-strategy--sources-and-significance)
5. [Data Curation — stages of the pipeline](#5-data-curation--stages-of-the-pipeline)
6. [CPT Phase — Level 1 broad register, Level 2 AML with replay](#6-cpt-phase--level-1-broad-register-level-2-aml-with-replay)
7. [Synthetic Data for SFT — target distribution](#7-synthetic-data-for-sft--target-distribution)
8. [SFT Phase — shuffle, analyse, filter, train](#8-sft-phase--shuffle-analyse-filter-train)

---

# WHAT

## 1. Problem Statement

Banks and financial institutions receive **millions of transaction-monitoring
alerts per year**. Each alert lands in an analyst's queue and must be
disposed within a regulator-mandated window: *file a Suspicious Activity
Report (SAR)*, or *close as benign* with a written disposition rationale.

Today this is almost entirely manual. For every alert, an analyst:

- Pulls **6–10 evidence sources** — KYC, transactions, sanctions screen,
  applicable policy excerpts, SOP playbooks.
- Reasons across them to arrive at a verdict.
- Writes a SAR narrative that must be **objective, evidence-grounded,
  non-accusatory, statute-cited**, and survive regulator review (FinCEN,
  OFAC, FATF examiners).

Two failure modes dominate the queue:

| Failure | Cost |
|---|---|
| **False positives** (flagging benign activity) | Analysts spend ~80% of their day on cases that close as not-suspicious; SAR queues balloon; real bad actors get buried in noise. |
| **False negatives** (closing real laundering as benign) | Direct regulatory risk; fines have hit **$1 B+** per institution in recent enforcement actions. |

### What we are building

A **domain-specialized LLM agent** that produces a regulator-grade SAR
judgment end-to-end:

```json
{
  "is_suspicious": true,
  "suspicious_activity_report":
    "<400–800 char grounded narrative — positives cite typology + statute,
      negatives cite the surface red flag and the bundle evidence that
      disposes of it>"
}
```

### Workshop measurement target

A **single custom-trained Nemotron-3-Nano (30 B total / 3.2 B active per
token, hybrid Mamba-2 + Transformer MoE)**, on a leak-free SFT corpus,
beats off-the-shelf frontier models on the precision-critical AML triage
task — by a margin large enough to justify fine-tuning a specialised
open-weight model vs renting a frontier API.

The benchmark is the **200-case prod-mimic demo eval set**, stratified
across all 8 typologies, plus `clean` (no-typology) and `near_miss`
(typology pattern present, verdict benign) cohorts. Same backend,
same bundle shape, four different models behind it:

| Endpoint | F1 | Precision | Recall | Near-miss specificity | Clean-cohort FPR |
|---|---:|---:|---:|---:|---:|
| **aml-custom-task-nim** (our SFT checkpoint) | **0.769** | **0.694** | 0.862 | **0.767** | **0.029** |
| gemma-4-31b-it (frontier teacher) | 0.510 | 0.356 | **0.897** | 0.400 | 0.207 |
| nemotron-3-nano (untuned base) | 0.494 | 0.350 | 0.840 | 0.222 | 0.136 |
| gpt-5.2 (frontier general-purpose) | 0.331 | 0.207 | 0.828 | 0.100 | 0.464 |

What this says:

- **Precision** — our model produces **11 false positives** out of 200; the
  frontier models produce 47–92. In an analyst queue, those are
  3–8 minutes of wasted review each.
- **Near-miss specificity** — when shown a near-miss decoy (typology
  pattern present but benign), the frontier models flag 60–90% as
  suspicious. Our model flags 23%. This is the metric where domain
  training matters most.
- **Recall** — our model catches 25 of 29 true positives. Gemma catches
  one more (26); the gap is within statistical noise.
- **Clean-cohort FPR** — on benign activity, our model fires on 2.9% of
  cases. GPT-5.2 fires on 46.4%.

The whole workshop story is that **a specialised open-weight model
(Nemotron-3-Nano-30B-A3B, 3.2 B active per token) beats a much larger
general-purpose frontier model when the data is right** — and runs at a
fraction of the cost-per-call.

---

## 2. Tools and Custom Skills — the *Why*

The agent has two kinds of capability — **deterministic tools** that
retrieve evidence, and **model skills** that interpret it.

### 2.1 Five deterministic data-fetch tools

These never call an LLM. They are the agent's eyes — every byte that
later reaches the model passes through one of them.

| Tool | Returns | Backed by | Why this is a tool, not an LLM call |
|---|---|---|---|
| **Tool 1 — Transaction DB** | `transactions[]` for the entity over the investigation window | Postgres over the merged transaction ledger | Ground truth of money movement. Hallucination is unacceptable — the txn list either is or isn't an exhaustive enumeration. |
| **Tool 2 — KYC / CRM Store** | `kyc_profile` | Postgres over entity masters | The legitimate baseline. Without declared business purpose and expected monthly volume, "high velocity" has no reference frame. |
| **Tool 3 — Sanctions / PEP API** | `sanctions_pep_hits[]` | REST over the OpenSanctions snapshot (OFAC + EU + UN + PEP registers) | Hard regulatory hits. A real OFAC match is a near-mandatory SAR; a fuzzy collision is a near-miss decoy the model must learn to dismiss. |
| **Tool 4 — Policy RAG** | `policy_excerpts[]` | Vector store over FFIEC + FATF + FinCEN + OFAC corpus | Regulatory citations the model must quote *currently*. Frozen-in-weights memorization rots; retrieval keeps citations fresh. |
| **Tool 5 — SOP Repository** | `sop_excerpts[]` | In-cluster service over hand-authored SOPs (~40 pages, 8 typologies × ~5 pages each) | Institution-specific procedure the model cannot learn from public corpora. Defines what "escalate", "close as benign", "request EDD" mean for *this* bank. |

The five tools together produce the **evidence half** of the SAR bundle.
They are the work that LLMs do *badly* — retrieval, enumeration, exact
matching — and that deterministic systems do *cheaply*.

### 2.2 Four custom skills — interpretation, not retrieval

The same custom-trained NIM is *also* invoked as four specialist
sub-agents, each with a different `task_type`. These are not separate
models — they are **the one trained checkpoint behaving differently
depending on the system prompt**.

| Skill | Input | Output | Why this is a model task, not a deterministic computation |
|---|---|---|---|
| **`auxiliary_behavioral`** | Transactions + KYC, rendered as a structured passage | `{summary, metrics, evidence}` — channel mix, velocity, counterparty concentration, declared-vs-observed volume ratio, loop / pass-through indicator | Blends deterministic aggregation (sums, counts) with interpretive judgement (loop detection, pass-through framing). Making it a model task means the SAR call can later **verify** the behavioral output against the same `transactions[]` in the bundle — a self-consistency check that doesn't exist if features arrive silently from upstream. |
| **`auxiliary_numeric`** | Financial passage / table + a numeric question | `{answer, calculation, evidence}` | Multi-step arithmetic and threshold comparisons. Pure math is brittle for a generalist LLM; the model is trained on FinQA + TAT-QA shapes so it answers reliably. |
| **`auxiliary_citation`** | Long policy passage + a question | `{answer, evidence_span}` (verbatim quoted span) | Forces the model to *quote* the regulation rather than paraphrase it. Cheap insurance against citation hallucination — a known LLM failure mode that gets institutions cited in examiner reports. |
| **`auxiliary_statutory`** | Statute text + a fact pattern | `{label ∈ {entailment, contradiction, neutral}, reasoning}` | Apply-rule-to-facts is the core of legal reasoning. LegalBench shapes anchor this skill; AML statute Q&A operationalises it. |

### 2.3 The fifth task — `sar_judgment`

A single end-to-end call that consumes **all five tool outputs plus the
four auxiliary findings** and emits the binary verdict + grounded
narrative. Same trained checkpoint, different system prompt.

### 2.4 Why this split

We measured. On the 200-case demo eval (§3.5):

- GPT-5.2 flags almost every near-miss decoy as suspicious (specificity
  0.10) — its **general calibration is wrong** for AML.
- Gemma-4-31B has the highest recall (0.897) but **4.3× the false
  positives** of our model — over-suspicion at scale.
- Our base Nemotron (the pre-training checkpoint, before SFT) is no
  better than Gemma on F1 — base capacity isn't the limiter.

The gap is closed by **training the specialised model on the right
data** (Sections 6–8), not by reaching for a larger general-purpose
model.

---

## 3. Target Application — Trigger → Tools → Skills → SAR

The production agent lives in
[`10.appln_buildout/backend/`](./10.appln_buildout/backend/), built on
NVIDIA's NeMo Agent Toolkit (NAT). **One alert in → one SAR judgment +
one persisted case trace out.**

### 3.1 Trigger

```json
{
  "case_id":              "DEMO_0042",
  "alert_id":             "ALERT_2026_04_27_00123",
  "entity_id":            "CUST_8814729",
  "investigation_window": {"start": "2026-04-01", "end": "2026-04-30"},
  "trigger":              "transaction_monitoring_rule_R-117 — sub-CTR cash velocity"
}
```

Sourced from upstream **transaction-monitoring systems** (e.g. Verafin,
Actimize). In the demo, the 200-case manifest plays that role — each
case is one alert.

### 3.2 The seven-phase orchestrated flow

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │   PHASE 1 — DATA FETCH (3 deterministic tools, in parallel)         │
   │                                                                     │
   │   get_transactions(entity, window)      → transactions[]            │
   │   get_kyc(entity)                       → kyc_profile               │
   │   screen_sanctions(entity, counterparts) → sanctions_pep_hits[]     │
   └─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │   PHASE 2 — INTERNAL TYPOLOGY GUESS (rule layer, never reaches LLM) │
   │                                                                     │
   │   guess_typology = classify_typology(tx, kyc, hits, trigger)        │
   │                                                                     │
   │   Used INTERNALLY for:                                              │
   │     • picking which question/passage to send to each aux skill      │
   │     • filtering policy + SOP retrieval                              │
   │                                                                     │
   │   Never passed into the final SAR-judgment user message.            │
   └─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │   PHASE 3 — RETRIEVAL (2 more deterministic tools)                  │
   │                                                                     │
   │   policy_focused = retrieve_policy(typology=guess_typology, k=3)    │
   │   policy_broad   = retrieve_policy(no_filter, k=2)   ← safety net   │
   │   sop_excerpts   = get_sop(typology=guess_typology)                 │
   │                                                                     │
   │   The broad-search safety net is mandatory: if Phase 2's guess is   │
   │   wrong, the model still has alternative regulatory context.       │
   └─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │   PHASE 4 — AUXILIARY SKILLS (3 parallel LLM calls + 1 Python calc) │
   │                                                                     │
   │   numeric_agent   : aux_numeric_call(passage, NUMERIC_Q[guess_typ]) │
   │   citation_agent  : aux_citation_call(policy_focused[0], CIT_Q)     │
   │   statutory_agent : aux_statutory_call(STATUTE[guess_typ], facts)   │
   │   behavioral      : computed deterministically in Python            │
   │                     (channel mix, velocity, counterparty            │
   │                      concentration, declared-vs-observed ratio,     │
   │                      loop/pass-through indicator)                   │
   │                                                                     │
   │   The user messages into each aux LLM call are the task-type        │
   │   schema only — no typology / frame / decision hint in any of them. │
   └─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │   PHASE 5 — AUX GATE (input guard → schema check → judge)           │
   │                                                                     │
   │   Each finding survives only if:                                    │
   │     1. The input passage was present and well-formed                │
   │     2. The output matches its typed schema                          │
   │     3. (Optional) An independent judge LLM confirms consistency     │
   │                                                                     │
   │   Failed findings are DROPPED. Their absence is recorded in the     │
   │   case trace so the analyst can see what was attempted vs trusted.  │
   └─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │   PHASE 6 — SAR JUDGMENT (one LLM call to the custom-trained NIM)   │
   │                                                                     │
   │   USER MESSAGE — exactly 7 keys, evidence only:                     │
   │     { task_type:"sar_judgment",                                     │
   │       transactions:[…], kyc_profile:{…},                            │
   │       sanctions_pep_hits:[…], policy_excerpts:[…],                  │
   │       sop_excerpts:[…], auxiliary_findings:{…}|null }               │
   │                                                                     │
   │   MODEL OUTPUT — exactly 2 fields:                                  │
   │     { is_suspicious: <bool>,                                        │
   │       suspicious_activity_report: "<400–800 char grounded text>" }  │
   │                                                                     │
   │   The model is the actual classifier. Nothing in the user message   │
   │   tells it the answer — it derives the verdict from raw evidence.   │
   └─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │   PHASE 7 — TRACE PERSISTENCE                                       │
   │                                                                     │
   │   A CaseTrace JSON is written to data/traces/<case_id>.json with    │
   │   every phase's inputs, the rule-layer typology guess (audit-only), │
   │   all aux findings (raw + gated), the model output, and latencies.  │
   │   This is the audit trail consumed by analytics + the eval API.     │
   └─────────────────────────────────────────────────────────────────────┘
```

### 3.3 The two hard rules driving everything

These are the contract that prevents the model from "cheating" — from
finding a shortcut feature that correlates with the label and ignoring
the actual evidence.

**Rule A — No label leakage in the user message.** The SAR-judgment user
message contains *evidence only*. It does not carry a pre-computed
regulatory frame, a pre-computed typology label, or a SAR-recommendation
hint. The model derives the verdict from raw evidence alone. This is
enforced at every layer — at SFT data construction, at the prod-mimic
data generation, and at backend request assembly (where the request
schema explicitly forbids any extra keys).

**Rule B — Both classes are grounded.** Every assistant message has a
non-empty `suspicious_activity_report` of 400–800 characters:

- **Positive cases** produce a regulator-grade SAR narrative — entity,
  dates, amounts, typology, statute citations.
- **Negative cases** produce a **disposition rationale** that explicitly
  names the surface red flag (e.g. "OFAC fuzzy match score 0.78") *and*
  cites the bundle evidence that disposes of it ("but the entity has
  held this counterparty in its KYC since 2019 and the match resolves
  to a different jurisdiction").

The model is never allowed to return an empty narrative on negatives.
This makes the negative training signal as informative as the positive
one, and it surfaces the model's reasoning to the human analyst at
disposition time.

### 3.4 The 7-key SAR bundle — the wire contract

```json
{
  "task_type": "sar_judgment",
  "transactions":       [{"date":"…","amount":…,"channel":"…","counterparty":"…", …}],
  "kyc_profile":        {"entity_id":"…","entity_type":"individual|business",
                         "expected_monthly_volume":…,"business_purpose":"…",
                         "risk_rating":"low|medium|high|enhanced",
                         "incorporation_jurisdiction":"…"},
  "sanctions_pep_hits": [{"name":"…","list":"OFAC|EU|UN|OpenSanctions","match_score":0.0–1.0}],
  "policy_excerpts":    [{"source":"FFIEC|FATF|FinCEN","section":"…","url":"…","text":"…"}],
  "sop_excerpts":       [{"sop_id":"…","section":"…","text":"…"}],
  "auxiliary_findings": {
    "behavioral": [{"summary":"…","metrics":{…},"evidence":"…"}],
    "numeric":    [{"question":"…","answer":"…","calculation":"…","evidence":"…"}],
    "citation":   [{"question":"…","answer":"…","evidence_span":"…"}],
    "statutory":  [{"question":"…","label":"…","reasoning":"…"}]
  }
}
```

Each entry inside `auxiliary_findings.*` is independently optional —
whatever the aux gate accepted. The model is trained to fall back to raw
evidence when any aux findings are missing.

### 3.5 What the workshop demo measures

- **Eval set**: 200 cases stratified across all 8 typologies + the
  `clean` (no-typology) and `near_miss` (typology pattern present but
  benign) cohorts.
- **Process**: each case is run through the seven phases above; the
  trace is persisted; a scoring script compares the model's
  `is_suspicious` against the gold label.
- **Headline metrics**: F1, precision, recall, **near-miss specificity**
  (does the model flag pattern-present-but-benign decoys?), and
  **clean-cohort FPR** (does the model flag benign activity?).
- **Side-by-side report**: the same 200 cases are run against the
  custom NIM, against the untuned base Nemotron, against Gemma, and
  against GPT-5.2 — same backend, same bundle shape, four different
  models. The scoring script emits one combined JSON consumed by the
  UI tile.

---

# HOW

## 4. Data Strategy — sources and significance

The training corpus assembled spans **two CPT layers** (broad financial
register + AML-specific domain) and **three SFT tiers** (primary
`sar_judgment` shapes + auxiliary skills + heavy-filter near-miss
negatives). Combined: ~5.7 GB raw → ~700 M–1.65 B CPT tokens + ~75 K
SFT records.

> **How we sourced it.** HuggingFace + Kaggle for instruction-shape data
> (FinQA, LegalBench, FinanceBench, SARSum, IBM AML); direct PDF scrapes
> of FinCEN / FATF / OFAC / FFIEC websites for regulatory corpora;
> `data.gov` + ICIJ + OpenSanctions for bulk structured releases. ~300 GB
> free disk for the raw materialization. Full source-by-source table in
> [`1.data_download/data_source_strategy.md`](./1.data_download/data_source_strategy.md).

### 4.1 CPT Layer 1 — broad financial register (~3.16 B tokens raw)

Teaches the base model to *speak* like a financial regulator, before
any AML specialization.

| Source | Significance | Sample record |
|---|---|---|
| **EDGAR-CORPUS** (SIC 6000–6999), 63 K records | 10-K filings from banks, brokerages, insurers. Teaches corporate-financial register at scale. | `{"text": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk. The Company's primary market risk exposures arise from changes in interest rates and foreign currency exchange rates…"}` |
| **Pile of Law — `cfr`** (Titles 12, 17, 31) | The Code of Federal Regulations covering banking, securities, money & finance — statute the model must quote verbatim. | `{"text": "12 CFR § 21.21 Procedures for monitoring Bank Secrecy Act (BSA) compliance…"}` |
| **Pile of Law — `uscode`** (Title 12, 18 ch. 95–96, 31) | The federal statute layer below CFR. RICO and the §§ 1956–1957 money-laundering statutes live here. | `{"text": "Title 31 § 5324(a) — No person shall, for the purpose of evading the reporting requirements…"}` |
| **Pile of Law — `federal_register`** (financial agencies) | Rulemaking notices from FinCEN / Fed / OCC / FDIC / SEC. The "this is how we propose to regulate X" voice. | `{"text": "FINANCIAL CRIMES ENFORCEMENT NETWORK 31 CFR Part 1010 RIN 1506-AB52 Beneficial Ownership Information…"}` |
| **Pile of Law — `sec`, `oig`, `doj_guidance`** | Enforcement orders, OIG audits, prosecutor guidance. Models the *applied* voice of regulators. | `{"text": "RELEASE NO. 99012 / December 5, 2023 Order Instituting Administrative Proceedings…"}` |
| **`uscode_house`** (Title 12, 18, 31) | The official congressional rendering of the same statutes — cross-deduped against the Pile-of-Law copy. | `{"text": "TITLE 31 — MONEY AND FINANCE SUBTITLE IV…"}` |

### 4.2 CPT Layer 2 — AML-specific (~40 M tokens raw)

Narrow domain adaptation. After L2 the model has read every public
FinCEN advisory, OFAC enforcement order, FATF typology report, and
AML court opinion.

| Source | Significance | Sample record |
|---|---|---|
| **FinCEN Advisories** — 365 PDFs | The most operational regulatory guidance on actual ML patterns. Advisory FIN-2014-A005 (structuring), FIN-2022-A002 (elder exploitation), FIN-2020-A006 (sanctions evasion), etc. | `{"text": "FIN-2022-A002 Advisory on Elder Financial Exploitation … Financial institutions should be alert to…"}` |
| **FinCEN SAR Activity Reviews** — 332 PDFs | Periodic reports summarising real SAR filings. Gold-standard examples of regulator-grade SAR prose. | `{"text": "SAR Activity Review — Trends, Tips & Issues Issue 23 … Case Study: Structuring by a Cash-Intensive Business…"}` |
| **FinCEN Federal Register** — 430 PDFs | Final-rule + proposed-rule text in BSA / AML. The textual foundation of US AML law. | `{"text": "Anti-Money Laundering Program Effective Date Pursuant to Section 326 of the USA PATRIOT Act…"}` |
| **FinCEN Enforcement** — 315 PDFs | Consent orders, civil-money-penalty actions. Teaches what triggers actual enforcement vs what closes benign. | `{"text": "Consent Order — In the Matter of [Bank], Civil Money Penalty $1,500,000 for willful violations of the BSA…"}` |
| **OFAC Guidance** — 1,451 PDFs (513 MB) | The complete OFAC published-guidance corpus. Teaches sanctions reasoning and 50-Percent Rule mechanics. | `{"text": "Frequently Asked Questions: Cuban Assets Control Regulations…"}` |
| **FATF Publications** — 16 PDFs | International typology reports (trade-based ML, terrorist financing, virtual assets). Anchors typology coverage outside the US. | `{"text": "FATF Report — Money Laundering Risks Arising from Trafficking in Human Beings and Smuggling of Migrants…"}` |
| **FinCEN Files (ICIJ)** — 91 articles | Investigative journalism on real cross-border laundering. Narrative exposure to the human side of these cases. | `{"text": "The FinCEN Files investigation revealed how some of the world's biggest banks have moved trillions…"}` |
| **`courtlistener`** AML opinions — 156 | Court-decided AML cases. Teaches how statutes get *applied* in adversarial settings. | `{"text": "UNITED STATES v. DEFENDANT — Indictment for violation of 18 U.S.C. § 1956…"}` |

### 4.3 SFT Tier 1 — `sar_judgment` primary sources

These provide the *labeled* AML examples around which the SFT corpus is
built. Sources differ in what they natively carry:

| Source | What's in it | Used as |
|---|---|---|
| **Enterprise Financial Crime AI** — 22 K alerts + 15 K investigations + 6 K SARs + 500 K transactions | Most complete real-shape AML data available. Carries transactions + KYC + sanctions + alert outcomes. | Transactional context + KYC bundle source. The narrative source for SAR records comes from SARSum, not this set's templated narratives. |
| **SARSum** — 2 K SAR sets × 7 prose notes × 6 quality tiers | Each note carries a Suspicious / Not concerning label + regulator-grade reasoning prose, indexed by typology + decision. | Primary gold-narrative library, paired with transaction bundles via a "pair-and-ground" rule (Appendix E of `training_strategy.md`). |
| **IBM AML HI-Small** — transactions + 7 typology pattern blocks | Synthetic transactions with explicit pattern labels and an `Is Laundering` ground truth flag. | Typology-labeled transaction bundles, especially for structuring + layering. |
| **AMLGentex** — 10 K accounts, 678 K transactions with `isSAR`, 618 alert patterns | Open-source synthetic generator with parameterised typology patterns. | Typology-targeted synthesis to fill typologies under-represented elsewhere (`trade_based_ml`, `shell_company`, etc.). |

### 4.4 SFT Tier 2 — auxiliary task sources

Each auxiliary skill has dedicated training data shaped to its task:

| Source | Records | Powers |
|---|---:|---|
| **FinQA** | 6,251 train | `auxiliary_numeric` — multi-step arithmetic over 10-K tables. |
| **TAT-QA** | ~2 K (sampled) | `auxiliary_numeric` — table + text hybrid Q&A. |
| **FinanceBench** | ~150 | `auxiliary_citation` — financial Q&A with evidence-span gold. |
| **LegalBench** (16 sub-tasks) | ~3 K | `auxiliary_statutory` — rule-to-fact entailment. |
| **FFIEC BSA/AML Manual** | 164 HTML pages → ~5 K Q&A | `auxiliary_citation` + AML-specific — generated by templating Q&A off the manual's section structure. |

### 4.5 SFT Tier 3 — heavy-filter / dual-role

| Source | Significance |
|---|---|
| **Finance-Instruct-500k** — 580 MB JSONL (~10–15% retained after AML keyword filter) | A general financial-instruct corpus, filtered aggressively for AML-relevant subsamples. Adds language-style diversity. |
| **CFPB Consumer Complaints** — ~289 K records (filtered) | Real consumer descriptions of activity that *looks* suspicious but is benign (billing disputes, identity errors, merchant chargebacks). The **near-miss negative class** — without these, the model degenerates into "any non-trivial activity ⇒ suspicious." |

---

## 5. Data Curation — stages of the pipeline

Curation happens in two passes — one for raw extraction (Step 2), one
for the CPT corpus (Step 3).

### 5.1 Step 2 — Raw extraction

Single CPU-only pass. Each phase has its own extractor:

1. **PDF extraction** — for FinCEN / OFAC / FATF / SEC PDFs. OCR
   fallback for scanned pages; documents where all pages are
   low-confidence OCR are discarded.
2. **HTML / structured parse** — FFIEC manual, EDGAR-CORPUS, ICIJ
   FinCEN Files articles → plain text with section anchors preserved.
3. **JSONL passthrough** — HuggingFace / Kaggle sources already in
   text form: copy + tokenize.
4. **Transactional passthrough** — files that back the agent's *tools*
   (transactions, KYC, sanctions snapshots) are copied as-is, not
   reshaped.
5. **Tokenization** — every JSONL row is tokenized with the **same
   Nemotron-3-Nano tokenizer** the model uses at training time, so the
   token budget reported in `summary.json` is the actual training cost.

Output: `data/cpt/level_{1,2}/<source>.jsonl` + `data/sft/<source>.jsonl`
+ `data/transactional/<source>/…` + `summary.json` with per-source
record counts + token totals.

### 5.2 Step 3 — CPT curation

Six staged phases, each checkpointed (resumable on rerun):

| Phase | What happens |
|---|---|
| **0. INGEST** | Read every `level_{1,2}/*.jsonl`, force-tag each record with its `layer`, split large files into chunks for pipeline parallelism. |
| **1. TEXT_CLEAN** | HTML strip → English-language filter → length filter → quality filter (alphanumeric / repeated-line / common-word ratios) → boilerplate removal → typed-tag PII redaction (`[SSN]`, `[EIN]`, `[ACCOUNT_ID]`, `[PHONE]`, …). |
| **2. EXACT_DEDUP** | SHA-256 over the `text` field. Catches identical-copy duplicates within a source. |
| **3. FUZZY_DEDUP** | MinHash (Jaccard ≈ 0.79). Catches near-duplicates from PDF reprints or text-with-different-whitespace. |
| **4. XSOURCE_DEDUP** | Targeted cross-source dedup with richness scoring. The same statute often appears in multiple sources (Pile-of-Law `uscode` + `uscode_house`; Pile-of-Law `cfr` + derived 31 CFR Chapter X). The richer rendering wins. |
| **5. WRITE_CURATED** | Group survivors by `(layer, source)`, write `cpt/level_1/<source>.jsonl` and `cpt/level_2/<source>.jsonl`. Both layers share one dedup pass — curation is a corpus-quality concern; curriculum is a training-time concern. |

PII strategy is **typed-tag redaction** (not masking). Tags appear in
the curated text; the original PII span is never persisted, only its
salted SHA-256 in an audit side-file.

### 5.3 SFT data prep

After the SFT corpus is generated (§7), three more stages prepare it
for the trainer:

| Stage | What it does |
|---|---|
| **1. Shuffle** | Normalize records to the chat-SFT envelope `{messages:[…]}`, globally shuffle, split into chunked train + val + test JSONL files. |
| **2. Analyse** | Read-only token-length percentile report. Used to pick `packed_sequence_size` and the per-record max-length cap. |
| **3. Filter** | Drop records exceeding the max-length cap; sanitise Unicode line separators; enforce the `last_message=assistant` invariant. |

The final cleaned corpus is what the SFT trainer consumes.

---

## 6. CPT Phase — Level 1 broad register, Level 2 AML with replay

Two sequential CPT runs on the FP8 Nemotron-3-Nano-30B-A3B base, on
8×H100 NVL. Full recipe in [`3.cpt/2.run_cpt/README.md`](./3.cpt/2.run_cpt/README.md).

```
   ┌──────────────────────────────────┐
   │  Base: Nemotron-3-Nano-30B-A3B-FP8 │
   └────────────────┬─────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────────────┐
   │  Phase 1 — L1-only, 1 epoch                      │
   │  ~1.82 B tokens trained                          │
   │  Goal: broad financial / regulatory register     │
   │                                                  │
   │  Output: cpt-l1-final/  + mid-snapshot (rollback)│
   └────────────────┬─────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────────────┐
   │  Phase 2 — L2 upsampled + L1 replay              │
   │  3–4 epochs, INTERLEAVED at every microstep      │
   │  ~2.16 B tokens trained                          │
   │  Goal: AML-specific domain specialization        │
   │                                                  │
   │  Output: cpt-l2-final/  → consumed by SFT        │
   └──────────────────────────────────────────────────┘
```

### 6.1 Level 1 — broad financial register

- **Goal**: install the financial / regulatory / legal language register.
  After Phase 1 the model recognises statute citations, SEC filing
  structure, federal-register style — but is *not yet* AML-specialized.
- **Data**: 1.82 B-token disjoint training slice from L1. EDGAR
  dominates (~82% post-cap); the seven supporting sources fill out the
  regulatory voice. EDGAR's raw 89% share is capped so the smaller
  sources can actually move the model.
- **Pool partition**: L1 is split *once* at data-prep time into three
  disjoint pools:
  - `L1-train-phase1` (~1.82 B tokens) — consumed in Phase 1.
  - `L1-replay-pool` (~1.25 B EDGAR-only) — **held back** for Phase 2's
    replay stream.
  - `L1-holdout-eval` (~85 M tokens) — per-source eval slice.
  - **No token appears in two pools** — the data-prep contract.
- **Schedule**: Warmup-Stable-Decay (WSD), peak LR 2.0e-5, ~1,734
  optimizer steps. WSD chosen over cosine so Phase 2 can resume from
  the decay endpoint without an LR reset.
- **Mid-snapshot preserved** as rollback insurance.

### 6.2 Level 2 — AML specialization with interleaved L1 replay

L2 alone (40 M tokens) is too small to dominate gradients in a one-shot
run — and even if it did, it would catastrophically forget L1. The fix
is **per-microstep interleaving**:

```
At every training microstep, the dataloader samples from one of two streams:

  ┌─────────────────────────────────────┐ 44% of microsteps
  │ L2 (upsampled, square-root tempered)│ ─────────────────►
  │   ofac 26%, fincen_fed_reg 14.6%,   │
  │   fincen_sar_reviews 13.8%, …       │   ~240 M tokens / epoch
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐ 56% of microsteps
  │ L1-replay-pool — fresh EDGAR        │ ─────────────────►
  │   (1.25 B tokens held back from L1) │   ~300 M tokens / epoch
  └─────────────────────────────────────┘
                                            ~540 M tokens / Phase-2 epoch
                                            × 4 epochs = ~2.16 B total

Critical property: across 4 Phase-2 epochs, total replay draw = 1,200 M ≈
size of the replay pool. The model never sees an EDGAR document twice
across the entire CPT pipeline.
```

Three deliberate choices:

1. **Square-root tempering on L2** — OFAC's natural 45% share would
   drown the smaller FATF / courtlistener sources. Tempering brings
   OFAC down to 26% and lifts FATF / courtlistener to ~6–7%.
2. **L1 replay > 50% of microsteps** — protects against catastrophic
   forgetting. The standard "rehearsal" result from continual learning.
3. **Replay pool is fresh tokens**, not Phase-1 leftovers — every token
   in `L1-replay-pool` was held back from Phase 1 specifically so
   Phase 2 has *unseen* anti-forgetting signal, not memorized tokens.

Lifetime CPT mix: **3,020 M L1 : 960 M L2 ≈ 3:1**, deliberately narrower
than a 10:1 mix because AML is the actual deployment domain. A 5 pp
MMLU regression triggers a re-run with replay share raised to 70%.

### 6.3 Why two phases instead of one mixed run

- One-shot mix at peak LR pulls the model toward AML minutiae too
  early, before the broad register has consolidated.
- Phase 2 lowers the LR and concentrates compute on the small AML
  corpus, which would be invisible in a one-shot 1:1 mix.
- The mid-Phase-1 checkpoint is rollback insurance against Phase 2
  over-narrowing the model. Rollback is cheap; mistraining a 30 B
  model is not.

---

## 7. Synthetic Data for SFT — target distribution

The SFT corpus is **75 K records**. The synthetic-data generation (SDG)
pipeline oversamples to ~115 K and per-record validators + corpus-level
audits drop sub-quality records to land at 75 K.

### 7.1 Distribution by task type (skill)

| `task_type` | Records | Share |
|---|---:|---:|
| `sar_judgment` | 50,000 | 67% |
| `auxiliary_numeric` | 6,250 | 8% |
| `auxiliary_citation` | 6,250 | 8% |
| `auxiliary_statutory` | 6,250 | 8% |
| `auxiliary_behavioral` | 6,250 | 8% |
| **Total** | **75,000** | **100%** |

Two-thirds of the corpus is the primary deployment task; the remaining
third is split evenly across the four specialist sub-skills. The model
sees the same shape at inference time on every call.

### 7.2 Distribution by label (for `sar_judgment`)

| Axis | Target |
|---|---|
| **Class balance** (`is_suspicious=true`) | **45% positive / 55% negative** (±2 pp) |
| Layering frame cap (the easiest typology to over-generate) | ≤ **25%** of corpus |
| Per-frame label balance | **No frame is one-sided** — every regulatory frame has ≥ 30% minority-class representation |

The per-frame label balance is the most important constraint in the
whole corpus. Without it, the model learns shortcuts of the form
`if frame=X then suspicious=true` and ignores the actual evidence.
The contract makes sure:

- The `benign` frame carries explicit positives (mixed-signal façade
  cases where the surface looks benign but evidence flags it).
- The `sanctions` frame carries explicit negatives (common-name PEP
  collisions, fuzzy OFAC matches that resolve to clean entities).
- For every typology, at least 30% of records have the opposite label
  of the majority — so the model is forced to read the evidence to
  predict.

### 7.3 Distribution by SAR variant — augmented / bare / adversarial

Every `sar_judgment` record is one of three variants depending on whether
it carries `auxiliary_findings`:

| Variant | Share | Purpose |
|---|---:|---|
| **Augmented** — `auxiliary_findings` populated with correct findings | ~70% | The dominant runtime case. Teaches the model to **cite** pre-computed findings verbatim rather than recompute. |
| **Bare** — `auxiliary_findings` absent or all sub-arrays empty | ~25% | Teaches the model to **fall back** on raw inputs when the aux gate dropped findings. |
| **Adversarial** — ≥ 1 deliberately wrong finding (numeric flipped, citation swapped, statutory inverted, or behavioral metrics corrupted) | ~5% | Teaches the model to **detect inconsistency** against raw inputs — prevents sycophancy to upstream errors. Especially powerful for the behavioral case, where the model can recompute the metrics from the same `transactions[]` in the bundle. |

### 7.4 Distribution by typology and source

Eight canonical typologies must each be represented across both the
suspicious and the near-miss-benign cohorts:

- `structuring`, `smurfing`, `layering`, `trade_based_ml`,
  `shell_company`, `human_trafficking`, `terrorist_financing`,
  `elder_exploitation`

For each typology the corpus carries:

- **Positive cases** — the typology pattern is present and the verdict
  is suspicious.
- **Near-miss negative cases** — the typology pattern is present but
  the verdict is benign (evidence in the bundle resolves the surface
  signal). This is the cohort that teaches the model nuance.
- The combined per-typology floor is large enough that no typology has
  below ~6% representation in the SAR-judgment slice.

Source mix within `sar_judgment`:

| Source | Share | Role |
|---|---:|---|
| Enterprise Financial Crime AI (real-shape transactions + KYC) | ~40% | Realism floor. |
| AMLGentex synthetic (typology-targeted) | ~30% | Coverage for typologies EFC is thin on. |
| IBM AML HI-Small (labeled patterns) | ~15% | Pattern-anchored positives. |
| CFPB Consumer Complaints (synthesized negatives) | ~10% | Near-miss negatives — the precision battleground. |
| Hand-authored exemplars (terrorist_financing, elder_exploitation, human_trafficking) | ~5% | Rare-typology coverage. |

### 7.5 Distribution by KYC profile and channel mix

| KYC risk_rating | Share |
|---|---:|
| low | 40% |
| medium | 40% |
| high | 15% |
| enhanced | 5% |

| Channel | Share of all transactions |
|---|---:|
| ACH | 40% |
| Wire | 25% |
| Card | 20% |
| Cash | 10% |
| Cheque | 5% |

These distributions match what the production data plane emits at
runtime — **zero distribution shift between training-time and
runtime-time inputs**. The same generators that fed the SFT corpus
also feed the demo's 200-case manifest, so the model is never asked
to generalize across a population it hasn't seen.

### 7.6 Corpus-level audits

A corpus that passes every per-record check can still violate the
distribution targets. Stage 9 of the SDG pipeline emits a
`phase_stats.json` and **blocks** the run if any of the following fail:

| Audit | Threshold |
|---|---|
| Per-typology floor | Every canonical typology has a meaningful presence in both label classes |
| No-typology-label-coupling | Each typology has ≥ 30% minority-class representation |
| Pos / neg ratio | 45 / 55 ± 5 pp |
| Variant ratio | 70 / 25 / 5 augmented / bare / adversarial ± 5 pp |
| Source contribution cap | No single source > 30% of records |
| Aux task balance | Each `auxiliary_*` task within ±2 pp of its 8% target |
| Schema validity | 100% of records validate against the 7-key SAR / 3-key aux schemas |

A failed audit is **actionable** — the pipeline emits a remediation
hint (e.g. "structuring positives below floor; boost typology=structuring
in the next run") and exits non-zero.

---

## 8. SFT Phase — shuffle, analyse, filter, train

A single training pass on the Phase-2 CPT checkpoint that teaches the
model all five task types from one checkpoint via system-prompt
differentiation. Pipeline in [`7.run_sft/`](./7.run_sft/), trainer
recipe in [`7.run_sft/4.run_sft/recipe_a100sxm-8.yaml`](./7.run_sft/4.run_sft/recipe_a100sxm-8.yaml).

### 8.1 Pipeline

```
   data/raw/       <-- 75 K SFT JSONL records from the SDG pipeline
       │
       │  (optional)  data/check_format_fix.py
       ▼
   data/fixed/     <-- drops malformed; enforces last-message=assistant
       │
       ▼
   1.shuffle_dataset/shuffle.py
       │              <-- terashuf globally + split into chunks + val + test
       ▼
   sft_mixed.{chunk.NN, val, test}.jsonl
       │
       │  2.analyse_dataset/analyse_dataset.py  (read-only percentile report)
       │
       ▼
   3.filter_dataset/filter_data.py    <-- drop records over max length
       │
       ▼
   3.filter_dataset/rebuild_sft_jsonl.py  <-- Unicode sanitation;
       │                                       enforce last_msg=assistant
       ▼
   final_data_clean/sft_mixed.{chunk.NN, val, test}.jsonl
       │
       ▼
   4.run_sft/recipe_a100sxm-8.yaml + finetune.py    <-- 8 × A100 SXM
       │
       ▼
   4.run_sft/checkpoints/LOWEST_VAL/    = sft-final
                                          → deployed as aml-custom-task-nim
```

### 8.2 Training setup

| Setting | Value | Rationale |
|---|---|---|
| **Base** | The Phase-2 CPT checkpoint (`cpt-l2-final`) | The output of §6. |
| **Sequence length** | Variable, packed up to ~5K tokens | The 7-key SAR bundle averages ~3K tokens; packing fills out the sequence with multiple records to maximise throughput. |
| **Global batch size** | 16 | Small enough to make optimizer steps frequent at 75 K records. |
| **Local batch size** | 1 | Memory-bound. |
| **LR schedule** | Cosine, peak 1e-5, warmup 100 | SFT convention; peak LR ~5× lower than the Phase-2 CPT peak to preserve CPT knowledge. |
| **Optimizer** | AdamW, β1=0.9, **β2=0.999** | SFT convention; the higher β2 dampens noise on small batches. |
| **Epochs** | 1 | 75 K records × ~3 K tokens ≈ 225 M training tokens; one pass is sufficient. |
| **Precision** | bf16 | A100 doesn't have native FP8 kernels. |
| **Parallelism** | EP=8, FSDP across all 8 ranks | Same shape as CPT. |
| **Wall-clock** | ~6–10 hours | For 75 K records on 8 × A100 SXM. |

### 8.3 What we ship

`4.run_sft/checkpoints/LOWEST_VAL/` is the **`sft-final` checkpoint** —
deployed as `aml-custom-task-nim-1` via the NIM container in
[`9.custom_model_deployment/`](./9.custom_model_deployment/). That
endpoint is what the agentic application in
[`10.appln_buildout/backend/`](./10.appln_buildout/backend/) hits for
every aux-skill and SAR call at demo time.

### 8.4 What SFT does *not* do

- **No separate classifier head.** The model emits only
  `{is_suspicious, suspicious_activity_report}`. Typology surfaces
  implicitly inside the narrative ("the pattern is consistent with
  layering pass-through per FFIEC guidance…"), not as a discrete output
  field.
- **No retrieval-augmented training.** The 7-key bundle is a *closed*
  context; retrieval is the agent's job at runtime, not the model's
  job during SFT.
- **No grammar-constrained decoding.** The SFT distribution alone
  teaches schema compliance — we explicitly avoid logit-biased decoding
  so the model produces valid JSON because it learned to, not because
  it was forced to.

### 8.5 Pass-criteria

The SFT checkpoint is gate-tested on the 200-case demo eval (§3.5).
Pass-criteria:

| Metric | Pass | Stretch |
|---|---|---|
| F1 | ≥ 0.65 | ≥ 0.75 |
| Precision | ≥ 0.55 | ≥ 0.65 |
| Recall | ≥ 0.80 | ≥ 0.85 |
| Near-miss specificity | ≥ 0.60 | ≥ 0.75 |
| Clean-cohort FPR | ≤ 0.10 | ≤ 0.05 |

Current checkpoint on the four-way benchmark:

| Endpoint | F1 | Precision | Recall | NM-Spec | Clean-FPR |
|---|---:|---:|---:|---:|---:|
| **aml-custom-task-nim** | **0.769** ✅ | **0.694** ✅ | 0.862 ✅ | **0.767** ✅ | **0.029** ✅ |
| gemma-4-31b-it | 0.510 ❌ | 0.356 ❌ | 0.897 ✅ | 0.400 ❌ | 0.207 ❌ |
| nemotron-3-nano (base) | 0.494 ❌ | 0.350 ❌ | 0.840 ✅ | 0.222 ❌ | 0.136 ❌ |
| gpt-5.2 | 0.331 ❌ | 0.207 ❌ | 0.828 ✅ | 0.100 ❌ | 0.464 ❌ |

The custom model passes the stretch target on every metric except
recall (where it's at the pass target, within one case of Gemma).

---

## Appendix — Where to find more

| Topic | Document |
|---|---|
| Full plan-of-record | [`training_strategy.md`](./training_strategy.md) |
| Source-by-source download recipe | [`1.data_download/README.md`](./1.data_download/README.md), [`1.data_download/data_source_strategy.md`](./1.data_download/data_source_strategy.md) |
| Raw extraction + tokenization | [`2.data_processing/README.md`](./2.data_processing/README.md) |
| CPT data curation | [`3.cpt/1.data_curation/README.md`](./3.cpt/1.data_curation/README.md) |
| CPT training recipe | [`3.cpt/2.run_cpt/README.md`](./3.cpt/2.run_cpt/README.md) |
| SDG strategy | [`4.sdg_sft/SDG_STRATEGY_SFT.md`](./4.sdg_sft/SDG_STRATEGY_SFT.md) |
| SFT training pipeline | [`7.run_sft/README.md`](./7.run_sft/README.md) |
| Custom NIM deployment | [`9.custom_model_deployment/README.md`](./9.custom_model_deployment/README.md) |
| Target agentic application | [`10.appln_buildout/backend/application.md`](./10.appln_buildout/backend/application.md), [`10.appln_buildout/backend/backend.md`](./10.appln_buildout/backend/backend.md) |
