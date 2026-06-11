# AML Investigation Agent — End-to-End Application Overview

> A workshop-grade AML application that turns a raw transaction alert into a
> regulator-grade SAR decision in one pass, using a custom-trained
> investigator model orchestrated through the NVIDIA NeMo Agent Toolkit.

---

## 1. What the application is

A REST API that takes a single AML alert and returns:

1. A boolean **SAR decision** (`is_suspicious`)
2. A grounded **SAR narrative** (or a disposition rationale when not suspicious)
3. A full **audit trace** of every tool call, retrieved policy, auxiliary
   finding, and final reasoning step — sufficient for a compliance officer
   to review and sign off.

The reasoning core is a **custom SFT/RL-trained model** (built on top of
**Nemotron-3-Nano** via CPT → SFT → RL) deployed as an NVIDIA NIM. The
NeMo Agent Toolkit (NAT) orchestrates a fixed sequence of tool calls and
LLM invocations around it.

**One sentence**: alert in → tools fetch context → 4 auxiliary skills derive
typed findings → the custom model writes the SAR judgment → trace out.

---

## 2. End-to-end flow

```
                    ┌────────────────────────────────────────────────────┐
                    │                ALERT (trigger)                     │
                    │  case_id, entity_id, window, trigger_summary       │
                    └──────────────────────┬─────────────────────────────┘
                                           │
                ┌──────────────────────────▼──────────────────────────┐
                │  PHASE 1 — Data plane (deterministic tool calls)    │
                │                                                     │
                │   get_transactions   get_kyc   screen_sanctions     │
                │   retrieve_policy    get_sop                        │
                └──────────────────────┬──────────────────────────────┘
                                       │
                ┌──────────────────────▼──────────────────────────────┐
                │  PHASE 2 — Auxiliary skills (4 parallel LLM calls)  │
                │                                                     │
                │   auxiliary_behavioral    auxiliary_numeric         │
                │   auxiliary_citation      auxiliary_statutory       │
                └──────────────────────┬──────────────────────────────┘
                                       │
                ┌──────────────────────▼──────────────────────────────┐
                │  PHASE 3 — SAR judgment (1 LLM call, custom model)  │
                │                                                     │
                │   { is_suspicious, suspicious_activity_report }     │
                └──────────────────────┬──────────────────────────────┘
                                       │
                ┌──────────────────────▼──────────────────────────────┐
                │  RESPONSE: decision + narrative + full case trace   │
                └─────────────────────────────────────────────────────┘
```

**Latency budget per alert** (single-tenant, local NIM):
~6–10 s end-to-end (Phases 1 + 2 + 3 in sequence with internal parallelism).

---

## 3. The trigger — what comes in

A single POST to `/api/investigate` with the alert payload.

```json
{
  "case_id": "CASE_20260204_001",
  "alert_id": "ALERT_8b1f3c",
  "entity_id": "SYN_6b558d20",
  "investigation_window_start": "2026-02-01",
  "investigation_window_end":   "2026-02-28",
  "trigger_summary": "Multiple sub-CTR cash deposits flagged by rule R-127."
}
```

The agent looks up the alert (or accepts the inline payload) and kicks off
the investigation. Every downstream decision is anchored to the
`investigation_window` so cases are reproducible.

---

## 4. Phase 1 — Data plane (deterministic tool calls)

Five tools are invoked, the first three in parallel. No LLM in this phase —
just structured fetches from the bank's data stores.

| Tool | Purpose | Returns |
|---|---|---|
| `get_transactions` | Pull tx in the window for the entity | `list[Transaction]` |
| `get_kyc` | Pull the entity's KYC profile | `KYCProfile` |
| `screen_sanctions` | OFAC/PEP screening per counterparty (fan-out, capped at 10) | `list[SanctionsHit]` |
| `retrieve_policy` | Typology-keyed policy excerpts from FFIEC / FinCEN | `list[PolicyExcerpt]` |
| `get_sop` | Bank's internal SOP for the typology | `list[SOPExcerpt]` |

### Sample tool outputs

**Transactions** (typically 5–50 rows per case):
```json
[
  { "date":"2026-02-04", "amount":9793.00, "currency":"USD",
    "counterparty":"AC501879", "channel":"cash", "notes":"" },
  { "date":"2026-02-04", "amount":9807.00, "currency":"USD",
    "counterparty":"AC501879", "channel":"cash", "notes":"" },
  { "date":"2026-02-05", "amount":9791.00, "currency":"USD",
    "counterparty":"AC501879", "channel":"cash", "notes":"" }
]
```

**KYC profile**:
```json
{
  "entity_id": "SYN_6b558d20",
  "entity_type": "business",
  "expected_monthly_volume": 35486.0,
  "business_purpose": "Professional / B2B services firm; primarily card and wire receipts.",
  "risk_rating": "low",
  "incorporation_jurisdiction": "US-CA"
}
```

**Sanctions screening hits** (empty in most cases):
```json
[
  { "name":"AC501879", "list":"OFAC SDN", "score":0.87,
    "match_type":"name_partial", "is_pep":false }
]
```

**Policy excerpt** (retrieved by typology):
```json
{
  "source": "FFIEC BSA/AML Manual",
  "section": "Structuring — Detection Indicators",
  "url": "https://bsaaml.ffiec.gov/...",
  "text": "Multiple cash deposits or withdrawals just below the $10,000 reporting threshold..."
}
```

A deterministic helper then computes a **semantic profile** (typology,
regulatory frame, activity descriptor) from these raw inputs. This profile
is used **internally** to route the right policy / statute to the model
— it is NOT shipped to the model in the SAR bundle, so the model must
derive the verdict from the raw evidence alone (no hint-following).

---

## 5. Phase 2 — Auxiliary skills (4 parallel LLM calls)

The orchestrator fires four specialised LLM calls **in parallel**, each
invoking the same trained model under a different `task_type` system prompt.
Every call returns one typed JSON finding that gets bundled into the SAR
judgment input.

| Skill | Input | Output | Role in SAR |
|---|---|---|---|
| `auxiliary_behavioral` | `[transactions]` + `[kyc_profile]` + precomputed metrics | `{ summary, metrics, evidence }` | Narrative summary of *what the activity looks like* |
| `auxiliary_numeric` | `[transactions]` + `[kyc_profile]` + typology-specific question | `{ answer, calculation, evidence }` | *The quantitative red flag* (sums, ratios, velocities) |
| `auxiliary_citation` | `[policy_excerpt]` + question | `{ answer, evidence_span }` | Verbatim regulatory quote that the SAR will cite |
| `auxiliary_statutory` | `statute` + `fact_pattern` + question | `{ answer, label, reasoning }` | Whether the conduct meets a specific statute's elements (entailment / contradiction / neutral) |

### Sample auxiliary findings (cash-structuring scenario above)

**Behavioral**:
```json
{
  "summary": "Entity SYN_6b558d20 executed 3 cash deposits totaling $29,391 over 2 days at counterparty AC501879. All deposits were within $300 of the $10,000 CTR threshold, with tx_count=3, tx_total_usd=29391.0, channel_mix={'cash': 1.0}, velocity_24h_max=2, vs_declared_volume_ratio=0.83. Pattern is consistent with cash structuring.",
  "metrics": { "tx_count":3, "tx_total_usd":29391.0,
               "channel_mix":{"cash":1.0}, "velocity_24h_max":2,
               "vs_declared_volume_ratio":0.83, "loop_detected":false },
  "evidence": "transactions[0..2]; kyc_profile.expected_monthly_volume"
}
```

**Numeric**:
```json
{
  "answer": "total cash deposits 29391.00 USD (0.83x declared monthly)",
  "calculation": "1. transactions[0].amount = 9793.00. 2. transactions[1].amount = 9807.00. 3. transactions[2].amount = 9791.00. 4. Sum = 29391.00. 5. kyc_profile.expected_monthly_volume = 35486.00. 6. Ratio = 29391/35486 = 0.83.",
  "evidence": "transactions[0..2]; kyc_profile.expected_monthly_volume"
}
```

**Citation**:
```json
{
  "answer": "Section identifies multiple cash deposits below the $10,000 threshold as a structuring indicator.",
  "evidence_span": "Multiple cash deposits or withdrawals just below the $10,000 reporting threshold"
}
```

**Statutory**:
```json
{
  "answer": "Yes — the conduct falls within 31 U.S.C. § 5324(a)(3).",
  "label": "entailment",
  "reasoning": "§ 5324(a)(3) prohibits structuring transactions to evade the $10,000 CTR reporting requirement. The three cash deposits of $9,793 / $9,807 / $9,791 sit within $300 of the threshold, at one counterparty, across two days — consistent with the statute's purpose-of-evading element."
}
```

---

## 6. Phase 3 — SAR judgment (the final decision)

The orchestrator assembles all of the above into a single bundle and makes
one call to the custom model. **This is the only place the model produces a
SAR decision.**

### SAR judgment input (the bundle)

A 7-key user message — no hint fields, no decision targets, just the
evidence the model must reason from:

```json
{
  "task_type": "sar_judgment",
  "transactions":      [ <tx>, ... ],
  "kyc_profile":       { ... },
  "sanctions_pep_hits": [ <hit>, ... ],
  "policy_excerpts":   [ <excerpt>, ... ],
  "sop_excerpts":      [ <sop>, ... ],
  "auxiliary_findings": {
    "behavioral": [ <finding> ],
    "numeric":    [ <finding> ],
    "citation":   [ <finding> ],
    "statutory":  [ <finding> ]
  }
}
```

### SAR judgment output

```json
{
  "is_suspicious": true,
  "suspicious_activity_report": "On 2026-02-04 and 2026-02-05, entity SYN_6b558d20 conducted three cash deposits of $9,793, $9,807, and $9,791 to counterparty AC501879, totaling $29,391. Each deposit sat within $300 of the $10,000 CTR reporting threshold; the pattern of multiple sub-threshold cash deposits at a single counterparty within 48 hours is consistent with structuring as defined under 31 U.S.C. § 5324(a)(3). The activity represents 0.83x the entity's declared monthly volume despite the entity's KYC profile describing a primarily card- and wire-based business with no expected cash activity. We defer the final filing decision to the human investigator."
}
```

**Decision rule the model was trained on**:
- `is_suspicious=true` → narrative is 250–800 chars, cites specific tx
  amounts/dates/counterparties from the bundle, names the regulatory frame
  (e.g., "31 U.S.C. § 5324"), uses qualified language ("consistent with",
  "warrants").
- `is_suspicious=false` → narrative is the *disposition rationale* — names
  the surface red flag and explains the disambiguator (KYC consistency,
  common-name PEP, historical pattern, etc.). Same length band.

### Negative-case sample (same workflow, common-name PEP)

```json
{
  "is_suspicious": false,
  "suspicious_activity_report": "Sanctions screening returned a name match for 'John Smith' against an OFAC SDN entry, but cross-referenced KYC (US-based individual, no jurisdictional overlap with the SDN entity, common-name profile) and 90-day transaction history (consistent with declared salary income, no anomalous flows) indicate this is a common-name false-positive. Activity is consistent with the customer's profile and warrants no SAR filing. Final disposition deferred to the human investigator."
}
```

---

## 7. What makes this "agentic" (and why it matters for the workshop)

| Property | How it shows up here |
|---|---|
| **Tool use** | 5 deterministic data-plane tools, fan-out parallelism on sanctions |
| **Skill composition** | 4 specialised LLM skills, each with a typed I/O contract |
| **Stateful reasoning** | Each phase's output becomes the next phase's grounded input |
| **Auditability** | Every tool call + LLM call + retrieved policy + finding is captured in a `CaseTrace` for compliance review |
| **Deterministic orchestration** | NAT orchestrator runs a fixed DAG — no LLM-driven tool-call planning — so behavior is reproducible across runs |
| **Observability** | OpenTelemetry traces flow into Phoenix for per-case timing, token usage, and reasoning inspection |

---

## 8. Compliance posture

| Concern | How it's addressed |
|---|---|
| **Grounding** | Every claim in the SAR narrative must cite a tx index, KYC field, policy span, or statute section — enforced at training time, verifiable post-hoc from `CaseTrace` |
| **Objectivity** | Trained on regulator-grade language ("consistent with", "warrants") with adversarial denylist for "definitely", "obviously", etc. |
| **Reproducibility** | Same alert + same data plane → byte-identical output (seeded, temperature=0 in the SAR call) |
| **Human-in-the-loop** | Every SAR narrative ends with deferral to the human investigator; the agent's role is recommendation + evidence assembly, not autonomous filing |
| **Adversarial robustness** | Model trained on `adversarial_aux` records where one aux finding is deliberately wrong; the model must detect inconsistencies and re-derive from raw inputs |

---

## 9. Why a custom-trained model

A base model can produce plausible-sounding SAR narratives but cannot, out
of the box:

- Cite specific transaction indices and metric values verbatim (grounding)
- Maintain the regulator-grade tone consistently across thousands of cases
- Refuse to invent quantities not present in the bundle
- Detect adversarial / inconsistent auxiliary findings and re-derive

The custom SFT/RL training corpus (75K records, balanced 45/55 positives /
negatives, 8 typologies, 4 auxiliary skill types) is the lever that makes
the model behave like an AML investigator rather than a generic LLM with a
SAR prompt template. The application backend is the harness that exposes
that capability behind a stable REST API.

---

## 10. One-slide summary for the workshop

> **Alert → 5 tools → 4 skills → 1 SAR decision → 1 audit trace.**
>
> Built on the NeMo Agent Toolkit. Powered by a custom-trained NVIDIA NIM
> investigator model. Every byte of the decision is grounded, cited, and
> auditable.

---

## 11. How this application was built — the 7 phases

The application was assembled in a strict sequential workflow. Each phase
consumes the previous phase's output; nothing runs out of order.

```
1.data_download → 2.data_processing → 3.cpt → 4.sdg_sft → 7.run_sft → 9.custom_model_deployment → 10.appln_buildout
   raw data       clean JSONL       continued      synthetic       supervised    deployed NIM         agent +
   (~300 GB)      + token stats     pre-training   SFT corpus      fine-tuning   endpoints            frontend
                                    on Nemotron-   (~75K records)  (cpt → sft)
                                    3-Nano
```

Each section below is one slide.

---

### Phase 1 — `1.data_download` (raw data acquisition)

**Goal**: pull every byte of source data the AML investigator will ever be
trained or evaluated against, in a reproducible layout.

| | |
|---|---|
| Inputs | HuggingFace tokens, Kaggle creds, NGC key |
| Sources | HF datasets (FinQA / TAT-QA / FinanceBench / LegalBench / IBM-AML / SARSum), FinCEN + FATF + OFAC + FFIEC PDFs, data.gov + OpenSanctions CSVs, ICIJ FinCEN Files, AMLGentex generator repo |
| Output layout | `data/raw/<phase>/<source>/` + `data/phase_stats.json` |
| Size | ~300 GB on disk |
| Compute | CPU-only, network-bound |

This phase is intentionally separated from any preprocessing — re-running
it should be unnecessary unless a source updates.

---

### Phase 2 — `2.data_processing` (clean + tokenize)

**Goal**: turn raw downloads (PDFs, CSVs, mixed-format JSONLs) into a
single canonical JSONL layout grouped by what they'll be used for.

| | |
|---|---|
| Tool | `run_extraction.py` — schema-aware per-source extractors |
| Heavy lift | `pypdfium2` parallel PDF extraction (`PDF_WORKERS=64` on 240-core box) |
| Token counter | Nemotron-3-Nano tokenizer (matches the eventual training tokenizer) |
| Output layout | `data/cpt/level_1/`, `data/cpt/level_2/` (CPT corpora), `data/sft/` (instruction Q&A seeds), `data/transactional/` (tool reference data) |
| Reports | `summary.json` + `extraction.log` |
| Compute | CPU-only, no GPU |

After this phase, the rest of the pipeline never touches raw PDFs again.

---

### Phase 3 — `3.cpt` (continued pre-training)

**Goal**: take the off-the-shelf Nemotron-3-Nano and teach it the *language*
of AML / financial regulation before any task-specific training.

Two sub-stages:

**3.1 — `1.data_curation`**
| | |
|---|---|
| Input | `2.data_processing/data/cpt/level_{1,2}/<source>.jsonl` |
| Action | Joint dedup across L1+L2, then layer-segregate at write time (FineWeb / Dolma pattern) |
| Output | Deduplicated per-source corpus ready for tokenize+pack |

**3.2 — `2.run_cpt`**
| | |
|---|---|
| Base | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |
| Strategy | 2-level CPT: L1 (broad financial register) → L2 (AML-specific) |
| Hardware | 8× H100 NVL, nemo-automodel container |
| Output checkpoint | `cpt-l2-final` (`checkpoints/LOWEST_VAL/model/consolidated/`) — the base for SFT |

Output of Phase 3 is the **language-adapted base model** — it now reads
SAR filings, FFIEC manuals, and BSA statutes natively. It still doesn't
know how to *do* anything; that's Phase 5.

---

### Phase 4 — `4.sdg_sft` (synthetic SFT corpus generation)

**Goal**: synthesise the ~75K labeled training records that will teach the
model to behave like an AML investigator.

| | |
|---|---|
| Teacher LLM | `google/gemma-4-31b-it` (deployed locally as 4× vLLM replicas behind nginx) |
| Generation framework | NVIDIA DataDesigner (DD) |
| Pipeline (v3) | 2 parallel sub-pipelines |
| Non-aux pipeline (9 stages) | Stage 1 drivers → 2 KYC → 3 transactions → 4 sanctions → 5 grounding → 6 inline aux findings → 7 SAR judgment → 8 adversarial mutations → 9 validate + consolidate |
| Aux pipeline (4 stages) | A1 extract → A2 generate → A3 assemble + A3b behavioral → A4 audit |
| Output corpus | ~75K records balanced 45/55 positives / negatives across 8 typologies, with 4 aux task types in a 24/24/24/28 mix |
| Quality gates | per-record validators, LLM judge reviewer, corpus-level distribution audits, MinHash dedup |

This is the most complex phase — see `4.sdg_sft/run-v2/SDG_STRATEGY_SFT.md`
for the full spec. The output is two JSONL deliverables that Phase 5
consumes directly.

---

### Phase 5 — `7.run_sft` (supervised fine-tuning)

**Goal**: fine-tune the CPT checkpoint on the Phase 4 corpus so the model
emits exactly the JSON shapes the agent expects.

| | |
|---|---|
| Base checkpoint | `cpt-l2-final` (from Phase 3) |
| Training data | `data/raw/` (chat-format JSONL from Phase 4); `data/check_format_fix.py` validates / repairs malformed records |
| Pipeline | shuffle → analyse → filter → train |
| Hardware | 8× A100 SXM (80 GB / GPU, NVSwitch full mesh) |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` |
| Recipe | `recipe_a100sxm-8.yaml` (FP8 disabled, EP=8, dp_size=8, torch_mm experts) |
| Wall clock | ~6–10 hours for a 100K-record corpus at `packed_sequence_size=5350`, 1 epoch |
| Output checkpoint | `sft-final` (`checkpoints/LOWEST_VAL/model/consolidated/`) — the deployable model |

After this phase, the model can produce typed SAR judgments and the 4
auxiliary findings on demand.

---

### Phase 6 — `9.custom_model_deployment` (NIM endpoints)

**Goal**: expose the trained checkpoint as a production-grade OpenAI-compatible
HTTP endpoint, plus a reference base-model endpoint for A/B comparison.

| | Endpoint A (custom) | Endpoint B (base) |
|---|---|---|
| Container | `nvcr.io/nim/nvidia/model-free-nim:2.0.5` | `nvcr.io/nim/nvidia/nemotron-3-nano:latest` |
| GPUs | 0, 3 (NVLink pair, TP=2) | 1, 2 (NVLink pair, TP=2) |
| Host port | 8088 | 8089 |
| Served model name | `aml-custom-task-nim` | `nvidia/nemotron-3-nano` |
| Purpose | Powers the agent's reasoning | Side-by-side comparison: "what did our training buy us?" |

Both endpoints share the same 8× H100 NVL host with carefully chosen
NVLink-paired GPU assignments. The dual-endpoint setup is what lets the
backend run side-by-side comparison evaluations
(`/api/demo/eval/compare`).

---

### Phase 7 — `10.appln_buildout` (agent backend + frontend)

**Goal**: wrap the NIM endpoints in a deterministic agent orchestrator and
expose it as a REST API the workshop can demo.

| | |
|---|---|
| Backend framework | NVIDIA NeMo Agent Toolkit (NAT) + FastAPI |
| Orchestrator | `aml_app.workflow.investigate_case` — fixed deterministic DAG, no LLM-driven tool planning |
| Skills wired | 5 data-plane tools + 4 auxiliary LLM skills (`aux_call`) + 1 SAR caller (`sar_caller`) |
| Schemas | Pydantic models for every tool input / output (transaction, KYC, sanctions hit, policy excerpt, SOP excerpt, all 4 aux findings, SAR judgment) — single source of truth shared with SDG |
| Observability | OpenTelemetry → Phoenix; per-case `CaseTrace` written to `data/traces/` |
| Reference docs | `backend.md` (architecture), `api_documentation.md` (every endpoint), `application.md` (this file) |
| Frontend | `frontend/` directory — the workshop UI for triggering investigations and inspecting traces |

After Phase 7 the agent is production-shaped: POST an alert in, get a
fully-grounded SAR decision out, with the full audit trail. The workshop
demo is built on top of this API.

---

## 12. End-to-end build picture (one paragraph)

> Phase 1 pulls raw AML data from 20+ sources. Phase 2 cleans and tokenizes
> it. Phase 3 continued-pre-trains Nemotron-3-Nano so the base model reads
> AML language fluently. Phase 4 uses gemma as a teacher to synthesise 75K
> labeled SAR-investigator examples. Phase 5 supervised-fine-tunes the
> CPT'd model on that corpus. Phase 6 deploys the result as a NIM
> alongside the unmodified base. Phase 7 builds a deterministic agent
> around the custom NIM and exposes it as a REST API. Every phase's
> output is the next phase's input — the chain runs once, end-to-end, and
> the workshop demo is whatever comes out the right-hand side.
