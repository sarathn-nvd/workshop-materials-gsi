# AML SFT Synthetic Data Generation — Strategy

## 1. Purpose

This document specifies the **supervised fine-tuning (SFT) corpus** for the AML
investigation language model that powers the production AML agent.

It covers three things, in this order:

1. **What the model must be able to do** — the SFT objective.
2. **How the target agentic application uses the model** — the production
   data flow, including tool calls, auxiliary-skill invocation, and the
   final SAR-judgment call.
3. **How the SDG pipeline produces a corpus that teaches the model that
   behaviour** — the stages, their functionality, and how they compose.

Read this before changing any prompt, validator, distribution target, or
stage in `scripts/`.

---

## 2. SFT objective

The model's job at inference time is **end-to-end SAR-judgment**: given a
bundle of AML evidence (transactions, KYC, sanctions screen, retrieved
policy excerpts, SOP excerpts, and any pre-computed auxiliary findings),
output two things:

1. A **binary verdict**: is this case suspicious enough to warrant filing
   a Suspicious Activity Report?
2. A **grounded narrative** explaining the reasoning. The narrative is
   required for **both classes** — a positive narrative explains *why* a
   SAR is warranted; a negative narrative explains *why the surface-level
   alert is not actionable* despite the signals that triggered it.

**Note on typology classification.** The model is *not* asked to emit a
structured typology field in its output — only the 2-field
`{is_suspicious, suspicious_activity_report}`. Typology reasoning happens
*implicitly inside the model* and surfaces only in the narrative prose
(e.g. *"the pattern is consistent with layering pass-through per FFIEC
guidance..."*). This keeps the output schema simple, matches the SFT
corpus shape, and avoids forcing the model to commit to a brittle
discrete-label output that production callers would have to enumerate.

In addition to the SAR-judgment task, the model is trained on **four
auxiliary specialist tasks** that the orchestrator invokes during
investigation. Each task takes a focused passage + question and returns a
typed finding:

| Task type            | What it does                                          | Output |
|----------------------|-------------------------------------------------------|--------|
| `auxiliary_numeric`    | Compute a numeric answer from a transactional bundle | `{question, answer, calculation, evidence}` |
| `auxiliary_citation`   | Find and verbatim-quote the relevant span in a regulatory passage | `{question, answer, evidence_span}` |
| `auxiliary_statutory`  | Apply a statute to a fact pattern; emit an entailment / contradiction / neutral label with reasoning | `{question, answer, label, reasoning}` |
| `auxiliary_behavioral` | Summarise a transactional bundle by citing pre-computed metrics verbatim | `{summary}` |

These auxiliary skills are *separate task types* the model sees during SFT.
At production time the agent invokes them in parallel before the final SAR
call and feeds their typed outputs (after gating) into the SAR bundle.

### 2.1 Two hard rules

**Rule A — No label leakage in the user message.**
The SAR-judgment user message contains *evidence only*. No field in the
input correlates with the verdict by construction. Specifically, the user
message does **not** carry:
- a pre-computed regulatory-frame label
- a pre-computed typology label
- a pre-computed decision target / SAR-recommendation hint

The model must derive its verdict from the underlying evidence alone.

**Rule B — Both classes are grounded.**
Every assistant message has a non-empty `suspicious_activity_report`
narrative of 400–800 characters, regardless of whether the verdict is
positive or negative. Negative narratives explicitly name the surface red
flag that triggered the alert and cite the bundle evidence that resolves
it.

These two rules drive every distribution target, prompt, and validator in
the rest of this document.

---

## 3. The target agentic application

The production agent lives in `10.appln_buildout/backend/` and exposes a
FastAPI surface backed by the NeMo Agent Toolkit (NAT). Below is the
end-to-end flow for a single investigation case.

### 3.1 End-to-end flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT — alert from upstream transaction monitoring                     │
│    {case_id, alert_id, entity_id, investigation_window, trigger}        │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — DATA-FETCH TOOLS (deterministic, no LLM)                     │
│                                                                         │
│    get_transactions(entity_id, window)     → transactions[]             │
│    get_kyc(entity_id)                      → kyc_profile                │
│    screen_sanctions(entity + counterparties) → sanctions_pep_hits[]     │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — INTERNAL TYPOLOGY GUESS (rule layer, never reaches the LLM)  │
│                                                                         │
│    guess_typology = classify_typology(tx, kyc, hits, trigger)           │
│                                                                         │
│  Used INTERNALLY for:                                                   │
│    • choosing the typology-specific question for aux skills (Phase 4)   │
│    • selecting policy + SOP excerpts to retrieve (Phase 3)              │
│                                                                         │
│  Never written into the SAR-judgment user message.                      │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — RETRIEVAL TOOLS (deterministic, no LLM)                      │
│                                                                         │
│    policy_focused = retrieve_policy(typology=guess_typology, k=3)       │
│    policy_broad   = retrieve_policy(no_filter, k=2)         ← fallback  │
│    sop_excerpts   = get_sop(typology=guess_typology)                    │
│                                                                         │
│    policy_excerpts = policy_focused + policy_broad                      │
│                                                                         │
│  The broad-search safety net ensures the model has alternative          │
│  regulatory context to fall back on when the rule-layer typology        │
│  guess is wrong.                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — AUXILIARY SKILLS (4 parallel LLM calls to the trained NIM)   │
│                                                                         │
│    behavioral_agent: aux_behavioral_call(bundle)                        │
│    numeric_agent   : aux_numeric_call(passage, NUMERIC_Q[guess_typ])    │
│    citation_agent  : aux_citation_call(policy_focused[0], CITATION_Q)   │
│    statutory_agent : aux_statutory_call(STATUTE_BY[guess_typ], facts)   │
│                                                                         │
│  Each agent is a tiny ReAct loop wrapping the aux LLM call. The         │
│  user-message into the LLM is the task-type schema only:                │
│    behavioral / numeric / citation: {task_type, passage, question}      │
│    statutory                       : {task_type, statute, fact_pattern, │
│                                       question}                         │
│                                                                         │
│  No typology / frame / label hint is in any aux user message.           │
│  The typology guess only selected WHICH question/passage/statute to     │
│  feed in — the value is never visible to the LLM.                       │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5 — AUX GATE (input guard → schema check → LLM-as-judge)         │
│                                                                         │
│  Each finding goes through three filters:                               │
│    1. Was the input passage present and well-formed?                    │
│    2. Does the output match the typed Pydantic schema?                  │
│    3. Does an independent judge LLM call confirm the finding is         │
│       consistent with its inputs?                                       │
│                                                                         │
│  Findings that pass all three are inlined into the SAR bundle. Findings │
│  that fail any stage are dropped (their absence is recorded in the      │
│  case trace).                                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 6 — SAR JUDGMENT (one LLM call to the trained NIM)               │
│                                                                         │
│  USER MESSAGE — exactly 7 keys, evidence only:                          │
│    {                                                                    │
│      "task_type":          "sar_judgment",                              │
│      "transactions":       [...],                                       │
│      "kyc_profile":        {...},                                       │
│      "sanctions_pep_hits": [...],                                       │
│      "policy_excerpts":    [...],   ← focused + broad                   │
│      "sop_excerpts":       [...],                                       │
│      "auxiliary_findings": {behavioral, numeric, citation, statutory}   │
│                              | null                                     │
│    }                                                                    │
│                                                                         │
│  MODEL OUTPUT — exactly 2 fields:                                       │
│    {                                                                    │
│      "is_suspicious":              <bool>,                              │
│      "suspicious_activity_report": "<grounded 400–800 char narrative,   │
│                                      non-empty for both classes>"       │
│    }                                                                    │
│                                                                         │
│  The model is the actual classifier. Nothing in the input tells it the  │
│  answer; it must derive the verdict from raw evidence alone.            │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 7 — TRACE PERSISTENCE                                            │
│                                                                         │
│  A CaseTrace JSON is written to ./data/traces/<case_id>.json            │
│  containing inputs at every phase, the rule-layer typology guess        │
│  (audit-only), all aux findings (raw + gated), the model output,        │
│  and timing. This is the audit trail consumed by the analytics +        │
│  eval API surface.                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 What each piece is doing

- **Tools (Phases 1 + 3)** are deterministic Python functions registered
  with NAT. They never call an LLM. They return Pydantic-validated payloads
  that the orchestrator passes downstream.
- **Rule layer (Phase 2)** is a small heuristic classifier (`classify_typology`
  + behavioural-metric checks) that produces an internal typology guess.
  This guess routes retrieval and aux-question selection. **It is not an
  input to the SAR-judgment LLM call.**
- **Auxiliary skills (Phase 4)** are four specialist sub-agents. Each one
  is a ReAct loop that may call data tools to gather context, then calls
  the trained NIM with one of the four `auxiliary_*` task types. The
  output is a typed Pydantic finding.
- **Aux gate (Phase 5)** is a three-stage filter that decides which aux
  findings are trustworthy enough to inline. Failed findings are dropped;
  their absence is part of the trace.
- **SAR judgment (Phase 6)** is the final LLM call. The bundle has seven
  keys — no hint fields. The model returns a verdict and a grounded
  narrative. This is the call the SFT corpus is designed to train.
- **Trace persistence (Phase 7)** writes the full audit trail. The trace
  is what the evaluation endpoints (`/api/demo/eval/*`) score against the
  ground-truth labels in `eval_keys.jsonl`.

### 3.3 What this means for the training corpus

Two implications drive everything in §4 and §5:

- The corpus must teach **SAR-judgment on a 7-key bundle** with a 2-field
  output. Every SAR-judgment SFT record must match that exact wire format.
- The corpus must teach **four auxiliary task types** with their typed
  outputs. Each aux SFT record must match its task-type schema.

These are five distinct task types the model is trained on simultaneously.

---

## 4. The training corpus

### 4.1 Composition

| Component                 | Target records | Share of corpus |
|---------------------------|---------------:|----------------:|
| SAR-judgment              |         50,000 |             67% |
| `auxiliary_numeric`       |          6,250 |              8% |
| `auxiliary_citation`      |          6,250 |              8% |
| `auxiliary_statutory`     |          6,250 |              8% |
| `auxiliary_behavioral`    |          6,250 |              8% |
| **Total**                 |     **75,000** |        **100%** |

The SDG pipeline generates ~1.5× the target (≈115K records) so the strict
filter and quota cap can drop sub-quality records and still land at 75K.

### 4.2 SAR-judgment record specification

**Wire format** — every SAR-judgment record is a chat-SFT envelope with
three messages:

```json
{
  "messages": [
    {"role": "system",    "content": "<SAR-judgment system prompt>"},
    {"role": "user",      "content": "<7-key JSON bundle>"},
    {"role": "assistant", "content": "<2-field JSON verdict + narrative>"}
  ],
  "metadata": {
    "typology_gold":  "<one of 9 typologies>",
    "frame_gold":     "<one of 8 frames>",
    "label_gold":     true | false,
    "sar_variant":    "augmented" | "bare" | "adversarial_aux",
    "near_miss":      true | false,
    "source":         "<source-pool id>"
  }
}
```

**User message — exactly 7 keys**:

```json
{
  "task_type":          "sar_judgment",
  "transactions":       [<Transaction>, ...],
  "kyc_profile":        <KYCProfile>,
  "sanctions_pep_hits": [<SanctionsHit>, ...],
  "policy_excerpts":    [<PolicyExcerpt>, ...],
  "sop_excerpts":       [<SOPExcerpt>, ...],
  "auxiliary_findings": {
    "behavioral": [<BehavioralFinding>, ...],
    "numeric":    [<NumericFinding>, ...],
    "citation":   [<CitationFinding>, ...],
    "statutory":  [<StatutoryFinding>, ...]
  } | null
}
```

**Assistant message — exactly 2 fields**:

```json
{
  "is_suspicious":              true | false,
  "suspicious_activity_report": "<400–800 char grounded narrative>"
}
```

#### Distribution targets

| Axis                                  | Target |
|---------------------------------------|---|
| Class balance (`is_suspicious=true`)  | 45% positive / 55% negative |
| Frame: `layering_passthrough`         | ≤ 25% of corpus |
| Frame: `benign`                       | 15–20% |
| Frame: `sanctions`                    | 8–12% |
| Frame: `ctr_structuring`              | 8–12% |
| Frame: `tbml`                         | 8–12% |
| Frame: `trafficking`                  | 6–10% |
| Frame: `elder`                        | 6–10% |
| Frame: `shell`                        | 4–8% |
| Frame: `te`                           | dropped — never emitted |
| Per-frame label balance               | No frame outside `30/70 ↔ 70/30` positive/negative — explicit positives required for `benign`, explicit negatives required for `sanctions` |
| `kyc.risk_rating`                     | low 40% / medium 40% / high 15% / enhanced 5% |
| Channel mix (across all transactions) | ach 40% / wire 25% / card 20% / cash 10% / cheque 5% |
| Jurisdiction                          | US ~80% / international ~20% |
| Median transactions per case          | ~20 |
| Variant mix (per `aux_variant`)       | 70% augmented / 25% bare / 5% adversarial_aux |
| Narrative length (both classes)       | 400–800 chars, median ~600, distributions matched across classes within 20% |
| Near-miss share of negatives          | ≥ 30% |

#### No-leak invariants — enforced by validators

- **No `_decision_target`, `_regulatory_frame`, `_typology_inferred`**, or
  any other rule-layer-verdict field anywhere in the user message.
- **No frame is fully one-sided.** Both `benign` and `sanctions` must
  carry minority-class examples (benign positives = mixed-signal façade
  cases; sanctions negatives = common-name PEP / OFAC-collision /
  pre-listing-date cases).
- **Negative narratives are non-empty.** Every `is_suspicious=false`
  record carries a grounded rationale that names the surface red flag and
  cites the bundle evidence resolving it.
- **Length distributions are matched.** Positive and negative narratives
  have comparable median + p90 lengths so narrative length cannot leak the
  label.

### 4.3 Auxiliary record specifications

All four auxiliary task types share the chat-SFT envelope shape:

```json
{
  "messages": [
    {"role": "system",    "content": "<task-type system prompt>"},
    {"role": "user",      "content": "<task-type schema JSON>"},
    {"role": "assistant", "content": "<task-type finding JSON>"}
  ],
  "metadata": {
    "task_type": "auxiliary_<numeric|citation|statutory|behavioral>",
    "source":    "<source-pool id>"
  }
}
```

The user-message schemas are task-type-specific:

| Task type              | User-message keys                                              | Assistant output                            |
|------------------------|----------------------------------------------------------------|---------------------------------------------|
| `auxiliary_numeric`    | `{task_type, passage, question}`                               | `{question, answer, calculation, evidence}` |
| `auxiliary_citation`   | `{task_type, passage, question}`                               | `{question, answer, evidence_span}`         |
| `auxiliary_statutory`  | `{task_type, statute, fact_pattern, question}`                 | `{question, answer, label, reasoning}`      |
| `auxiliary_behavioral` | `{task_type, passage, question}`                               | `{summary}`                                 |

No aux user message contains a typology hint, regulatory-frame hint, or
label hint. The model derives its finding from the passage + question only.

#### Source mix per aux task

| Task                  | Source mix                                                                                              | AML-relevance |
|-----------------------|---------------------------------------------------------------------------------------------------------|---------------|
| `auxiliary_numeric`    | 100% synthesised over SAR-pipeline transactional bundles. Questions are AML-typology-specific arithmetic. | 100%          |
| `auxiliary_citation`   | 40% FFIEC manual, 40% sarsum SAR-narrative regulatory-citation contexts, 20% EFC SAR-bundle context. Per-source cap 50%. | 100%          |
| `auxiliary_statutory`  | 60% LegalBench SARA (real tax-law statutory reasoning — transfers the *reasoning pattern*) + 40% DataDesigner-synthesised AML-statute snippets (BSA §§5311–5332, §5324, FinCEN regs, sanctions law). Synthesised records flagged in metadata. | ~70% (40% direct, 30% via reasoning transfer) |
| `auxiliary_behavioral` | 100% synthesised over SAR-pipeline transactional bundles + pre-computed metrics. Summaries must cite each metric value verbatim. | 100%          |

#### Per-task quality invariants

- **Numeric**: the gold answer is computed deterministically from the
  passage; the generator's answer must agree within rounding tolerance.
  Calculation must reference passage values verbatim.
- **Citation**: `evidence_span` must be a verbatim substring of the
  passage (≤ 250 chars).
- **Statutory**: the gold label is derived from the source pool (for SARA
  records) or held back during generation and validated post-hoc (for
  synthesised AML statutes). `reasoning` must cite the statute by section
  identifier.
- **Behavioral**: every quantitative claim in the summary must be a
  verbatim copy of a value from the pre-computed metrics block. CTR /
  structuring framing is only valid when `cash > 0` in the channel mix.

---

## 5. The SDG pipeline

The pipeline produces the corpus in §4 by running two pipelines in
parallel: the **SAR pipeline** (9 stages) for the SAR-judgment records and
the **auxiliary pipeline** (4 stages) for the four aux task types. Both
write into `data/interim/` per-stage; the final stage of each emits a
JSONL deliverable into `data/final/`; a strict-filter + quota-cap step
trims to the 75K target.

Every LLM-driven stage uses **DataDesigner** against the local
`gemma-4-31b-it` cluster at `localhost:8080`. All endpoint / model / token
parameters are sourced from `scripts/common/dd_helpers.py`.

### 5.1 Orchestration

`scripts/main.py` is the single entry point:

```bash
python -m scripts.main --total-records 115000
```

It spawns two `multiprocessing.Process` children — one per pipeline — and
joins them at the end. The two pipelines share the vLLM cluster's
concurrency budget; the orchestrator sets the per-pipeline cap so neither
starves the other.

After both pipelines complete:

```bash
python -m scripts.audit all          # corpus + per-record + judge audits
python -m scripts.filter_and_cap     # strict filter, dedup, cap to 75K
```

### 5.2 SAR pipeline (9 stages)

Each stage reads the previous stage's parquet artifact and writes its own.
Stage gates run per-record validators inline and emit a manifest with
pass / fail counts.

| Stage | Functionality |
|-------|---------------|
| **1 — Drivers** | Sample the driver tuple (typology, label, severity, surface pattern, aux variant, entity archetype) per record, respecting (a) per-path budget shares against the source pools (EFC, IBM, AMLGentex, SARSum, CFPB, plus a synthetic typology-fill path), (b) per-typology label biases for the rare typologies (HT/TF/EE), (c) the v3 corpus marginals from §4.2, and (d) hard-zero compatibility rules (`typology=none ⇒ label=false`, `near_miss ⇒ label=false`, typology↔archetype compatibility matrix). Each record carries an opaque `source_payload` blob with the raw pool row for downstream stages to project from. |
| **2 — KYC** | Fill the nullable KYC fields from Stage 1: `incorporation_jurisdiction` (US-heavy, ~80% US per §4.2), `risk_rating` (low/medium-heavy per §4.2), `expected_monthly_volume` (per-archetype bands), and the prose `business_purpose` (DataDesigner-generated, archetype-specific). |
| **3 — Transactions** | Produce the `transactions[]` array via three paths: direct read for pool records that carry rich transaction lists; deterministic template synthesis keyed on typology + severity for the synthetic-fill path; DataDesigner LLM extraction for prose-source records (SARSum narratives, CFPB complaints). Channel mix follows §4.2's distribution per typology. Case-size targets the median-20 tx target. The extractor is **label-aware**: when `label=false` the generated transactions look like ordinary banking activity (no sub-CTR cash splitting, no high-risk counterparties); when `label=true` the transactions reflect the typology's pattern. |
| **4 — Sanctions** | Run a deterministic fuzzy match (`rapidfuzz` `token_set_ratio`) of every counterparty in the record against the local OFAC + PEP pools. Emit hits at score ≥ 0.55. For records whose driver tuple is `_regulatory_frame=sanctions`, ensure ≥ 1 high-confidence hit; for sanctions-negatives (the common-name PEP / OFAC-collision / pre-listing-date subset that breaks the sanctions=100%-positive pattern), inject a noise hit at score 0.85–0.94 with no other red flags. |
| **5 — Grounding** | Retrieve policy excerpts and SOPs keyed on the driver typology. The retrieval is stratified across the four policy sources (FinCEN, FFIEC, FATF, OFAC) so no excerpt set is dominated by a single source. `typology=none` records get no policy excerpts (the rule layer would not retrieve any in production). |
| **6 — Aux findings** | Construct the per-record `auxiliary_findings` block according to the `aux_variant` driver. **Findings are generated inline per SAR record** via DataDesigner — the four task types here share their *schema* and *system prompts* with the standalone auxiliary corpus (§5.3) but are *separately generated*, tailored to this specific SAR bundle's transactions / KYC / policy / sanctions context. `augmented` records get a per-record combo (`AUX_COMBOS` in `config.py` controls which subset of {numeric, citation, statutory} is populated, mixed at the corpus level). `bare` records get `auxiliary_findings=null`. `adversarial_aux` records get one deliberately wrong finding (numeric flipped, statutory label inverted, or citation evidence-span swapped) — the model is trained to detect the inconsistency and re-derive from raw inputs. |
| **7 — Assemble** | Build the chat-SFT envelope. The system prompt is the SAR-judgment v3 system prompt that specifies the 7-key user schema, the 2-field output schema, and the both-classes-narrative-required rule. The user message is the 7-key JSON bundle (no `_regulatory_frame`, no `_typology_inferred`, no `_decision_target`). The assistant message is built by invoking DataDesigner once per record to generate the grounded narrative — for both positive and negative records — and wrapping it with the `is_suspicious` verdict from the driver. Per-record rules enforce: narrative non-empty for both classes, narrative length 400–800 chars, no objectivity-denylist phrases, narrative grounding (every cited fact appears in the bundle), positive-class verbatim-citation rule for augmented variants, negative-class disposition-marker rule (the negative narrative must name the surface red flag and cite a disambiguator). Re-roll on length / objectivity / grounding failures; drop on persistent failures. |
| **8 — Adversarial regen** | For records where the `aux_variant=adversarial_aux` variant produced a narrative that did not contain a detection marker (the assistant should have flagged the deliberately wrong finding), regenerate the narrative once with a stricter system prompt that mandates verification + inconsistency flagging. Drop on second failure. |
| **9 — Validate + consolidate** | Run the corpus-level audits: class balance ≤ 2pp from the 45/55 target, per-frame share within range, per-frame label balance within `30/70 ↔ 70/30`, layering cap at ≤ 25%, near-miss floor at ≥ 30% of negatives, per-typology positive coverage (every canonical typology has ≥ 1 positive record), no source > 30% of the corpus. Apply MinHash deduplication at Jaccard ≥ 0.85 on `user + assistant` content. Emit the final `sar_judgment_non_auxillary_corpus.jsonl`. |

### 5.3 Auxiliary pipeline (4 stages)

The aux pipeline runs in parallel with the SAR pipeline. It has its own
stage numbering (`A1`–`A4`).

| Stage | Functionality |
|-------|---------------|
| **A1 — Extract** | Read the available local source pools per task type and project each row into a uniform per-record format `{task_type, source, raw_payload}`. The source mix per task type follows §4.3: numeric / behavioral records project from SAR-pipeline transactional bundles (read from `data/interim/nonaux/stage_3_transactions.parquet`); citation records project from FFIEC manual chunks + sarsum SAR-narrative regulatory-citation contexts + EFC SAR-bundle context; statutory records project from LegalBench SARA passages + synthesised AML-statute snippets. The synthesised AML statutes are generated by DataDesigner as part of this stage (gemma writes a plausible BSA / FinCEN / sanctions statute snippet given a section identifier; the snippet is held back as `metadata.synthetic_aml_statute=true` for audit). |
| **A2 — Generate** | Run DataDesigner once per record to produce the task-type-specific finding. The generation prompts derive the answer from the passage only — the gold answer (where available from the source pool) is never pre-supplied to the LLM; it is used only for post-validation. This prevents the model from learning to back-rationalise pre-supplied answers. Each task type uses its own system prompt: financial-analyst prompt for numeric (step-by-step calculation + evidence cells), regulatory-analyst prompt for citation (verbatim evidence_span ≤ 250 chars), legal-analyst prompt for statutory (entailment/contradiction/neutral label + section-identifier reasoning), and behavioral-analyst prompt for behavioral (4–8-sentence summary citing every metric verbatim). |
| **A3 — Assemble** | Validate every generated finding against its Pydantic schema and post-validate against the gold (where available). Numeric records are dropped when the model's answer disagrees with the gold beyond rounding tolerance. Statutory records are dropped when the model's derived label disagrees with the gold label. Citation records are dropped when the evidence_span is not a verbatim substring of the passage. Behavioral records pass through to A3b. Then wrap each surviving record into the chat-SFT envelope with the task-type system prompt and the schema-compliant user message. |
| **A3b — Behavioral** | A dedicated sub-stage for `auxiliary_behavioral` because its generation has stricter constraints. The pre-computed metrics block is derived from the SAR-pipeline transactional bundle by `scripts/common/behavioral_features.py`. The DataDesigner prompt enforces the metric-verbatim rule and the channel-coherent regulatory-framing rule (no CTR / structuring language when `cash_present=False`). The reviewer step inside this stage rejects summaries that paraphrase metrics or invoke wrong-channel framing; re-roll once, then drop. |
| **A4 — Audit** | Run the corpus-level audits for the aux side: schema-pass rate per task type (≥ 95%), per-task source-cap (no single source > 50% per task type for citation/statutory, no per-task cap for numeric/behavioral which are 100% synthesised), task-type balance against §4.1 targets, per-task content-length distributions sane (no degenerate single-token outputs). Emit the final `auxiliary_corpus.jsonl`. |

### 5.4 How the two pipelines stitch together

```
                ┌─────────────────────────────────────────────┐
                │ main.py — spawns 2 multiprocessing.Process  │
                └─────────────────────────────────────────────┘
                              │                    │
            ┌─────────────────┘                    └──────────────────┐
            ▼                                                         ▼
   ┌─────────────────┐                                       ┌─────────────────┐
   │ SAR pipeline    │                                       │ AUX pipeline    │
   │ Stage 1 → 9     │                                       │ Stage A1 → A4   │
   └─────────────────┘                                       └─────────────────┘
            │                                                         │
            ▼                                                         │
   data/interim/nonaux/                                               │
     stage_1_drivers.parquet                                          │
     stage_2_kyc.parquet                                              │
     stage_3_transactions.parquet ─────────────────────────────────┐  │
     stage_4_sanctions.parquet                                     │  │
     stage_5_grounding.parquet                                     │  │
     stage_6_aux_findings.parquet ◄──── reads aux corpus to fill ──┼──┤
     stage_7_assemble.jsonl                                        │  │
     stage_8_adversarial.jsonl                                     │  │
     stage_9_validate_consolidate.jsonl                            │  │
            │                                                     │  │
            │   ┌─── SAR Stage A1 reads ───────────────────────────┘  │
            │   │     stage_3_transactions.parquet  (for numeric +    │
            │   │     behavioral aux passages)                        │
            │   ▼                                                     │
            │   data/interim/aux/                                     │
            │     stage_a1_extract.parquet                            │
            │     stage_a2_generate.parquet                           │
            │     stage_a3_assemble.parquet                           │
            │     stage_a3b_behavioral.parquet                        │
            │     stage_a4_audit.parquet                              │
            │            │                                            │
            ▼            ▼                                            │
       data/final/sar_judgment_non_auxillary_corpus.jsonl             │
       data/final/auxiliary_corpus.jsonl                              │
                              │                                       │
                              ▼                                       │
                ┌──────────────────────────────────────────┐          │
                │ scripts/audit.py (per-record + corpus +  │          │
                │ LLM-judge buckets)                       │          │
                └──────────────────────────────────────────┘          │
                              │                                       │
                              ▼                                       │
                ┌──────────────────────────────────────────┐          │
                │ scripts/filter_and_cap.py (strict-judge  │          │
                │ filter + MinHash dedup + 75K quota cap)  │          │
                └──────────────────────────────────────────┘          │
                              │                                       │
                              ▼                                       │
                ┌──────────────────────────────────────────┐          │
                │ data/final/sft_corpus_75k.nonaux.jsonl   │          │
                │ data/final/sft_corpus_75k.aux.jsonl      │          │
                └──────────────────────────────────────────┘          │
                              │                                       │
                              └────────► fed to 7.run_sft training ───┘
```

The cross-pipeline dependency is one-directional: the **aux pipeline
reads from the SAR pipeline's Stage 3 output** to source the
transactional bundles that drive numeric + behavioral aux records.
Because the SAR pipeline's Stages 1–3 run first (they are CPU-bound and
finish before the LLM-bound Stages 6–7), the aux pipeline can start its
A1 extract as soon as Stage 3 commits its parquet — both pipelines stay
busy for most of the wall clock.

### 5.5 Audit and filter

After both pipelines emit their final JSONLs, two post-processing steps
gate what gets shipped:

- **`scripts/audit.py`** runs three audit modes: per-record (re-applies
  every per-record validator and emits a JSONL of per-record verdicts),
  judge (a separate LLM-judge pass that scores narrative grounding,
  objectivity, and class-specific quality — positive narratives are
  scored against grounding rules; negative narratives are scored against
  disposition-marker rules), and corpus (re-applies every corpus-level
  audit and compares against the §4.2 distribution targets). Audit
  failures do not modify the corpus; they emit a report.

- **`scripts/filter_and_cap.py`** consumes the audit reports and the
  generated JSONLs. It drops any record that any rule (per-record or
  judge) flagged. It re-runs MinHash dedup. It then samples down to the
  75K quota while preserving the §4.2 distribution targets (frame caps,
  per-frame label balance, near-miss floor, etc.). Records that would
  break a target on inclusion are skipped; records that are necessary to
  meet a floor are prioritised. The output is the final shippable corpus.

---

## 6. Acceptance criteria

The v3 corpus is shippable when **all** of these are satisfied:

### 6.1 Corpus-level

- Class balance within 2pp of 45/55.
- No frame outside its §4.2 share range.
- No frame outside `30/70 ↔ 70/30` positive/negative balance.
- Per-typology positive coverage: every canonical typology has ≥ 30
  positive records.
- Near-miss share of negatives ≥ 30%.
- No source > 30% of the corpus.
- MinHash dedup removed < 5% of records (high removal indicates a
  generator collapsing onto repeated templates).
- Median narrative chars for positives and negatives within 20% of each
  other (length cannot leak the label).
- Zero records contain `_decision_target`, `_regulatory_frame`, or
  `_typology_inferred` in the user message.
- Zero negative-class records have empty `suspicious_activity_report`.

### 6.2 Per-task aux quality

- Each aux task type ≥ 95% schema-validity pass rate.
- `auxiliary_numeric`: gold-answer agreement ≥ 90%.
- `auxiliary_citation`: evidence_span verbatim-substring rate = 100%
  (any failure is a generator bug — fix and re-run).
- `auxiliary_statutory`: gold-label agreement ≥ 85% on SARA records;
  synthesised AML records pass the LLM-judge sanity check at ≥ 90%.
- `auxiliary_behavioral`: metric-verbatim citation rate ≥ 90%; zero
  records with CTR / structuring language when `cash_present=False`.

### 6.3 Downstream eval

The trained checkpoint is scored end-to-end against the held-out demo set
in `10.appln_buildout/backend/data/demo/eval_keys.jsonl` via
`/api/demo/eval`. The shipping criteria for a checkpoint trained on this
corpus:

- F1 (SAR class) ≥ 0.72 with **no rule-layer calibrator** in the inference
  path (i.e. `OMIT_DECISION_TARGET=1` and `decision_target=""`).
- Recall (SAR class) ≥ 0.87.
- Precision (SAR class) ≥ 0.60.
- 0 JSON parse errors.
- Median negative-class narrative ≥ 400 chars.

If a checkpoint hits the corpus-level criteria (§6.1, §6.2) but misses the
downstream criteria (§6.3), the gap is most likely in distribution drift
between the v3 corpus and the production-mimic distribution — re-tune the
distribution targets in §4.2 and regenerate.

---

## 7. Operational notes

- **LLM endpoint**: `http://localhost:8080/v1`, model `google/gemma-4-31b-it`,
  served by 4 vLLM replicas behind nginx (see `model_deployment/`).
  Overridable via `LLM_ENDPOINT` / `LLM_MODEL` env vars.
- **Concurrency**: `Concurrency.max_parallel_llm=256`,
  `Concurrency.max_cpu_workers=200` per `scripts/config.py`. When SAR + aux
  pipelines run in parallel each gets ~128 concurrent LLM slots.
- **Wall-clock estimate**: a full v3 run at `N=115_000` (1.5× oversample of
  the 75K target) is ~6 h on the current cluster — dominated by Stage 7
  (SAR narrative generation, now running on both positive and negative
  records) and Stage 6 + A2 (aux findings + aux generation).
- **Failure recovery**: every stage is resumable via
  `--resume-from <stage_id>`. Each stage reads its predecessor's parquet/
  jsonl and writes its own; intermediate artefacts are preserved.
- **Audit lockfile**: `data/manifests/_audit_judge_full.lock` prevents
  concurrent judge audits.

---

## 8. Glossary

- **SAR** — Suspicious Activity Report. The end-product of an AML
  investigation; a regulatory filing made when an institution determines
  that activity may constitute money laundering, terrorist financing,
  fraud, or related crime.
- **Typology** — the pattern of AML activity (`structuring`, `layering`,
  `trade_based_ml`, `shell_company`, `human_trafficking`,
  `terrorist_financing`, `elder_exploitation`, `smurfing`, `none`).
- **Regulatory frame** — the regulatory framing under which an activity is
  analysed (`ctr_structuring`, `layering_passthrough`, `tbml`, `shell`,
  `sanctions`, `elder`, `trafficking`, `benign`). One per record;
  internally derived by the rule layer for retrieval routing.
- **Near-miss** — a negative case whose surface evidence looks suspicious
  (sub-CTR cash, foreign wires, escalating volumes) but has a legitimate
  explanation in the KYC / business context. The model must learn not to
  default to SAR on these.
- **Variant** — `augmented` records carry full (correct) auxiliary
  findings, `bare` records carry no findings (model derives from raw
  inputs), `adversarial_aux` records carry one deliberately wrong finding
  (model must detect and overrule).
- **DataDesigner (DD)** — the NVIDIA SDG primitive used to run controlled
  LLM passes with structured input columns. All LLM-driven stages in this
  pipeline are DD passes.
