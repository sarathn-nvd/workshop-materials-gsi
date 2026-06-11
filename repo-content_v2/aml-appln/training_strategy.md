# AML Investigation Agent — Training Pipeline

This document is the focused plan-of-record for the **training pipeline** of a domain-specialized AML (Anti-Money Laundering) investigation agent. Backup of the prior longer version: [`backup/bkp_training_strategy_2ndMay.md`](backup/bkp_training_strategy_2ndMay.md).

## Table of Contents

1. [Overall Training Objective](#1-overall-training-objective)
2. [Datasets In Hand](#2-datasets-in-hand)
3. [Training Phases & Objective Outcomes](#3-training-phases--objective-outcomes)
4. [CPT Phase](#4-cpt-phase)
5. [SFT Phase](#5-sft-phase)
6. [RL Phase](#6-rl-phase)
7. [Data Constraints (SFT data + Agentic Application)](#7-data-constraints-sft-data--agentic-application)
8. [Appendix — Construction Details and Quality Gates](#8-appendix--construction-details-and-quality-gates)

---

## 1. Overall Training Objective

We are building a domain-specialized AML investigation agent. At runtime, the agent gathers five inputs via tool calls and reasons over them to produce a Suspicious Activity Report (SAR) judgment.

**Primary task (`sar_judgment`)** — given a bundle of (transactions, KYC profile, sanctions/PEP hits, policy excerpts, SOP excerpts, optional pre-computed auxiliary findings), produce:

```json
{
  "is_suspicious": true,
  "suspicious_activity_report": "<regulator-grade narrative — only when is_suspicious=true>"
}
```

The narrative must be **objective, evidence-based, non-accusatory**, must cite transaction facts and applicable regulation, and must defer final judgment to the human investigator.

**Auxiliary tasks** — four focused sub-tasks the model is also trained on. The agent (NAT) invokes these mid-investigation **before** the final SAR call, then feeds their outputs back into `sar_judgment` via the `input.auxiliary_findings` block (Section 7.1). This makes grounding **explicit** — the SAR narrative cites pre-computed behavioral analyses, numbers, citations, and statutory conclusions rather than recomputing them.

| Task type | What the model does |
|---|---|
| `auxiliary_behavioral` | Analyze a raw transactional bundle (transactions[] + KYC) and emit a behavioral summary: channel mix, velocity, counterparty concentration, declared-vs-observed volume ratios, loop / pass-through detection, country-risk concentration. Output is a structured `{summary, metrics, evidence}` block where `metrics` are computable from the rows and `summary` is a prose interpretation that cites them. |
| `auxiliary_numeric` | Multi-step arithmetic over financial passages and tables; threshold comparisons. |
| `auxiliary_citation` | Locate and quote exact spans from a long policy passage. |
| `auxiliary_statutory` | Apply a stated rule to a fact pattern and conclude entailment. |

**Why `auxiliary_behavioral` is a model task, not a tool**: behavioral analysis blends deterministic aggregation (counts, sums, channel mix) with interpretive judgment (loop detection, pass-through framing, baseline-vs-observation reasoning). Lifting it into a model task — rather than a separate analytics service — keeps the system self-contained against any raw transaction stream, makes the behavioral summary verifiable by the same model at SAR-time (the model can detect inconsistencies between its own auxiliary output and the raw rows in the bundle), and matches the symmetric pattern of the other three aux tasks. The numeric portion of the output is gold-anchored at training time via deterministic computation (Section 7.4), so the model learns to *reproduce* exact aggregations, not invent them.

**Compositional flow** — gather raw inputs → invoke 0–4 auxiliary calls → assemble `auxiliary_findings` → final `sar_judgment` call. Same model serves all five task types via prompt-template differentiation.

---

## 2. Datasets In Hand

### 2.1 CPT Layer 1 — Broad Financial Register

| Source | Bytes | Records |
|---|---:|---:|
| EDGAR-CORPUS (SIC 6000–6999) | 3,579 MB | 63,366 |
| Pile of Law `oig` (Treasury / FDIC / OCC / Fed) | 277 MB | 20,057 |
| Pile of Law `federal_register` (financial agencies) | 38 MB | 3,808 |
| Pile of Law `sec` | 26 MB | 10,805 |
| Pile of Law `cfr` (Title 12, 17, 31) | 11 MB | 13 |
| `uscode_house` (Title 12, 18, 31) | 32 MB | — |
| Pile of Law `doj_guidance` | 5 MB | 335 |
| Pile of Law `uscode` (Title 12, 18 Ch. 95–96, 31) | 1.3 MB | 2 |

**Combined L1: ~3.97 GB raw → ~615M–1.5B tokens after curation.**

### 2.2 CPT Layer 2 — AML-Specific

| Source | Bytes | Files |
|---|---:|---:|
| FinCEN Advisories | 274 MB | 365 |
| FinCEN Federal Register | 288 MB | 430 |
| FinCEN SAR Reviews | 297 MB | 332 |
| FinCEN Enforcement | 278 MB | 315 |
| FinCEN Files (ICIJ) | 5.7 MB | 91 articles |
| OFAC Guidance | 513 MB | 1,451 PDFs |
| OFAC Enforcement | 1.1 MB | 6 |
| FATF Publications | 32 MB | 16 of 91 PDFs (rest origin-blocked; recovered via Wayback) |
| `courtlistener` (AML opinions) | 1.7 MB | 156 |

**Combined L2: ~1.69 GB raw → ~60–150M tokens after extraction + curation.**

### 2.3 SFT Tier 1 — `sar_judgment` primary

| Source | Bytes | What |
|---|---:|---|
| Enterprise Financial Crime AI | 446 MB | 22,585 alerts + 15,001 investigations + 6,001 SARs + 500K transactions |
| SARSum | 84 MB | 2,000 SAR sets × 7 prose notes × 6 quality tiers; explicit Suspicious / Not concerning labels |
| IBM AML HI-Small | 510 MB | Transactions with `Is Laundering` flag + 7 typology pattern blocks |
| AMLGentex | 26 MB | 10,000 accounts; 678,984 transactions with `isSAR`; 618 alert-pattern instances |

### 2.4 SFT Tier 2 — `auxiliary_*` sources

| Source | Records | Task type |
|---|---:|---|
| FinQA | 6,251 train | `auxiliary_numeric` |
| TAT-QA | ~2K (sampled from ~13K) | `auxiliary_numeric` |
| FinanceBench | ~150 | `auxiliary_citation` |
| LegalBench (16 sub-tasks) | ~3K total | `auxiliary_statutory` |
| FFIEC BSA/AML Manual | 164 HTML pages → ~5K Q/A | `auxiliary_citation` |

### 2.5 SFT Tier 3 — heavy filter / dual-role

| Source | Records | Role |
|---|---:|---|
| Finance-Instruct-500k | 580 MB JSONL (~10–15% retained after AML keyword filter) | `auxiliary_statutory` |
| CFPB Consumer Complaints | 289,714 (filtered) | Negative-case `sar_judgment` synthesis + entity-extraction auxiliary |

---

## 3. Training Phases & Objective Outcomes

| # | Phase | Input | Objective Outcome |
|---|---|---|---|
| 1 | **CPT** | Raw text from §2.1 + §2.2 (no schema, no labels) | Model acquires financial register and AML-domain knowledge. After CPT, the model "speaks regulatory" — recognizes statute citations, FinCEN advisories, FATF typologies, BSA terminology. |
| 2 | **SFT** | Multi-task labeled records — 4 `task_type`s (Section 5) | Model learns to (a) recognize the four prompt schemas, (b) emit the matching output schema for each, (c) compose auxiliary findings into a SAR narrative when present, (d) fall back gracefully when auxiliary findings are absent. |
| 3 | **RL** | Preference pairs + verifiable-reward records | Model is aligned for objectivity (no accusatory phrasing), grounding (cites evidence), JSON validity, and length sanity. Sub-skills sharpened via verifiable rewards on FinQA / FinanceBench / LegalBench held-outs. |

---

## 4. CPT Phase

### 4.1 Two Sub-Levels of Training

CPT is **two sequential sub-passes** on the same checkpoint, curriculum-ordered from broad to narrow:

| Sub-level | Input | Tokens | Purpose |
|---|---|---:|---|
| **Layer 1 — Broad financial register** | §2.1 corpora (EDGAR, US Code, CFR, federal register, OIG/SEC/DOJ guidance) | ~615M–1.5B | Install the financial / regulatory / legal language register the model lacks at base. |
| **Layer 2 — AML-specific** | §2.2 corpora (FinCEN, OFAC, FATF, AML caselaw) | ~60–150M | Narrow into AML domain: typologies, advisories, sanctions guidance, enforcement patterns. |

Layer 1 is run first; Layer 2 starts from the Layer 1 final checkpoint. A mid-Layer-1 checkpoint is preserved as a fallback in case Layer 2 over-narrows the register.

### 4.2 Data Curation Constraints

CPT consumes raw text from §2.1 + §2.2 with **no schema transformation, no per-record content generation**. Curation is heuristic and applies uniformly across both sub-levels:

1. **PDF extraction** — most of L2 (FinCEN, OFAC, FATF) and parts of L1 are PDFs. Extract text + tables + charts; OCR fallback for scanned pages; discard documents where all pages are low-confidence OCR.
2. **HTML / structured-source parsing** — FFIEC manual HTML and EDGAR-CORPUS markup parsed to plain text with section anchors preserved.
3. **Language filter** — keep `en` only.
4. **Length bounds** — reject documents `< 200 chars` or `> 1M chars`.
5. **Quality filter** — minimum `alphanumeric_ratio`, maximum `repeated_line_ratio`, minimum `common_word_ratio`. Strips OCR garbage, table-of-contents noise, and repetitive boilerplate.
6. **Boilerplate removal** — recurring-line detection per source + per-source denylist (e.g., DOJ template headers, FFIEC page headers, FinCEN cover-page seals, EDGAR XBRL footers).
7. **PII scrub** — detector + regex pass replacing detected PII with typed tags (`[SSN]`, `[EIN]`, `[ACCOUNT_ID]`, `[PHONE]`, `[NAME]`). **Mandatory** for FinCEN Files (ICIJ leaks contain real PII), `courtlistener` (case captions name parties), and any CFPB excerpts.
8. **Cross-source overlap awareness** — known statute/regulation duplicates exist between source pairs (`pile_of_law_uscode` ↔ `uscode_house`; `pile_of_law_cfr` ↔ derived 31 CFR Chapter X). These need to be reconciled to a single canonical rendering before training, otherwise the model over-weights statute text.

### 4.3 Sample CPT Shard Line

```jsonl
{"text": "Under 31 U.S.C. § 5324, no person shall, for the purpose of evading the reporting requirements of section 5313(a) or any regulation prescribed under any such section: (1) cause or attempt to cause a domestic financial institution to fail to file a report required under section 5313(a)..."}
```

### 4.4 Output Artifacts

- `cpt-l1-final/` — fallback checkpoint (broad-register only)
- `cpt-l2-final/` — the CPT artifact consumed by SFT

---

## 5. SFT Phase

### 5.1 Multi-Task Training

A single SFT pass on `cpt-l2-final` trains **five `task_type`s** from the same checkpoint via prompt-template differentiation. The model learns to recognize the input shape and emit the matching output shape.

| `task_type` | Objective |
|---|---|
| `sar_judgment` | Given a bundle of (transactions, KYC, sanctions/PEP, policy, SOP, optional auxiliary findings), emit `{is_suspicious, suspicious_activity_report}`. **Primary deployment task.** |
| `auxiliary_behavioral` | Given a transactional bundle (transactions[] + kyc_profile rendered as structured passage), emit a behavioral summary: a prose interpretation of the activity plus a structured metrics block (counts, channel mix, velocity, counterparty concentration, declared-vs-observed volume ratios, loop / pass-through indicators) plus an evidence pointer back to the rows. |
| `auxiliary_numeric` | Given a financial passage and a question, emit a final numeric answer plus a step-by-step calculation trace and the evidence span the number was derived from. |
| `auxiliary_citation` | Given a policy passage and a question, emit an answer plus a verbatim quoted span from the passage as evidence. |
| `auxiliary_statutory` | Given a statute passage and a fact-pattern question, emit a label (entailment / contradiction / neutral) plus a brief justification grounded in the statute and the facts. |

### 5.2 Dataset Distribution

Total SFT corpus: **~60–75K records → ~135–170M tokens** (3 epochs). The auxiliary share rises slightly with the addition of `auxiliary_behavioral`; `sar_judgment` share is held constant.

| `task_type` | Records | Share |
|---|---:|---:|
| `sar_judgment` (positive — suspicious) | 25–30K | ~42% |
| `sar_judgment` (negative — not suspicious) | 15–18K | ~25% |
| `auxiliary_behavioral` | 5–7K | ~8% |
| `auxiliary_numeric` | 5–7K | ~8% |
| `auxiliary_citation` | 4–6K | ~8% |
| `auxiliary_statutory` | 5–7K | ~9% |

**Combined: ~67% `sar_judgment` + ~33% auxiliary** (auxiliary share split roughly evenly across the four sub-types).

**`sar_judgment` variant split** — every `sar_judgment` record is one of three variants depending on `input.auxiliary_findings`:

| Variant | Share of `sar_judgment` | Purpose |
|---|---:|---|
| **Augmented** — `auxiliary_findings` populated with correct findings (any non-empty subset of `behavioral`, `numeric`, `citation`, `statutory`) | ~70% | Teach the model to cite pre-computed findings verbatim; do not recompute. The dominant runtime case. |
| **Bare** — `auxiliary_findings` absent or all sub-arrays empty | ~25% | Teach the model to fall back on internal sub-skills when NAT skipped auxiliary calls. |
| **Adversarial-aux** — `auxiliary_findings` contains ≥1 deliberately wrong finding (numeric flipped, citation swapped, statutory inverted, **or behavioral metrics corrupted**) | ~5% | Teach the model to detect inconsistency against raw inputs; prevent sycophancy to upstream errors. The behavioral variant is uniquely high-signal: the model can recompute the metrics from the same transactions[] in the bundle and detect mismatch directly. |

**Narrative source policy** — gold SAR narratives in `sar_judgment` records come from one of two sources, in this order of preference:

| Source | Where it's used | How |
|---|---|---|
| **SARSum prose notes** | Primary narrative library | Each SARSum case carries 7 prose notes with explicit per-pattern `Suspicious / Not concerning / Not suspicious` labels and a `key_facts` paraphrase set. Indexed by typology × decision. Paired with transactional context at generation time as a soft template the LLM grounds in the bundle. |
| **Hand-authored exemplars** | Rare typologies (terrorist_financing, elder_exploitation, human_trafficking) | Small set (~20–40) anchored on FinCEN advisories and FATF typology reports. Used where SARSum coverage is thin. |

**EFC narrative passthrough is dropped.** The `sar_reports.narrative` field in the Enterprise FC bundle is a templated stub ("The institution identified activity requiring further review involving X CURRENCY from C1 to C2…"); it does not carry regulatory or typology-specific reasoning and is not suitable as gold SAR narrative. EFC is retained as a **transactional context source** for its rich behavioral feature columns (`tx_velocity_24h`, `degree_centrality`, `transaction_loop_flag`, etc.) but its narrative field is not surfaced to Stage 7.

### 5.3 Sample Records (one per task type)

#### `sar_judgment` (positive, augmented variant)

```json
{
  "task_type": "sar_judgment",
  "instruction": "Review the following activity bundle and produce a SAR judgment as JSON. Cite any provided auxiliary_findings verbatim; do not recompute.",
  "input": {
    "transactions": [
      {"date": "2026-04-01", "amount": 9500.00, "currency": "USD", "counterparty": "Cash deposit, Branch #14 Manhattan", "channel": "cash", "notes": "Sub-CTR-threshold cash deposit"},
      {"date": "2026-04-04", "amount": 9700.00, "currency": "USD", "counterparty": "Cash deposit, Branch #21 Queens", "channel": "cash", "notes": "Third sub-threshold deposit in 4 days"},
      {"date": "2026-04-22", "amount": 75000.00, "currency": "USD", "counterparty": "VIRA TRADING LLC, Cyprus", "channel": "wire", "notes": "Outbound wire, recipient not in KYC counterparty list"}
    ],
    "kyc_profile": {"entity_id": "CUST_8814729", "entity_type": "business", "expected_monthly_volume": 50000, "business_purpose": "Retail jewelry store; declared $600K annual revenue; 30% cash receipts", "risk_rating": "medium", "incorporation_jurisdiction": "US-NY"},
    "sanctions_pep_hits": [{"name": "VIRA TRADING LLC", "list": "OFAC", "match_score": 0.78}],
    "policy_excerpts": [{"source": "FinCEN", "section": "Advisory FIN-2014-A005", "url": "https://www.fincen.gov/...", "text": "Multiple cash deposits structured below the $10,000 currency transaction reporting threshold ... may constitute structuring under 31 U.S.C. § 5324."}],
    "sop_excerpts": [{"sop_id": "SOP-STRUCTURING-01", "section": "Step 3", "text": "When sub-$10,000 cash deposit pattern is identified across ≥3 branches ... escalate to SAR filing."}],
    "auxiliary_findings": {
      "numeric": [
        {
          "question": "Sum the cash deposits in the 8-day window April 1–8, 2026 and compare to KYC declared monthly cash volume.",
          "answer": "$57,300 over 8 days, equivalent to ~$214,875 monthly run-rate — 4.3× the declared monthly cash volume of $50,000.",
          "calculation": "1. Cash deposits Apr 1–8: 9,500 + 9,700 + 9,600 + 9,800 + 9,200 + 9,500 = 57,300.\n2. 8-day rate × (30/8) = 57,300 × 3.75 = 214,875 monthly run-rate.\n3. Declared monthly cash volume per KYC = 50,000.\n4. Ratio: 214,875 / 50,000 = 4.30.",
          "evidence": "transactions[0..5], kyc_profile.expected_monthly_volume"
        }
      ],
      "citation": [
        {
          "question": "What FinCEN guidance governs sub-threshold cash deposit patterns by cash-intensive businesses?",
          "answer": "FinCEN Advisory FIN-2014-A005, which addresses BSA/AML risks posed by cash-intensive businesses and describes structuring patterns.",
          "evidence_span": "Multiple cash deposits structured below the $10,000 currency transaction reporting threshold ... may constitute structuring under 31 U.S.C. § 5324."
        }
      ],
      "statutory": [
        {
          "question": "Does the observed pattern of 6 sub-$10,000 cash deposits across 4 branches in 8 days fall within 31 U.S.C. § 5324(a)(3)?",
          "label": "entailment",
          "reasoning": "31 U.S.C. § 5324(a)(3) prohibits structuring transactions with a domestic financial institution for the purpose of evading section 5313(a) reporting requirements. The observed pattern (each deposit deliberately under the $10,000 CTR threshold, spread across 4 branches in 8 days, by a cash-intensive business whose declared monthly cash volume is $50,000 versus an observed $57,300 in 8 days) fits the statutory definition of structuring."
        }
      ]
    }
  },
  "output": {
    "is_suspicious": true,
    "suspicious_activity_report": "Suspicious activity is identified for entity CUST_8814729 (retail jewelry store, NY, declared monthly cash volume $50,000). Between April 1 and April 8, 2026, the entity made 6 sub-$10,000 cash deposits at 4 separate branches totaling $57,300 — equivalent to a ~$214,875 monthly run-rate, or 4.3× the declared monthly cash volume of $50,000. The pattern is consistent with structuring as described in FinCEN Advisory FIN-2014-A005 and falls within 31 U.S.C. § 5324(a)(3), which prohibits structuring transactions to evade section 5313(a) reporting requirements. On April 22, 2026, the entity initiated a $75,000 outbound wire to VIRA TRADING LLC in Cyprus; this counterparty produced an OFAC list match (score 0.78) and is not present in the customer's KYC counterparty list. The combined cash-intensive structuring pattern and outbound wire to a sanctioned-list-adjacent recipient warrant filing a Suspicious Activity Report. Investigator review is recommended for final determination."
  },
  "metadata": {"task_type": "sar_judgment", "sar_variant": "augmented", "typology": "structuring"}
}
```

The SAR narrative cites the numeric ratio (4.3×), the statutory subsection (§ 5324(a)(3)), and the advisory ID **verbatim from `auxiliary_findings`** — it does not recompute them.

#### `auxiliary_behavioral`

```json
{
  "task_type": "auxiliary_behavioral",
  "instruction": "Analyze the transactional activity bundle below and produce a behavioral summary. Cite metric values verbatim; do not invent numbers. Ground every claim in the transactions[] or kyc_profile.",
  "input": {
    "passage": "[transactions]\n2021-11-09 wire 9,322 USD AC501275→AC503656\n2021-11-09 wire 9,350 USD AC501275→AC503656\n2021-11-09 wire 9,462 USD AC501275→AC503656\n2021-11-09 wire 9,507 USD AC501275→AC503656\n2021-11-09 wire 9,374 USD AC501275→AC503656\n2021-11-09 wire 9,439 USD AC501275→AC503656\n2021-11-09 wire 9,319 USD AC501275→AC503656\n2021-11-09 wire 9,541 USD AC501275→AC503656\n\n[kyc_profile]\nentity_id: AC501275\nentity_type: business\nbusiness_purpose: Retail convenience and grocery store, NL\nexpected_monthly_volume: 58,265\nrisk_rating: medium\nincorporation_jurisdiction: NL",
    "context_type": "transactional"
  },
  "output": {
    "answer": {
      "summary": "On 2021-11-09 entity AC501275 (retail grocery store, NL) executed 8 wire transfers to a single counterparty AC503656 totaling 75,314 USD. The 24-hour transaction velocity of 8 wires represents a ≈5.3× spike over the 30-day average of 1.5/day. Channel mix is 100% wire (no cash). Counterparty exposure is concentrated to a single beneficiary (1 unique counterparty in the prior 7 days). Cited transaction total 75,314 USD is ≈1.29× the declared monthly volume of 58,265 — within reasonable range for the declared profile. The pattern is consistent with concentrated pass-through layering rather than CTR-evasion structuring (channels are non-cash; the $10,000 threshold framing does not apply).",
      "metrics": {
        "tx_count": 8,
        "tx_total_usd": 75314.0,
        "channel_mix": {"wire": 1.0, "cash": 0.0, "ach": 0.0},
        "velocity_24h_max": 8,
        "velocity_24h_avg_30d": 1.5,
        "unique_counterparties_7d": 1,
        "amount_z_score_max": 2.3,
        "country_risk_max": 0.0,
        "loop_detected": false,
        "vs_declared_volume_ratio": 1.29
      },
      "evidence": "transactions[0..7]; kyc_profile.expected_monthly_volume"
    }
  },
  "metadata": {"task_type": "auxiliary_behavioral", "source": "enterprise_fc"}
}
```

The `metrics` block is computed deterministically at training time (Pandas aggregations over the source rows). The `summary` is LLM-written prose that cites the metrics verbatim. The `evidence` field indexes back to the input transactions and KYC fields used. At runtime the agent invokes this task on the raw bundle and uses the resulting `{summary, metrics}` as the `behavioral` entry of `auxiliary_findings` in the subsequent `sar_judgment` call.

#### `auxiliary_numeric`

```json
{
  "task_type": "auxiliary_numeric",
  "instruction": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response by relying on the cash flow statement.",
  "input": {"passage": "[3M 2018 10-K cash flow statement excerpt]\nNet cash provided by operating activities ... 6,439 ...\nCash Flows from Investing Activities ...\nPurchases of property, plant and equipment (PP&E) ... (1,577) ...\nProceeds from sale of PP&E and other assets ... 262 ...", "context_type": "mixed"},
  "output": {"answer": "$1,577 million\n\nCalculation:\n1. Locate 'Purchases of property, plant and equipment (PP&E)' on the FY2018 cash flow statement.\n2. Read the FY2018 column value: (1,577) — parenthesized indicating an outflow.\n3. The capital expenditure (CapEx) for FY2018 is therefore $1,577 million.\n\nEvidence: \"Purchases of property, plant and equipment (PP&E) ... (1,577)\""},
  "metadata": {"task_type": "auxiliary_numeric", "source": "finqa"}
}
```

#### `auxiliary_citation`

```json
{
  "task_type": "auxiliary_citation",
  "instruction": "Which FFIEC BSA/AML Manual provision defines the threshold for filing a Currency Transaction Report?",
  "input": {"passage": "[FFIEC BSA/AML Examination Manual — Currency Transaction Reporting]\nThe Bank Secrecy Act (BSA) requires financial institutions to file CTRs ... financial institutions are required to file a Currency Transaction Report (CTR) for each cash transaction in currency of more than $10,000 conducted by, through, or to the financial institution, on a single business day, by or on behalf of the same person. The reporting institution must aggregate transactions ...", "context_type": "text"},
  "output": {"answer": "The FFIEC BSA/AML Examination Manual section on Currency Transaction Reporting establishes the $10,000 single-business-day threshold for CTR filing.\n\nEvidence: \"financial institutions are required to file a Currency Transaction Report (CTR) for each cash transaction in currency of more than $10,000 conducted by, through, or to the financial institution, on a single business day, by or on behalf of the same person.\""},
  "metadata": {"task_type": "auxiliary_citation", "source": "ffiec"}
}
```

#### `auxiliary_statutory`

```json
{
  "task_type": "auxiliary_statutory",
  "instruction": "Does the following hold? The bank's customer made 15 cash deposits of $9,500 across 5 different branches over 8 days for the apparent purpose of avoiding the $10,000 CTR threshold. This conduct falls within 31 U.S.C. § 5324(a)(3).",
  "input": {"passage": "31 U.S.C. § 5324(a) — No person shall, for the purpose of evading the reporting requirements of section 5313(a) ...:\n(1) cause or attempt to cause a domestic financial institution to fail to file a report ...;\n(2) cause or attempt to cause a domestic financial institution to file a report ... that contains a material omission or misstatement of fact; or\n(3) structure or assist in structuring, or attempt to structure or assist in structuring, any transaction with one or more domestic financial institutions.", "context_type": "statute"},
  "output": {"answer": "Entailment\n\nReasoning: 31 U.S.C. § 5324(a)(3) prohibits any structuring of transactions with a domestic financial institution for the purpose of evading the section 5313(a) reporting requirements. The customer's pattern of 15 cash deposits of $9,500 — each deliberately below the $10,000 CTR threshold — across 5 branches in 8 days fits the statutory definition of structuring. Therefore the conduct described falls within § 5324(a)(3)."},
  "metadata": {"task_type": "auxiliary_statutory", "source": "legalbench"}
}
```

---

## 6. RL Phase

### 6.1 Objective

Sharpen the SFT checkpoint along four axes that SFT alone cannot reliably enforce:

1. **Objectivity** — eliminate accusatory / speculative phrasing in SAR narratives ("clearly laundering", "definitely illegal").
2. **Grounding** — reward narratives that cite the supplied policy/SOP/auxiliary-finding evidence; penalize unsupported claims.
3. **JSON validity** — produce schema-conformant JSON across all five task types.
4. **Verifiable correctness** — for `auxiliary_*` task types, reward outputs that match held-out gold answers from FinQA / FinanceBench / LegalBench.

Two RL record kinds are used: **preference pairs** (chosen vs. rejected output for the same input) for axes 1–3, and **verifiable-reward records** (input + ground-truth answer + reward function) for axis 4.

The same record schema as SFT is reused — `metadata.phase ∈ {rl_preference, rl_verifiable}` discriminates. Same toolset-backed `input.*` shapes as SFT, so no distribution shift between phases.

### 6.2 Sample Records

#### Preference pair (`metadata.phase = "rl_preference"`)

```json
{
  "task_type": "sar_judgment",
  "instruction": "Review the following activity bundle and produce a SAR judgment as JSON.",
  "input": {
    "transactions": [{"date": "2026-04-01", "amount": 9500.00, "currency": "USD", "counterparty": "Cash deposit, Branch #14", "channel": "cash", "notes": "Sub-CTR-threshold deposit"}],
    "kyc_profile": {"entity_id": "CUST_8814729", "entity_type": "business", "expected_monthly_volume": 50000, "business_purpose": "Retail jewelry store", "risk_rating": "medium", "incorporation_jurisdiction": "US-NY"},
    "sanctions_pep_hits": [],
    "policy_excerpts": [{"source": "FinCEN", "section": "Advisory FIN-2014-A005", "url": "https://www.fincen.gov/...", "text": "Multiple cash deposits structured below the $10,000 ... may constitute structuring under 31 U.S.C. § 5324."}],
    "sop_excerpts": [{"sop_id": "SOP-STRUCTURING-01", "section": "Step 3", "text": "When sub-$10,000 cash deposit pattern is identified ... escalate to SAR filing."}]
  },
  "output": {
    "is_suspicious": true,
    "suspicious_activity_report": "Suspicious activity is identified for entity CUST_8814729. The pattern of multiple sub-$10,000 cash deposits appears designed to evade reporting requirements, consistent with FinCEN Advisory FIN-2014-A005. Investigator review is recommended for final determination."
  },
  "rejected_output": {
    "is_suspicious": true,
    "suspicious_activity_report": "This customer is clearly laundering money. They are obviously evading the bank's reporting and should be reported immediately. The deposits are definitely illegal."
  },
  "preference_axis": "objectivity",
  "metadata": {"task_type": "sar_judgment", "phase": "rl_preference", "source": "manual_template"}
}
```

#### Verifiable record (`metadata.phase = "rl_verifiable"`)

```json
{
  "task_type": "auxiliary_numeric",
  "instruction": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
  "input": {"passage": "[3M 2018 10-K cash flow excerpt] ... Purchases of PP&E ... (1,577) ...", "context_type": "mixed"},
  "output": {"answer": "$1,577 million ..."},
  "ground_truth": "1577",
  "reward_fn": "finqa_numeric",
  "metadata": {"task_type": "auxiliary_numeric", "phase": "rl_verifiable", "source": "finqa"}
}
```

---

## 7. Data Constraints (SFT data + Agentic Application)

This section is the contract between training-time data construction and the production agentic application. Every constraint here ties an SFT record shape to a runtime tool the agent will call. **A record that violates any of these constraints will produce a model that does not work in production.**

### 7.1 Canonical Record Schema

A **single record schema** covers SFT and RL. `input` and `output` are **discriminated unions keyed on `task_type`**.

```json
{
  "task_type": "sar_judgment | auxiliary_behavioral | auxiliary_numeric | auxiliary_citation | auxiliary_statutory",
  "instruction": "<system / user instruction text>",
  "input":  { "...": "shape varies by task_type — see below" },
  "output": { "...": "shape varies by task_type — see below" },

  "rejected_output": { "...": "RL preference only — same shape as output" },
  "preference_axis": "objectivity | grounding | hedging | numeric | legal | json_format",
  "ground_truth":   "...",
  "reward_fn":      "legalbench_rule_qa | financebench_citation | finqa_numeric",

  "metadata": {
    "phase": "sft | rl_preference | rl_verifiable",
    "source": "<dataset id>",
    "typology": "structuring | smurfing | layering | trade_based_ml | shell_company | human_trafficking | terrorist_financing | elder_exploitation | none",
    "sar_variant": "augmented | bare | adversarial_aux | not_applicable"
  }
}
```

#### Input Shape A — `task_type = "sar_judgment"` (this IS the runtime contract)

```json
{
  "transactions": [{"date": "...", "amount": 0.0, "currency": "...", "counterparty": "...", "channel": "wire | ach | cash | card | cheque | crypto", "notes": "..."}],
  "kyc_profile": {"entity_id": "...", "entity_type": "individual | business", "expected_monthly_volume": 0, "business_purpose": "...", "risk_rating": "low | medium | high | enhanced | prohibited", "incorporation_jurisdiction": "..."},
  "sanctions_pep_hits": [{"name": "...", "list": "OFAC | EU | UN | OpenSanctions", "match_score": 0.0}],
  "policy_excerpts": [{"source": "FFIEC | FATF | FinCEN", "section": "...", "url": "...", "text": "..."}],
  "sop_excerpts": [{"sop_id": "...", "section": "...", "text": "..."}],

  "auxiliary_findings": {
    "behavioral": [{"summary": "...", "metrics": {"tx_count": 0, "tx_total_usd": 0.0, "channel_mix": {}, "velocity_24h_max": 0, "velocity_24h_avg_30d": 0.0, "unique_counterparties_7d": 0, "amount_z_score_max": 0.0, "country_risk_max": 0.0, "loop_detected": false, "vs_declared_volume_ratio": 0.0}, "evidence": "..."}],
    "numeric":    [{"question": "...", "answer": "...", "calculation": "...", "evidence": "..."}],
    "citation":   [{"question": "...", "answer": "...", "evidence_span": "..."}],
    "statutory":  [{"question": "...", "label": "entailment | contradiction | neutral", "reasoning": "..."}]
  }
}
```

`auxiliary_findings` and each of its four sub-arrays are **independently optional** — the runtime case is determined by which were populated by NAT before the final SAR call.

#### Input Shape B — `task_type = "auxiliary_behavioral"`

```json
{"passage": "<transactions[] + kyc_profile rendered as a structured passage; one transaction per line, KYC fields keyed>", "context_type": "transactional"}
```

The passage MUST include enough structure for the model to compute aggregations: per-transaction `(date, amount, currency, counterparty, channel)` and KYC fields `(entity_id, entity_type, expected_monthly_volume, business_purpose, risk_rating, incorporation_jurisdiction)`. The same transactions[] / kyc_profile that appear in a downstream `sar_judgment` bundle are rendered into this passage at training and runtime.

#### Input Shape C — `task_type ∈ {auxiliary_numeric, auxiliary_citation, auxiliary_statutory}`

```json
{"passage": "<source-material text; may include tables, statute text, multi-paragraph context>", "context_type": "text | table | mixed | statute"}
```

#### Output Shapes

| `task_type` | `output` schema | Required content |
|---|---|---|
| `sar_judgment` | `{"is_suspicious": bool, "suspicious_activity_report": str}` | Positive: narrative ≥ 200 chars with entity / date / amount / typology; cites any provided `auxiliary_findings` verbatim. Negative: narrative `""` or `null`. |
| `auxiliary_behavioral` | `{"answer": {"summary": str, "metrics": object, "evidence": str}}` | `summary`: prose interpretation citing metric values verbatim. `metrics`: structured aggregations (tx_count, channel_mix, velocity_24h_max, velocity_24h_avg_30d, unique_counterparties_7d, amount_z_score_max, country_risk_max, loop_detected, vs_declared_volume_ratio, tx_total_usd). `evidence`: reference back to specific transaction indices and KYC fields. |
| `auxiliary_numeric` | `{"answer": str}` | `<final_number>\n\nCalculation:\n<step-by-step trace>\n\nEvidence: <span>` |
| `auxiliary_citation` | `{"answer": str}` | `<answer>\n\nEvidence: <verbatim span from input.passage>` |
| `auxiliary_statutory` | `{"answer": str}` | `<label>\n\nReasoning: <justification grounded in statute and facts>` |

### 7.2 Each `input.*` Sub-Field Maps to a Production Tool

The whole point of the schema is that the SFT record's `input` looks identical to what the runtime agent will assemble from its tool calls. The mapping is exact:

| `input` sub-field | Backing production tool | What the tool returns |
|---|---|---|
| `transactions[]` | **Tool 1 — Transaction DB** (Postgres) | Money-movement events for the entity over the investigation window. |
| `kyc_profile` | **Tool 2 — KYC / CRM Store** (Postgres) | Customer profile: entity type, declared business purpose, expected monthly volume, risk rating, jurisdiction. |
| `sanctions_pep_hits[]` | **Tool 3 — Sanctions / PEP API** (mock REST over OpenSanctions snapshot) | Fuzzy-match results against OFAC + EU + UN + OpenSanctions + PEP registers. Empty for ~95% of records. |
| `policy_excerpts[]` | **Tool 4 — Policy RAG** (vector store over FFIEC + FATF + FinCEN + OFAC corpus) | Top-k retrieved regulatory passages with source / section / URL. |
| `sop_excerpts[]` | **Tool 5 — SOP Repository** (in-cluster service over hand-authored SOPs) | Institution-internal investigation playbook excerpts matching the typology. |
| `auxiliary_findings` | **Custom Task NIM auxiliary calls** (the same model, invoked with `task_type ∈ {auxiliary_behavioral, auxiliary_numeric, auxiliary_citation, auxiliary_statutory}`) | Pre-computed behavioral / numeric / citation / statutory findings the orchestrator chose to compute before the final SAR call. The behavioral call takes the same `transactions[] + kyc_profile` rendered as a passage; the others take their respective source passages. |

**Implication**: any toolset built for SFT data construction is the same toolset that backs the runtime agent. Same generators, same schemas, same sample distributions.

### 7.3 Two-Stage Toolset (Same Generators, Different Volumes)

Every backing data source for the five tools has two stages:

| Stage | When | Volume | Purpose |
|---|---|---|---|
| **Stage 1 — Training-scale** | Built before SFT data construction | Just enough to populate ~60–75K SFT records + ~16K RL records. | Backs SFT/RL data construction. |
| **Stage 2 — Production-scale** | Built before NAT deployment | ~10× Stage 1 volume. **Same generators, same schemas, same sample shape.** | Backs the runtime agentic application. |

| Tool | Stage 1 source | Stage 2 source |
|---|---|---|
| Transaction DB | AMLGentex `tx_log.parquet` (678K rows) + IBM AML `Trans.csv` + Enterprise FC `transactions.csv` — clustered by `entity × pattern × case` and indexed by entity / typology / country. | AMLGentex re-run at `n=100K accounts` (~50M txns) + synthetic typology-targeted top-up for under-represented typologies. |
| KYC / CRM Store | Enterprise FC `entities_master.csv` + AMLGentex `accounts.csv` + IBM AML `accounts.csv`, with missing canonical fields (`business_purpose`, `risk_rating`, `expected_monthly_volume`, `incorporation_jurisdiction`) inferred from native fields. | Same generator at ~100K+ entities across more jurisdictions and entity types. |
| Sanctions / PEP API | One-time public download — OpenSanctions consolidated (CC BY 4.0) + OpenSanctions PEPs (~200–500 MB combined). Sampled per entity's country-risk score and counterparty country. | Refresh OpenSanctions snapshot to latest (publishes ~daily). Same shape. |
| Policy RAG | Already on disk from §2.2: FFIEC HTML (164 pages), FinCEN PDFs, FATF PDFs, OFAC Guidance. Retrieved by typology keyword filter for SFT records. | Same corpus, vector-indexed in Milvus / pgvector. No expansion needed. |
| SOP Repository | **Hand-authored** ~40 pages (8 typology SOPs × ~5 pages each). Anchored on Wolfsberg Group standards + FFIEC examiner procedures + ACAMS practitioner guidance. ~1 person-day effort. | Optional expansion to ~80–120 pages (sub-types: retail / commercial / broker-dealer; jurisdiction-specific procedures). |

**Zero distribution shift between training-time inputs and runtime inputs** is the contract. If Stage 2 changes a field, schema, or sampling rule, Stage 1 must change in lockstep and the SFT data must be regenerated.

### 7.4 `auxiliary_findings` Construction for `sar_judgment` Records

For each `sar_judgment` SFT record, the variant assignment determines how `auxiliary_findings` is populated:

| Variant | Construction rule |
|---|---|
| **Augmented** (~70%) | For each typology-relevant sub-call type, run the CPT/SFT-checkpoint as the corresponding `auxiliary_*` task against the record's raw inputs (`transactions`, `kyc_profile`, `policy_excerpts`, `sop_excerpts`) and inline the parsed output into `input.auxiliary_findings.{behavioral,numeric,citation,statutory}`. **Triggers**: `behavioral` when `len(transactions) ≥ 1` (i.e. always when there is any activity); `numeric` when `len(transactions) ≥ 3` or typology is volume-driven (structuring / smurfing / trade_based_ml); `citation` when `len(policy_excerpts) ≥ 1`; `statutory` when typology maps to a known statute. The behavioral call is the most universally applicable, since any non-empty bundle yields a meaningful summary. |
| **Bare** (~25%) | Leave `auxiliary_findings` absent. No additional construction. The gold SAR narrative must be derivable from raw inputs alone. |
| **Adversarial-aux** (~5%) | Start from an augmented record, then mutate one finding via a rule-based transform: flip a numeric total by ±20%, swap a citation to an unrelated section, invert a statutory label, **or corrupt one behavioral metric (e.g., flip `loop_detected`, swap `velocity_24h_max` with a wrong value, or alter `vs_declared_volume_ratio`)**. Regenerate the gold narrative against the **correct** raw inputs so it detects the inconsistency rather than parroting the wrong finding. The behavioral mutation is a uniquely strong adversarial signal because the model can recompute the metric from the same `transactions[]` already in the bundle. |

**Inlining shape constraint** — the JSON shape of each `auxiliary_findings.{behavioral|numeric|citation|statutory}` entry must match the parsed-output shape produced by the corresponding `auxiliary_*` task at runtime. If the auxiliary task output format changes, the inlining shape must change in lockstep, otherwise the model will see different shapes during training and inference.

**Behavioral metrics — gold-anchored construction.** Unlike numeric/citation/statutory findings, where the LLM produces both the answer and the reasoning, behavioral metrics are computed **deterministically at training time** by a Pandas-based feature computer over the source transactions and KYC. The LLM only writes the prose `summary` portion of `auxiliary_behavioral`'s output, with the gold `metrics` block injected as ground truth. This bounds math errors at construction time: the model learns to reproduce gold metrics exactly (RL phase enforces this via verifiable reward, §6.1 axis 4), not to compute them from scratch. At runtime the model produces both `summary` and `metrics`; correctness of the metrics block becomes a deployable quality metric.

**Gold SAR narrative source policy** (per §5.2): augmented- and bare-variant `sar_judgment` records use SARSum prose notes as the primary narrative source, paired with the transactional context as a soft template the LLM grounds in the bundle's specific facts. EFC `sar_reports.narrative` fields are NOT used as gold narratives — they are templated stubs. EFC remains valuable for `transactions[] + kyc_profile + behavioral feature columns`.

### 7.5 Runtime Composition (NAT)

The orchestrator (NAT) assembles a `sar_judgment` request at runtime in a fixed sequence:

1. **Gather raw inputs** — call Tool 1 → `transactions`; Tool 2 → `kyc_profile`; Tool 3 → `sanctions_pep_hits`; Tool 4 → `policy_excerpts`; Tool 5 → `sop_excerpts`.
2. **Decide auxiliary calls** — based on the gathered raw inputs, decide which of the four `auxiliary_*` task types to invoke (0–4 calls). Heuristic: `behavioral` whenever `len(transactions) ≥ 1` (essentially always); `numeric` when `len(transactions) ≥ 3` or volume-driven typology; `citation` when `len(policy_excerpts) ≥ 1`; `statutory` when typology maps to a known statute.
3. **Invoke auxiliary calls** — issue each chosen `auxiliary_*` request to the Custom Task NIM. For `auxiliary_behavioral`, render the gathered `transactions[] + kyc_profile` into the structured passage format (§7.1 Input Shape B) and pass that as the `input.passage`. Parse each output into the matching `auxiliary_findings.{behavioral|numeric|citation|statutory}` entry shape from §7.1.
4. **Final SAR call** — issue the `sar_judgment` request to the Custom Task NIM with the assembled `input` (raw inputs + populated `auxiliary_findings`).

The model encounters identical-shaped inputs at training time (§7.4) and runtime (this section), with the only difference being entity / counterparty / amount distribution (Stage 2 is broader than Stage 1).

**Why `auxiliary_behavioral` is invoked, not pre-attached**: behavioral metrics could be attached to Tool 1's response by the data layer (a feature-store-backed "transaction query returns enriched rows" pattern). We deliberately reject that: making behavioral analysis a model task means the SAR-time call has the option to **recompute / verify** the behavioral findings against the raw transactions in the same bundle, which is the core mechanism behind the adversarial-aux variant (§7.4). This self-verification path does not exist if features are silently delivered by an upstream tool. The reasoning model can still call `auxiliary_behavioral` once, cache the result, and feed it forward — but the data flow is symmetric with the other three aux tasks, and the model is the single source of truth for interpretive analysis.

---

## 8. Appendix — Construction Details and Quality Gates

These items are not part of the main flow but are non-trivial to reconstruct later if omitted. Each is a constraint or check on top of Sections 4–7.

### A. Positive vs. Negative Construction for `sar_judgment`

§5.2 specifies the `sar_judgment` corpus is ~42% positive + ~25% negative. The positive / negative split is **orthogonal** to the augmented / bare / adversarial-aux variant split in §5.2 — every record carries both a label (`is_suspicious`) and a variant. Negative records come from two construction paths:

| Path | Source | Construction rule |
|---|---|---|
| **Real-label negatives** (~60% of negatives) | SARSum "Not concerning" tier; IBM AML rows with `Is Laundering = 0`; AMLGentex rows with `isSAR = 0` | Use the source's transactions and KYC as-is. `output.is_suspicious=false`, `narrative=""` or `null`. |
| **Synthesized negatives** (~40% of negatives) | CFPB Consumer Complaints | Extract entity, counterparty, and transaction summary from the complaint narrative; assemble `transactions[]` and `kyc_profile`. Output is always `is_suspicious=false`, `narrative=""`. |

**Why the split matters**: real-label negatives often look "easy" (no transactions, no hits, trivially benign). Without a meaningful share of *near-miss negatives* — records with sub-threshold cash deposits, foreign wires, or sanctions-adjacent counterparties that nonetheless have legitimate business explanations — the model degenerates into "any non-trivial activity ⇒ suspicious." CFPB-derived negatives provide the near-miss class because complaints often describe activity patterns that look superficially suspicious but have benign root causes (billing disputes, identity errors, merchant chargebacks).

**Gold-narrative rule for negatives**: `output.suspicious_activity_report` MUST be empty (`""` or `null`). Any non-empty narrative on a negative record contaminates the model into producing SARs even when `is_suspicious=false`.

### B. Per-Source SFT Conversion Rules (Tier 1)

Each Tier 1 `sar_judgment` source has a different native shape. The bridge between native fields and the canonical schema (§7.1) is fixed per source:

| Source | Native shape | Canonical mapping |
|---|---|---|
| **Enterprise Financial Crime AI** | Relational: `alerts.csv`, `investigations.csv`, `sar_reports.csv`, `transactions.csv`, `entities_master.csv` joined on `case_id` | `transactions.csv` → `input.transactions[]` (filtered by `case_id`); `entities_master.csv` → `input.kyc_profile`; `sar_reports.narrative` → `output.suspicious_activity_report` (use as-is when `≥ 200 chars`); `investigation_cases.scenario` → `metadata.typology`. |
| **SARSum** | Prose notes (7 per SAR set) × 6 quality tiers + Suspicious / Not concerning label | Notes prose → parse `${amount}`, `{date}`, `{counterparty}`, `{channel}` slots → `input.transactions[]`; level-6 (highest quality) `summary` → `output.suspicious_activity_report`; label → `output.is_suspicious`. Levels 0–4 reserved for RL preference pairs. |
| **IBM AML HI-Small** | `Trans.csv` + `Patterns.txt` (BEGIN/END pattern blocks) + `Is Laundering` flag | Cluster `Trans.csv` rows by entity × pattern → `input.transactions[]`; map `Patterns.txt` block label → `metadata.typology` (`FAN-OUT → structuring`, `CYCLE → layering`, `STACK → layering`, etc.); `Is Laundering` → `output.is_suspicious`; **gold narrative does not exist natively — must be constructed per record**. |
| **AMLGentex** | `tx_log.parquet` + `accounts.csv` + `alert_models.csv` joined on `patternID` | Cluster `tx_log` rows by `patternID` → `input.transactions[]`; `accounts.csv` → `input.kyc_profile`; `alert_models.type` → `metadata.typology` (`fan_out → structuring`, `cycle → layering`, `peeling_chain → layering`, `round_number → smurfing`, etc.); `isSAR` → `output.is_suspicious`; **gold narrative does not exist natively — must be constructed per record**. |

**Implication**: Enterprise FC and SARSum carry real gold narratives — use them. IBM AML and AMLGentex carry only labels — gold narratives must be synthesized at construction time from the (transactions, kyc_profile, typology) tuple.

Tier 2 (`auxiliary_*`) sources are mostly direct conversions — the source `passage` field maps to `input.passage`, the source `question` to `instruction`, the source `answer` to `output.answer` — so no per-source table is needed.

### C. Per-Typology Coverage Floor

The 8 canonical typologies are **unevenly represented** in the native data:

| Typology | Native availability | Construction strategy |
|---|---|---|
| `structuring` | High (IBM AML FAN-OUT, AMLGentex fan_out, Enterprise FC structuring scenarios) | Use native; cap if over-represented. |
| `layering` | High (IBM AML CYCLE/STACK, AMLGentex cycle/peeling_chain) | Use native. |
| `smurfing` | Medium | Use native + light synthetic top-up. |
| `trade_based_ml` | Low | Synthetic top-up required. |
| `shell_company` | Low | Synthetic top-up required. |
| `human_trafficking` | Near-zero | Synthetic top-up required (anchor on FATF / FinCEN typology guidance from §2.2). |
| `terrorist_financing` | Near-zero | Synthetic top-up required (anchor on OFAC + UN guidance). |
| `elder_exploitation` | Near-zero | Synthetic top-up required (anchor on FinCEN Advisory FIN-2022-A002 and similar). |

**Floor**: every typology must have **≥ 1.5K `sar_judgment` records** (positive + negative combined) before SFT. Below this floor, per-typology classification F1 collapses while overall F1 stays misleadingly high. Construction order: real-label first, synthetic top-up second to fill the under-represented buckets. A `phase_stats.json` is emitted at construction time with per-typology counts; any typology below floor blocks the SFT run.

### D. Stage 1 ↔ Stage 2 Schema Parity Check

§7.3 states the contract: Stage 1 (training-scale) and Stage 2 (production-scale) toolsets must share generators, schemas, and sample distributions. To enforce rather than aspire to this:

| Check | Rule | When |
|---|---|---|
| **Schema diff** | JSON-schema diff between a sample of Stage 1 outputs and Stage 2 outputs, per tool. Fields, types, and enum values must match exactly. | Before NAT deployment. Mismatch blocks rollout. |
| **Distribution drift** | Per-tool summary statistics (e.g., for `transactions`: avg / p95 amount; channel mix; counterparty country distribution) compared between Stage 1 and Stage 2 samples. | Before NAT deployment. Drift beyond ±20% on any key statistic surfaces a warning for review (does not necessarily block, but requires sign-off). |
| **Stage 2 → SFT replay** | Sample N=500 Stage 2 records, route them through the SFT data pipeline, and verify they produce valid records with no schema errors. | Before NAT deployment. Any failure blocks rollout. |
| **`auxiliary_findings` shape parity** | The JSON shape of each `auxiliary_findings.{behavioral|numeric|citation|statutory}` entry produced by parsing Custom Task NIM output at runtime must match the entry shape used at SFT inlining time (§7.4). | Continuous — checked on every NAT deployment via a smoke test invoking each `auxiliary_*` task and validating the parsed output schema. |

**Why this matters**: even subtle drift (a new `currency` enum value in Stage 2, a renamed `match_score` field, a wrapping object around `auxiliary_findings`) silently degrades model behavior at runtime because the model never saw the new shape during training. The four checks above turn a "we'll be careful" promise into a deployment-blocking gate.

### E. Narrative Library (SARSum-as-Primary)

Gold SAR narratives for `sar_judgment` records are produced by **pairing** a transactional context (from EFC raw / IBM / AMLGentex / CFPB) with a narrative reasoning pattern from a curated library. The library has two tiers:

| Tier | Source | Contents | Indexed by |
|---|---|---|---|
| **Primary** | SARSum (~2,000 cases × 7 prose notes per case = ~14K notes) | Per-pattern `Suspicious / Not concerning / Not suspicious` decisions with regulator-grade reasoning prose. Each case also carries a `key_facts` block with 5 paraphrase variants per fact. | `(typology, decision, severity_band)` |
| **Secondary** | Hand-authored exemplars (~20–40) anchored on FinCEN advisories and FATF typology reports | Coverage for typologies where SARSum is thin: `terrorist_financing`, `elder_exploitation`, `human_trafficking`. | `(typology)` |

**Pair-and-ground generation rule** — at SFT construction time, for each transactional context bundle assigned to a `sar_judgment` record:

1. Compute the bundle's `semantic_profile` (Appendix F).
2. Retrieve the top-k matching narrative patterns from the library, scored by `(typology match, decision match, channel-coherence match)`.
3. Pass BOTH the transactional bundle AND the matched patterns to the LLM with the instruction: *"Produce a SAR narrative for THIS bundle using the reasoning style and statutory framing of these patterns. Cite evidence from the bundle only; do not introduce facts from the patterns."*
4. Mechanically validate the resulting narrative against the bundle's `transactions[]`, `kyc_profile`, and `auxiliary_findings`.

**Why pair-and-ground rather than free generation**: free generation gives the LLM no anchor for typology-coherent statutory framing — it pulls from training-data priors and produces inconsistent reasoning (e.g., applying CTR to wires, citing structuring statutes for layering). Library pairing supplies a real human-written reasoning pattern that is by construction coherent with the (typology, decision, channel) profile; the LLM's task narrows from "invent" to "ground a known pattern in given facts", which is a much more reliable LLM job.

**EFC sar_reports.narrative is dropped from the library.** Its 6,001 narratives all conform to a single template ("The institution identified activity requiring further review involving X CURRENCY from C1 to C2…") and carry no regulatory or typology-specific reasoning. EFC's value is in transactions + KYC + behavioral feature columns, not narrative gold.

### F. Semantic Profile (Internal Cross-Stage Contract)

The construction pipeline has multiple stages that retrieve, generate, or validate against the bundle. Today these stages use independent heuristics — Stage 5 retrieves policies by typology keyword, Stage 6 generates findings by typology, Stage 7 generates narrative by typology — and a typology change in one stage does not propagate to the others. This causes incoherence (e.g., a layering bundle still retrieves CTR policies because the layering keyword pulls structuring-adjacent text).

The fix is a **single semantic profile** computed once from the bundle and read by every downstream stage. It is an **internal construction-pipeline contract**, not a runtime field — it does not appear in the SFT record schema (§7.1).

```json
{
  "channel_mix": {"cash": 0.0, "wire": 1.0, "ach": 0.0, "card": 0.0, ...},
  "cash_present": false,
  "regulatory_frame": "layering_passthrough | ctr_structuring | tbml | shell | te | sanctions | elder | trafficking | benign",
  "declared_volume_band": "under | match | over",
  "geo_risk": "low | medium | high",
  "typology_inferred": "<one of the 8 canonical typologies, after channel-coherence remap>"
}
```

**Computation point** — Stage 1 (driver sampling + source extraction). Every downstream stage reads the profile rather than the raw typology field:

- Stage 4 (Sanctions/PEP firing): keyed on `regulatory_frame`.
- Stage 5 (Policy retrieval): filtered by `regulatory_frame` rather than typology keywords.
- Stage 6 (`auxiliary_findings` generation): refuses CTR-statute citations when `regulatory_frame != ctr_structuring`; refuses sanction-list reasoning when `geo_risk == low` and no PEP hits.
- Stage 7 (SAR narrative): library retrieval (Appendix E) keyed on `(typology_inferred, decision, channel_mix)`; prompt receives `regulatory_frame` as the primary framing instruction.
- Stage 8 (Adversarial mutations): mutations are scoped to the active `regulatory_frame` so the inconsistency is detectable.

**Why this is appendix and not §7.1**: the SFT record never carries the semantic profile. It is consumed by construction, then discarded. The model learns from the resulting bundle + narrative + findings; the profile is the construction-time mechanism that keeps those three coherent.

### G. Corpus-Level Blocking Gates (Distribution Enforcement)

Per-record validation (mechanical rules in §7 + §B) is necessary but insufficient: a corpus that passes every per-record check can still violate the strategy's distribution targets. The construction pipeline emits a `phase_stats.json` at completion and **blocks** if any of the following gates fail:

| Gate | Threshold | Source rule |
|---|---|---|
| **Per-typology floor** | Every canonical typology ≥ 1.5K records (positive + negative combined), proportional to N for smaller runs (e.g., ≥ 10 per typology for N=500) | Appendix C |
| **Per-typology positive coverage** | Every typology has ≥ 1 positive record | New — closes "0 structuring positives" gap |
| **Pos/neg ratio** | 42:25 ± 5pp | §5.2 |
| **Variant ratio** | 70/25/5 augmented/bare/adversarial ± 5pp | §5.2 |
| **Source contribution cap** | No single source contributes > 30% of records | New — prevents EFC dominance |
| **Aux task balance** | `auxiliary_{behavioral,numeric,citation,statutory}` shares within ±2pp of target (8/8/8/9%) | §5.2 |
| **Schema validity** | 100% of records validate against §7.1 schemas | §7.1 |

A failed gate is **actionable**, not just informational. The pipeline emits a remediation hint (e.g., "structuring-positive count = 0; rerun Stage 1 with `boost_typology=structuring` and `min_positive_count=10`") and exits non-zero. The next run picks up the hint and rebalances.

**Why this matters**: a corpus that ships with zero positive records of an entire typology produces a model that cannot recognize that typology at runtime, while overall accuracy stays misleadingly high. The blocking gate is the only mechanism that turns "we'll fix it next time" into "the run cannot finish until it's fixed".
