# Revised Data Strategy — AML Investigation Agent
## Scope: Model Training Only (CPT / SFT / RL)

This document supersedes the data sections of `training_plan.md` and `dataset_recommendations.md` for the three training phases. It is the single source of truth for what to download, how to extract from PDFs, and how to curate into training-ready corpora.

---

## Guiding Principles

1. **CPT is for language register.** Use CPT only for data that teaches the model to *write like* a financial regulator or AML analyst. Do not CPT on prose-converted structured data — it degrades register.
2. **Capability comes from SFT.** Arithmetic, entity extraction, citation-grounded answering, typology classification — all belong in SFT, not CPT.
3. **RL needs verifiable reward signals.** Preference pairs alone cap quality; ground-truth answers (LegalBench) and ground-truth evidence spans (FinanceBench) give automatic verification.
4. **Every source earns its slot.** Scope creep (EU regs, press corpora, narrow NER benchmarks) is cut even if the data is good — it competes for finite time.

---

## Canonical Folder Structure (applies to every step)

Every step in the pipeline writes its outputs under a `data/` directory that mirrors the same phase-oriented layout. This keeps the pipeline auditable and makes each step trivially restartable.

```text
<step>/
|-- data/
    |-- phase_stats.json
    |-- raw/
    |   |-- cpt/
    |   |   |-- level_1/
    |   |   |   |-- pile_of_law_sec/
    |   |   |   |-- pile_of_law_cfr/
    |   |   |   |-- pile_of_law_federal_register/
    |   |   |   |-- pile_of_law_uscode/
    |   |   |   |-- pile_of_law_oig/
    |   |   |   |-- pile_of_law_doj_guidance/
    |   |   |   +-- edgar_corpus/
    |   |   +-- level_2/
    |   |       |-- fincen_advisories/
    |   |       |-- fincen_federal_register/
    |   |       |-- fincen_sar_reviews/
    |   |       |-- fincen_enforcement/
    |   |       |-- fincen_files/
    |   |       |-- fatf_publications/
    |   |       |-- ofac_enforcement/
    |   |       |-- ofac_guidance/
    |   |       |-- cfr_chapter_x/
    |   |       +-- caselaw_access_project/
    |   |-- sft/
    |   |   |-- enterprise_financial_crime/
    |   |   |-- sarsum/
    |   |   |-- amlgentex/
    |   |   |-- finance_instruct_500k/
    |   |   |-- finqa/
    |   |   |-- tat_qa/
    |   |   |-- financebench/
    |   |   |-- legalbench/
    |   |   |-- ffiec_manual/
    |   |   |-- ibm_aml_transactions/
    |   |   +-- cfpb_complaints/
    |   +-- rl/
    |       |-- sarsum/
    |       |-- legalbench/
    |       |-- financebench/
    |       +-- finqa/
    +-- final/
        |-- cpt/
        |   |-- level_1/
        |   +-- level_2/
        |-- sft/
        +-- rl/
```

- `raw/` — the output of the current step, per source, untouched downstream.
- `final/` — aggregated, step-complete output ready for the next step to consume.
- `phase_stats.json` — per-source size, record counts, filter/extraction/dedup stats for that step.

The same layout is used by all three steps:
- `1.data_download/data/…` — raw files as downloaded
- `2.pdf_extraction/data/…` — text extracted from PDFs
- `3.data_preparation/data/…` — curated, filtered, deduped, training-ready shards

A file downloaded to `1.data_download/data/raw/cpt/level_2/fatf_publications/pdfs/fatf-mer-singapore-2024.pdf` produces its extracted text at `2.pdf_extraction/data/raw/cpt/level_2/fatf_publications/fatf-mer-singapore-2024.txt` and ends up contributing shards to `3.data_preparation/data/final/cpt/level_2/shards/*.jsonl.zst`. Paths stay stable; content transforms.

---

# Step 1 — Data Download

Download raw data as-is. No extraction, no filtering beyond structural subsetting (e.g., selecting HuggingFace configs, keyword pre-filter for massive corpora). This step is I/O-bound.

## 1.1 CPT Layer 1 — Financial Regulatory & Corporate Compliance

Teaches broad financial regulatory legal language, enforcement writing style, statutory frameworks, and corporate compliance disclosures.

| Source | URL | License | What to Download | Structural Filter Applied at Download |
|---|---|---|---|---|
| Pile of Law: `sec` | `huggingface.co/datasets/pile-of-law/pile-of-law` | CC BY-NC-SA 4.0 | HF config `sec` | None — all content is financial enforcement |
| Pile of Law: `cfr` | same | CC BY-NC-SA 4.0 | HF config `cfr` | Title 12 (Banks & Banking), Title 17 (Commodity & Securities Exchanges), Title 31 (Money & Finance) |
| Pile of Law: `federal_register` | same | CC BY-NC-SA 4.0 | HF config `federal_register` | Financial agencies only: FinCEN, Treasury, SEC, CFTC, FDIC, OCC, Federal Reserve |
| Pile of Law: `uscode` | same | CC BY-NC-SA 4.0 | HF config `uscode` | Title 12, Title 18 Ch. 95 (Racketeering) & Ch. 96 (Money Laundering §§ 1956–1957), Title 31 |
| Pile of Law: `oig` | same | CC BY-NC-SA 4.0 | HF config `oig` | Financial agency OIGs only: Treasury OIG, FDIC OIG, OCC OIG, Federal Reserve OIG |
| Pile of Law: `doj_guidance` | same | CC BY-NC-SA 4.0 | HF config `doj_guidance` | None (small; already corporate-enforcement-focused) |
| EDGAR-CORPUS | `huggingface.co/datasets/eloukas/edgar-corpus` | Public (SEC data) | All year configs (1993–2020) | SIC codes 6000–6999 (Finance, Insurance, Real Estate) |

**Output:** `1.data_download/data/raw/cpt/level_1/<source>/…`

## 1.2 CPT Layer 2 — AML-Specific Domain Corpus

Narrow domain adaptation: money laundering typologies, BSA compliance, SAR writing, sanctions screening, FinCEN/FATF guidance, financial-crime jurisprudence.

| Source | URL | License | What to Download | Notes |
|---|---|---|---|---|
| FinCEN Advisories & Guidance | `fincen.gov/resources/statutes-regulations/guidance/` | Public domain (US gov) | All public PDFs | Scrape index pages; save PDFs as-is |
| FinCEN Federal Register Notices | `fincen.gov/resources/statutes-regulations/federal-register-notices` | Public domain | All public PDFs | Same |
| FinCEN SAR Activity Reviews | `fincen.gov/sar-activity-review-trends-tips-issues` | Public domain | All 23 issues (2000–2013) | Direct PDF download |
| FinCEN Enforcement Actions | `catalog.data.gov/dataset/fincen-enforcement-actions-for-violations-of-the-bank-secrecy-act` | CC0 | CSV + linked case docs | Direct download |
| FinCEN Files (ICIJ) | `icij.org/investigations/fincen-files/download-fincen-files-transaction-data/` | ICIJ public release | CSV + narrative PDFs | **Split at later step:** narratives → CPT L2, structured rows → agent tools |
| FATF Publications | `fatf-gafi.org/en/publications/{mutualevaluations,methodsandtrends,fatfrecommendations,fatfgeneral,high-risk-and-other-monitored-jurisdictions}.html` | FATF terms (public) | All linked PDFs | Two-pass strategy: `cloudscraper` → Playwright fallback for JS-rendered pages |
| OFAC Enforcement Actions | `opensanctions.org/datasets/us_ofac_enforcement_actions/` | CC BY 4.0 (OpenSanctions) | Bulk JSON/CSV | Direct download |
| OFAC Guidance & Frameworks | `ofac.treasury.gov/file-finder` | Public domain | Compliance guidance PDFs | Scrape file-finder index |
| 31 CFR Chapter X | Subset of `cfr` HF config | CC BY-NC-SA 4.0 | Derived from Layer 1 `cfr` | Filter at prep step; no separate download |
| **Caselaw Access Project** | `huggingface.co/datasets/free-law/Caselaw_Access_Project` | CC BY-NC-SA | HF dataset | Pre-filter at download: `jurisdiction ∈ {federal district, 2nd Cir, 9th Cir, NY state, CA state, FL state, TX state}` AND keyword match on `{money laundering, Bank Secrecy Act, structuring, wire fraud, RICO, 5324, 1956, 1957, willful blindness}` |

**Output:** `1.data_download/data/raw/cpt/level_2/<source>/…`

## 1.3 SFT — Instruction-Response Sources

Covers five capabilities: (A) SAR narrative drafting, (B) numerical reasoning, (C) citation-grounded answering, (D) typology classification + entity extraction, (E) compliance Q&A / statutory reasoning.

| Source | URL | License | Capability | What to Download |
|---|---|---|---|---|
| Enterprise Financial Crime AI Dataset | `huggingface.co/datasets/Webopen2026/enterprise-financial-crime-ai-dataset` | Research | A | Full dataset (310K records) |
| SARSum | `kaggle.com/datasets/leonardovalves/sarsum` | Research (Feedzai) | A | All 2K SAR sets × 6 quality variants |
| AMLGentex | `github.com/aidotse/AMLGentex` | Open | A, D | Generator repo (run inline for synthetic transactions) |
| **Finance-Instruct-500k** | `huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k` | Apache 2.0 | E | Full dataset |
| **FinQA** | `huggingface.co/datasets/dreamerdeo/finqa` | MIT | B | Full dataset |
| **TAT-QA** | `huggingface.co/datasets/next-tat/TAT-QA` | MIT | B | Full dataset |
| **FinanceBench** | `huggingface.co/datasets/PatronusAI/financebench` | MIT | C | Full dataset |
| **LegalBench** | `huggingface.co/datasets/nguha/legalbench` | MIT | E | 6 logical families mapped to real HF configs: **rule_qa** (`rule_qa`) · **statutory reasoning (SARA)** (`sara_entailment`, `sara_numeric`) · **hearsay** (`hearsay`) · **consumer contracts QA** (`consumer_contracts_qa`) · **contract_nli** (representative subset: `contract_nli_confidentiality_of_agreement`, `contract_nli_explicit_identification`, `contract_nli_limited_use`, `contract_nli_no_licensing`, `contract_nli_notice_on_compelled_disclosure`, `contract_nli_return_of_confidential_information`) · **supply_chain_disclosure** (representative subset: `supply_chain_disclosure_disclosed_accountability`, `supply_chain_disclosure_disclosed_training`, `supply_chain_disclosure_disclosed_verification`, `supply_chain_disclosure_best_practice_accountability`, `supply_chain_disclosure_best_practice_training`) |
| **FFIEC BSA/AML Examination Manual** | `bsaaml.ffiec.gov/manual/Introduction/01` | Public domain | E | Scrape every section page as HTML |
| **IBM AML Transactions (HI-Small)** | `kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | CDLA-Sharing | D | HI-Small variant only (for workshop scope) |
| **CFPB Consumer Complaint Database** | `catalog.data.gov/dataset/consumer-complaint-database` | Public domain | D | CSV with narratives; pre-filter: `has_consumer_narrative = True` AND `product ∈ {"Checking or savings account", "Money transfers, virtual currency, virtual currency", "Credit card"}` |

**Output:** `1.data_download/data/raw/sft/<source>/…`

## 1.4 RL — Preference / Reward Signal Sources

| Source | URL | License | Role | What to Download |
|---|---|---|---|---|
| SARSum (reused from SFT download) | — | — | Graded preference pairs | No separate download |
| **LegalBench (reused from SFT download)** | — | — | Verifiable online reward during GRPO | Hold out 30% of sub-task examples from SFT; use as RL reward set |
| **FinanceBench (reused from SFT download)** | — | — | Citation-grounding verifiable reward | Hold out 30% |
| **FinQA (reused from SFT download)** | — | — | Numeric verifiable reward | Hold out 30% |

**Output:** `1.data_download/data/raw/rl/<source>/…` — symlinks or manifests pointing to SFT download locations; the held-out split is materialized at Step 3 (data preparation).

## 1.5 Step 1 Deliverables

- All raw files under `1.data_download/data/raw/<phase>/<source>/…`
- `1.data_download/data/phase_stats.json` — per-source bytes downloaded, file count, structural filter pass-rate
- `1.data_download/data_download_output.md` — human-readable summary
- No text extraction, no language filtering, no deduplication. Those are Steps 2 and 3.

---

# Step 2 — PDF Extraction (NVIDIA RAG Blueprint / NV-Ingest)

Convert every PDF downloaded in Step 1 into clean, structured text using the **NVIDIA RAG Blueprint** — specifically the **NVIDIA Ingest (NV-Ingest)** multimodal document ingestion microservice. This replaces ad-hoc Python PDF libraries with a GPU-accelerated, model-based extraction pipeline that handles text, tables, charts, and scanned pages in a single pass.

PDFs live in these sources: FinCEN Advisories, FinCEN Federal Register Notices, FinCEN SAR Reviews, FinCEN Enforcement Actions (linked case docs), FinCEN Files narratives, FATF Publications, OFAC Guidance.

## 2.1 Deployment

NV-Ingest is deployed as a set of NIM microservices. For this workshop, use the reference deployment from the NVIDIA RAG Blueprint:

- **NV-Ingest service** — orchestrates ingestion jobs and result collection
- **`nemoretriever-parse` NIM** — primary text extraction with layout awareness
- **`nemoretriever-page-elements` NIM** — page-level element detection (text blocks, tables, figures, charts)
- **`nemoretriever-table-structure` NIM** — table structure recognition; outputs structured HTML / markdown
- **`nemoretriever-graphic-elements` NIM** — chart and figure caption extraction
- **PaddleOCR NIM** — OCR fallback for scanned pages, image-only PDFs, and low-text-yield pages

Deployment targets the same 8×H100 node used for training (the services are gated behind a single ingestion endpoint). Run the Blueprint's `docker compose` profile during Step 2 execution, then tear it down to free GPU memory for training.

## 2.2 Extraction Pipeline

For each PDF under `1.data_download/data/raw/**/*.pdf`, submit a job to the NV-Ingest endpoint with the following task specification:

| Task | Enabled | Purpose |
|---|---|---|
| `extract` (text) | Yes | Primary text extraction via `nemoretriever-parse` |
| `extract` (tables) | Yes | Structured table extraction via `nemoretriever-table-structure` |
| `extract` (charts) | Yes | Chart caption/text extraction via `nemoretriever-graphic-elements` |
| `extract` (infographics) | Yes | Figure/infographic text recovery |
| `extract` (images via OCR) | Auto (triggered when a page yields < 500 chars of native text) | Scanned or image-heavy pages |
| `split` | No | Deferred to Step 3 (we chunk at curation time, not ingestion time) |
| `embed` | No | Not needed for training data (RAG blueprint use case only) |
| `store` | No | Blueprint-level VDB storage is not used; we persist raw extractions to disk |

### 2.2.1 Output Schema

NV-Ingest returns one record per extracted element (text block, table, chart, etc.) with source-page metadata. For Step 2 we collapse these into one normalized `.jsonl` per source PDF:

```json
{"doc_id": "<source_pdf_basename>", "page": 12, "element_type": "text|table|chart|ocr_text", "content": "...", "bbox": [x0,y0,x1,y1], "confidence": 0.97}
```

Plus a per-document aggregated `.txt` — page-ordered concatenation of all elements (tables rendered as markdown inline at their original document position) — for consumption by the Step 3 CPT preparation pipeline.

## 2.3 Per-Source Configuration

The Blueprint accepts per-source tuning via the ingestion job spec:

| Source | Notes |
|---|---|
| FinCEN Advisories | Default task set; high text-yield expected |
| FinCEN SAR Reviews | Older issues (2000–2004) are scanned — OCR auto-triggers per page |
| FinCEN Enforcement linked docs | Mixed formats; default task set |
| FinCEN Files narratives | Enable all tasks (tables are common in narrative attachments) |
| FATF Mutual Evaluations | 200–300 page docs; rely on `nemoretriever-parse` layout awareness for section hierarchy preservation |
| FATF Recommendations | Heavily formatted; table extraction critical |
| FATF Typology Reports | Charts and infographics common — keep graphic-elements extraction on |
| OFAC Guidance | Short docs; default task set |

## 2.4 HTML Sources

The FFIEC BSA/AML Examination Manual is the one non-PDF source handled in this step. NV-Ingest accepts HTML input through the same ingestion API. Submit scraped section HTML files with the same task set (minus OCR, which won't trigger). Output schema is identical.

## 2.5 Discard & Quality Gates

- Discard a document when **all** pages return `element_type="ocr_text"` with `confidence < 0.5` AND total `len(content) < 500` — indicates a corrupted or unreadable PDF, not a genuine scan.
- Flag (but do not discard) documents where > 50% of pages triggered OCR — these are scans that downstream quality filtering in Step 3 will scrutinize more aggressively.

## 2.6 Step 2 Deliverables

- One `.jsonl` and one aggregated `.txt` per input PDF under `2.pdf_extraction/data/raw/<phase>/<source>/…`
- `2.pdf_extraction/data/final/<phase>/<source>/manifest.jsonl` with per-file metadata: `{source_pdf, extracted_jsonl, aggregated_txt, page_count, element_counts_by_type, ocr_page_count, mean_confidence, discarded}`
- `2.pdf_extraction/data/phase_stats.json` — per-source aggregates: PDFs in, docs out, element-type distribution, OCR rate, discard rate
- Ingestion-job logs archived under `2.pdf_extraction/logs/` for reproducibility

---

# Step 3 — Data Preparation (Curation, Filtering, Cleaning, Deduplication)

Transform the raw text (from Step 1 HF datasets and Step 2 extracted PDFs) into training-ready corpora. This step is where every per-phase choice is enforced.

## 3.1 Common Pipeline (applied everywhere)

Implemented using `nemo-curator` primitives where available.

1. **Language filter** — `fastText lid.176`, keep `en` only.
2. **Document-length bounds** — reject docs < 200 chars (noise) or > 1M chars (splitting artifacts).
3. **Boilerplate removal** — strip standard headers/footers using recurring-line detection and a per-source denylist of known boilerplate (e.g., "FOR OFFICIAL USE ONLY", DOJ template headers).
4. **Quality filter** — heuristic filters: alphanumeric ratio ≥ 0.7, repeated-line ratio ≤ 0.3, mean line length ∈ [20, 2000], common-word ratio ≥ 0.5.
5. **PII scrub** — `nemo-curator` PII detector + regex for SSN / EIN / routing numbers / account numbers. Replace with typed tags: `[SSN]`, `[EIN]`, `[ACCOUNT_ID]`, `[ROUTING]`, `[PHONE]`, `[EMAIL]`. Required for FinCEN Files, Caselaw, CFPB complaints.
6. **Exact deduplication** — doc-level SHA-256 hash; collapse duplicates across sources (critical: 31 CFR Ch. X will exist in `cfr` and as its own Layer 2 filtered subset).
7. **Fuzzy deduplication** — MinHash (128 perms) + LSH, Jaccard threshold 0.8, within-source and across-source.

## 3.2 CPT Layer 1 Preparation

Input: `1.data_download/data/raw/cpt/level_1/*/` (HF datasets are already text; no Step 2 needed).

Per-source additions to the common pipeline:

| Source | Additional Processing |
|---|---|
| Pile of Law subsets | Apply the structural filter if not pre-filtered at download; strip Pile of Law metadata headers; preserve `url` / `case` identifier in metadata |
| EDGAR-CORPUS | Re-verify SIC 6000–6999 filter; drop the `sections` metadata wrapper and flatten to plain text per filing; preserve CIK + filing-year in metadata |

After common pipeline, across all Layer 1 sources:

1. Cross-source MinHash dedup (EDGAR overlaps with Pile of Law `sec`)
2. Tokenize with base-model tokenizer (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`)
3. Pack to 4096-token sequences
4. Shard into Megatron-Bridge-compatible `.jsonl.zst` (≈ 1 GB per shard)
5. 1% validation holdout (stratified by source; held-out shards tagged `_val`)

**Output:** `3.data_preparation/data/final/cpt/level_1/{shards,val_shards}/`

## 3.3 CPT Layer 2 Preparation

Input: HF-downloaded text (`cfr` → 31 CFR Ch. X filter, Caselaw) + Step 2 extracted PDF text for FinCEN / FATF / OFAC / FinCEN Files narratives.

Per-source additions to the common pipeline:

| Source | Additional Processing |
|---|---|
| FinCEN advisories / FR notices / SAR reviews / enforcement | Post-OCR cleanup for pages flagged `ocr_page_count > 0` in Step 2 manifest (spell-check against a financial-term whitelist to avoid over-correcting domain jargon in scanned 2000–2004 SAR reviews) |
| FinCEN Files | Split: narrative text → Layer 2; structured CSV rows → agent-tool data (out of training scope) |
| FATF Publications | De-duplicate across country MERs (boilerplate methodology sections repeat across 200+ reports — this is the single biggest dedup win in Layer 2) |
| OFAC Enforcement | JSON/CSV → natural-language rendering of each enforcement case (one doc per case) |
| OFAC Guidance | Standard PDF cleanup |
| 31 CFR Chapter X | Filter from `cfr` at this step; deduplicate against Pile-of-Law `cfr` by URL identifier before fuzzy dedup |
| **Caselaw Access Project** | Second-pass relevance classifier (cheap LLM: Nemotron-Nano-9B) scoring `"is this opinion primarily about financial crime? y/n"` on the syllabus/headnote — drop no's. Strip court-administrative boilerplate (syllabi, counsel lists). Preserve `case_name`, `court`, `year` in metadata. |

After common pipeline:

1. Cross-source MinHash dedup
2. Tokenize, pack, shard as in Layer 1
3. 1% validation holdout

**Output:** `3.data_preparation/data/final/cpt/level_2/{shards,val_shards}/`

## 3.4 SFT Preparation

Input: all sources under `1.data_download/data/raw/sft/*/` + Step 2 extractions for FFIEC Manual.

### 3.4.1 Unified Schema

Every record normalized to:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "source": "...",
  "capability": "A|B|C|D|E",
  "metadata": {}
}
```

### 3.4.2 Per-Source Conversion

| Source | Conversion Rule |
|---|---|
| Enterprise Financial Crime AI | Keep only records with `len(sar_report) ≥ 200` OR `len(investigation_summary) ≥ 200`. Template: instruction = "Draft a SAR narrative for [entity] given [transaction_facts] and [KYC_profile]"; output = actual narrative. Records with thin text fields are dropped (no point filling shallow metadata rows). |
| SARSum | **For SFT: only quality level 6** (the "correct" output). Remaining levels 1–5 are reserved for RL. Template: "Summarize the suspicious activity for [entity]" → level-6 summary. |
| AMLGentex | Run generator inline; build instruction pairs: "Analyze these transactions and draft a SAR paragraph identifying the typology" → template SAR response with ground-truth typology. |
| Synthetic (NeMo Data Designer + CPT ckpt) | Generate after CPT finishes. Prompts constructed from: (i) AMLGentex scenarios, (ii) IBM AML transaction clusters, (iii) FFIEC manual topics. Ground-truth answers pulled from source manuals or typology definitions. |
| Finance-Instruct-500k | **Aggressive relevance filter**: keep records in categories `{compliance, regulation, AML, KYC, financial-QA, NER, summarization}`; drop `{stock-prediction, price-forecasting, sentiment}`. Expect ~30% retention. |
| FinQA | Preserve the chain-of-thought reasoning trace in `output`. Teaches step-by-step arithmetic. |
| TAT-QA | Preserve the hybrid table+text context in `input`. |
| FinanceBench | **Preserve the evidence citation in `output`**: append `"Evidence: [exact passage]"`. This format is what the RL citation reward will match against. |
| LegalBench | Only the 16 real HF configs listed at download (covering the 6 logical families). Each sub-task has its own prompt template — keep LegalBench's official formatting. |
| FFIEC Manual | For each scraped section, chunk to ≤ 1500 tokens. For each chunk, generate 2–3 Q/A pairs via NeMo Data Designer prompted with the CPT checkpoint. Answer = direct quote from section. Preserve section ID + URL in metadata for citation. |
| IBM AML Transactions (HI-Small) | Sample transaction clusters grouped by labeled typology. Prompt CPT checkpoint: "Describe the suspicious activity shown in these transactions and classify the typology." Reference answer = rendered typology description with the ground-truth label injected. |
| CFPB Complaints | Template: "Extract SAR-relevant facts from this consumer complaint" → structured JSON output of `{entities, suspicious_facts, recommended_sar_category}`. Facts synthesized from complaint narrative fields. |

### 3.4.3 SFT Quality & Deduplication

1. Length bounds: 50 ≤ `instruction` ≤ 2000 chars; 20 ≤ `output` ≤ 4000 chars
2. Language filter: `en` only
3. **Objectivity pre-filter** — reject outputs matching a denylist of speculative/accusatory phrases (`"definitely"`, `"is guilty of"`, `"clearly laundering"`, `"without a doubt"`, etc.). No point teaching what we'll RL against.
4. MinHash dedup (Jaccard 0.85) across all SFT sources — expect meaningful overlap between Finance-Instruct-500k and FinQA/TAT-QA.
5. **Capability balancing** — after dedup, cap any single source at 30% of the final dataset. If Finance-Instruct dominates, downsample (preserve its category balance). If FinanceBench is tiny, keep all of it.
6. 90 / 5 / 5 train / val / test split, stratified by `(source, capability)`.

**Output:** `3.data_preparation/data/final/sft/{train,val,test}.jsonl`

## 3.5 RL Preparation

Input: SARSum (all quality levels) + held-out splits of LegalBench / FinanceBench / FinQA + SFT checkpoint (used to generate candidates once Phase 4 completes).

### 3.5.1 Preference Pair Unification

All preference-based records:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "source": "...",
  "axis": "objectivity|grounding|hedging|numeric|legal"
}
```

### 3.5.2 Preference Construction

| Source | Rule |
|---|---|
| SARSum expanded | Three preference tiers per SAR set: (a) level 6 vs level 1-2, (b) level 3-4 vs level 1-2, (c) level 6 vs level 3-4 |
| Manual template pairs | Hand-crafted during prep for 8 AML typologies (structuring, smurfing, layering, trade-based ML, shell company, human trafficking, terrorist financing, elder exploitation). Contrast axes: objective vs. accusatory; evidence-cited vs. generalized; deferential vs. autonomous. |
| SFT checkpoint generation | (Built *after* Phase 4 finishes.) Sample held-out SFT prompts; generate 4 candidates at T=0.9; score with reward ensemble (citation-match + legal-match + objectivity-denylist + format-sanity); rank-1 → chosen, rank-4 → rejected. |
| HF distilabel legal-preference pipeline *(optional, if time permits)* | Run HuggingFace cookbook pipeline over FFIEC + FATF + OFAC PDF text to generate regulatory-grounded chosen vs. plausible-but-non-compliant rejected. |

### 3.5.3 Verifiable Reward Datasets

These are not preference pairs; they are prompt + ground-truth pairs consumed online during GRPO by a reward function.

```json
{
  "prompt": "...",
  "ground_truth": "...",
  "reward_fn": "legalbench_rule_qa|financebench_citation|finqa_numeric"
}
```

| Source | Held-Out Split | Reward Function |
|---|---|---|
| LegalBench (16 HF configs across the 6 families) | 30% held out from SFT | Sub-task-specific evaluator (exact match, entailment label, etc.) |
| FinanceBench | 30% held out from SFT | Embedding similarity between model-cited evidence and ground-truth evidence span, thresholded |
| FinQA | 30% held out from SFT | Numeric exact-match on final answer |

**Critical hygiene rule:** any example used in SFT must *not* appear in the RL verifiable set. The 70/30 split is fixed upstream at this step and recorded in `phase_stats.json` with record IDs.

### 3.5.4 RL Quality & Balancing

1. **Length matching** — within each preference pair, `len(chosen) / len(rejected) ∈ [0.67, 1.5]`. Prevents the model from learning "longer = better" (or vice versa).
2. **Axis balancing** — target roughly even representation across `{objectivity, grounding, hedging, numeric, legal}` so no single axis dominates gradient.
3. **Deduplication** — exact + MinHash dedup on `(prompt, chosen)` across sources.

**Output:**
- `3.data_preparation/data/final/rl/preferences.jsonl`
- `3.data_preparation/data/final/rl/verifiable_{legalbench,financebench,finqa}.jsonl`

## 3.6 Step 3 Deliverables

- CPT: `3.data_preparation/data/final/cpt/level_1/` and `level_2/` shards + validation holdouts
- SFT: `3.data_preparation/data/final/sft/{train,val,test}.jsonl`
- RL: `3.data_preparation/data/final/rl/preferences.jsonl` + 3 verifiable reward sets
- `3.data_preparation/data/phase_stats.json` — per-source pre-filter counts, post-filter counts, dedup loss, final record counts per phase
- `3.data_preparation/data_preparation_output.md` — human-readable summary

---

## Pipeline Summary

| Step | Input | Output | Operation |
|---|---|---|---|
| 1. Data Download | URLs | `1.data_download/data/raw/` | Acquisition + structural subsetting |
| 2. PDF Extraction | PDFs from Step 1 | `2.pdf_extraction/data/raw/` | NVIDIA RAG Blueprint / NV-Ingest (nemoretriever-parse + table/chart/graphic NIMs + PaddleOCR fallback) |
| 3. Data Preparation | Step 1 text + Step 2 text | `3.data_preparation/data/final/` | Curation, filtering, cleaning, dedup, format conversion |

Every step writes to the same `raw/` and `final/` hierarchy keyed by training phase (`cpt/level_1`, `cpt/level_2`, `sft`, `rl`), making each step independently restartable and each output trivially auditable.

---

## Source-to-Phase Map (one-page reference)

| Source | CPT L1 | CPT L2 | SFT | RL |
|---|:---:|:---:|:---:|:---:|
| Pile of Law (`sec`, `cfr`, `federal_register`, `uscode`, `oig`, `doj_guidance`) | x | | | |
| EDGAR-CORPUS (SIC 6000–6999) | x | | | |
| FinCEN Advisories / FR / SAR Reviews / Enforcement / Files (narratives) | | x | | |
| FATF Publications | | x | | |
| OFAC Enforcement + Guidance | | x | | |
| 31 CFR Chapter X | | x | | |
| Caselaw Access Project (filtered) | | x | | |
| Enterprise Financial Crime AI | | | x | |
| SARSum (level 6) | | | x | |
| SARSum (all levels, graded) | | | | x |
| AMLGentex | | | x | |
| Synthetic (post-CPT) | | | x | |
| Finance-Instruct-500k (filtered) | | | x | |
| FinQA | | | x | x (held-out) |
| TAT-QA | | | x | |
| FinanceBench | | | x | x (held-out) |
| LegalBench (16 HF configs) | | | x | x (held-out) |
| FFIEC BSA/AML Manual | | | x | |
| IBM AML Transactions (HI-Small) | | | x | |
| CFPB Consumer Complaints | | | x | |
| Manual template preference pairs | | | | x |
| SFT checkpoint generations | | | | x |
