# Step 1 — Data Download

Acquires all raw training data for the AML Investigation Agent (CPT / SFT / RL) per `revised_strategy.md`. PDFs are saved as-is; text extraction happens in Step 2 (NV-Ingest), curation in Step 3.

## What It Does

- Pulls HuggingFace datasets with per-source structural filters (HF configs, SIC codes, jurisdictions, sub-tasks)
- Downloads Kaggle datasets (SARSum, IBM AML Transactions HI-Small)
- Scrapes PDFs from FinCEN, FATF, OFAC, FFIEC
- Downloads CSV / bulk archives from data.gov (FinCEN enforcement, CFPB complaints) and OpenSanctions (OFAC)
- Fetches the ICIJ FinCEN Files release
- Clones the AMLGentex generator repo (used later in Phase 3 for synthetic SFT data)
- Writes everything under a canonical `data/raw/<phase>/<source>/` layout and produces `data/phase_stats.json`

## Prerequisites

- Python 3.10+ (the shared `../jupyter-env/` virtualenv uses 3.12)
- HuggingFace token with read access — exported as `HF_TOKEN`. The token's account must have accepted the terms of `free-law/Caselaw_Access_Project` at https://huggingface.co/datasets/free-law/Caselaw_Access_Project (otherwise that one source will fail with 403; everything else still works).
- Kaggle credentials at `~/.kaggle/kaggle.json` (JSON form `{"username":"...","key":"..."}`) — required for SARSum and IBM AML Transactions
- ~300 GB free disk

## One-time Setup

Step 1, Step 2, and the Jupyter dev environment share a single virtualenv at `../jupyter-env/` (one level above this folder). The Step 1 dependency list lives at the repo root as `../requirements.txt`.

```bash
cd 1.data_download
./setup.sh
source ../jupyter-env/bin/activate
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
export COURTLISTENER_TOKEN="xxxxxxxxx"
```

`setup.sh` creates `../jupyter-env/` if missing, installs `../requirements.txt` into it, and downloads the Playwright Chromium browser used for FATF.

## Pre-flight Checklist

Run these immediately before launching to confirm everything is wired:

```bash
pwd                                                    # .../1.data_download
which python                                           # .../jupyter-env/bin/python
echo "${HF_TOKEN:0:3}"                                 # prints "hf_"
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; \
           a=KaggleApi(); a.authenticate(); print('ok')"
df -h .                                                # ≥ 300 GB free
```

## Running

### End-to-end run (background, survives terminal close)

```bash
nohup python -u download.py --tasks all --parallel-workers 8 \
  > run.out 2>&1 &
echo $! > download.pid

tail -f run.out                                        # watch progress
```

### Selective runs

Each task is independently restartable. Already-downloaded files are skipped on re-run.

```bash
python download.py --tasks cpt_l1                     # Pile of Law + EDGAR only
python download.py --tasks cpt_l2                     # All CPT Layer 2 sources
python download.py --tasks sft                        # All SFT sources
python download.py --tasks fincen,fatf                # Specific source groups
python download.py --tasks huggingface                # All HuggingFace-hosted sources
python download.py --tasks kaggle                     # SARSum + IBM AML only
python download.py --list                             # List all tasks and groups
```

### Monitoring

```bash
tail -f run.out                                        # live stdout/stderr
tail -f output.log                                     # structured per-source log
ps -p "$(cat download.pid)" && echo "still running"   # health check
jq '.sources | to_entries | map({k:.key, status:.value.status, mb:(.value.bytes_written/1048576|floor)})' \
  data/phase_stats.json                                # per-source results
```

### Stopping

```bash
kill "$(cat download.pid)"
```

The script is restartable — re-running the same `--tasks` selection picks up where it left off.

## Outputs

- `data/raw/<phase>/<source>/…` — raw downloaded artifacts
- `data/phase_stats.json` — per-source bytes, file count, structural-filter pass-rate
- `data_download_output.md` — human-readable summary generated at end of run
- `run.out` — stdout/stderr capture (only when launched with `nohup ... > run.out`)
- `output.log` — structured per-source log (always written)

---

## Dataset Inventory

Every source from `revised_strategy.md`, classified by **training phase** (where it goes in the model pipeline) and **data type** (what shape the raw data is in).

### CPT Layer 1 — Financial Regulatory & Corporate Compliance

Teaches the model to write like a financial regulator.

| Source | HF / URL | Data Type | Filter Applied at Download | Output Path |
|---|---|---|---|---|
| Pile of Law `sec` | `pile-of-law/pile-of-law` (config `sec_administrative_proceedings`) | Long-form text | None (all SEC enforcement) | `data/raw/cpt/level_1/pile_of_law_sec/` |
| Pile of Law `cfr` | same (config `cfr`) | Long-form text | Title 12 (Banks), 17 (SEC/CFTC), 31 (Money & Finance) | `data/raw/cpt/level_1/pile_of_law_cfr/` |
| Pile of Law `federal_register` | same | Long-form text | Financial agencies only (FinCEN/Treasury/SEC/CFTC/FDIC/OCC/Fed) | `data/raw/cpt/level_1/pile_of_law_federal_register/` |
| Pile of Law `uscode` | same | Long-form text | Title 12, Title 18 Ch.95 (RICO) & Ch.96 (§§1956–1957), Title 31 | `data/raw/cpt/level_1/pile_of_law_uscode/` |
| Pile of Law `oig` | same | Long-form text | Treasury/FDIC/OCC/Fed OIGs only | `data/raw/cpt/level_1/pile_of_law_oig/` |
| Pile of Law `doj_guidance` | same (config `doj_guidance_documents`) | Long-form text | None (already corporate-enforcement-focused) | `data/raw/cpt/level_1/pile_of_law_doj_guidance/` |
| EDGAR-CORPUS | `eloukas/edgar-corpus` | Long-form text (10-K filings) | SIC 6000–6999 (Finance, Insurance, Real Estate) | `data/raw/cpt/level_1/edgar_corpus/` |

### CPT Layer 2 — AML-Specific Domain Corpus

Narrow domain adaptation for ML typologies, BSA, SAR writing, sanctions, FinCEN/FATF guidance.

| Source | URL | Data Type | Output Path |
|---|---|---|---|
| FinCEN Advisories | `fincen.gov/resources/advisoriesbulletinsfact-sheets` | Government PDFs | `data/raw/cpt/level_2/fincen_advisories/pdfs/` |
| FinCEN Federal Register Notices | `fincen.gov/resources/statutes-regulations/federal-register-notices` | Government PDFs | `data/raw/cpt/level_2/fincen_federal_register/pdfs/` |
| FinCEN SAR Activity Reviews | `fincen.gov/sar-activity-review-trends-tips-issues` | Government PDFs | `data/raw/cpt/level_2/fincen_sar_reviews/pdfs/` |
| FinCEN Enforcement | `catalog.data.gov` (CKAN) + `fincen.gov` scrape fallback | Government PDFs + CSV metadata | `data/raw/cpt/level_2/fincen_enforcement/` |
| FinCEN Files (ICIJ) | `icij.org/.../download-fincen-files-transaction-data/` | Investigative PDFs + CSV (transactional) | `data/raw/cpt/level_2/fincen_files/{pdfs,data}/` |
| FATF Publications | 5 publication index pages on `fatf-gafi.org` | Government PDFs | `data/raw/cpt/level_2/fatf_publications/pdfs/` |
| OFAC Enforcement | `data.opensanctions.org/datasets/.../us_ofac_enforcement_actions/` | Bulk JSON / CSV | `data/raw/cpt/level_2/ofac_enforcement/` |
| OFAC Guidance | `ofac.treasury.gov/file-finder` | Government PDFs | `data/raw/cpt/level_2/ofac_guidance/pdfs/` |
| Caselaw Access Project | `free-law/Caselaw_Access_Project` (HF, gated) | Court-opinion parquet | `data/raw/cpt/level_2/caselaw_access_project/{raw_parquet,filtered}/` |
| 31 CFR Chapter X | (derived from `pile_of_law_cfr` at Step 3) | — | n/a — no separate download |

### SFT — Instruction-Response Sources

Teaches specific capabilities (SAR drafting, numerical reasoning, citation, classification, statutory Q&A).

| Source | URL | Data Type | Capability | Output Path |
|---|---|---|---|---|
| Enterprise Financial Crime AI | `Webopen2026/enterprise-financial-crime-ai-dataset` | Instruction CSVs (310K SAR-drafting records) | A. SAR drafting | `data/raw/sft/enterprise_financial_crime/` |
| SARSum | `kaggle.com/datasets/leonardovalves/sarsum` | JSON (2K SARs × 6 quality variants) | A. SAR drafting | `data/raw/sft/sarsum/` |
| AMLGentex | `github.com/aidotse/AMLGentex` | Generator repo (run inline at Step 3) | A, D | `data/raw/sft/amlgentex/repo/` |
| Finance-Instruct-500k | `Josephgflowers/Finance-Instruct-500k` | Instruction JSON | E. Compliance Q&A | `data/raw/sft/finance_instruct_500k/` |
| FinQA | `dreamerdeo/finqa` | Q&A parquet (numeric reasoning) | B. Numerical reasoning | `data/raw/sft/finqa/default/` |
| TAT-QA | `next-tat/TAT-QA` | Q&A JSON (table+text) | B. Numerical reasoning | `data/raw/sft/tat_qa/` |
| FinanceBench | `PatronusAI/financebench` | Q&A JSONL (with evidence spans) | C. Citation-grounded answering | `data/raw/sft/financebench/` |
| LegalBench (16 configs) | `nguha/legalbench` | Per-subtask TSV | E. Statutory reasoning | `data/raw/sft/legalbench__<subtask>/` (×16) |
| FFIEC BSA/AML Manual | `bsaaml.ffiec.gov/manual/Introduction/01` | HTML sections | E. Compliance Q&A | `data/raw/sft/ffiec_manual/html/` |
| **IBM AML Transactions HI-Small** | `kaggle.com/.../ibm-transactions-for-anti-money-laundering-aml` | **Transactional CSV** (HI-Small only) | D. Typology classification | `data/raw/sft/ibm_aml_transactions/` |
| **CFPB Consumer Complaints** | `files.consumerfinance.gov/ccdb/complaints.csv.zip` (Socrata fallback) | **Transactional CSV** (consumer complaints) | D. Entity extraction | `data/raw/sft/cfpb_complaints/` |

### RL — Reward / Preference Sources

**No separate downloads.** Per `revised_strategy.md` §1.4, RL re-uses SFT artifacts:
- 30% holdouts of LegalBench / FinanceBench / FinQA → verifiable-reward sets
- All-quality SARSum levels → graded preference pairs
- Manual template pairs + post-CPT generations → built at Step 3

Held-out splits are materialized in `3.data_preparation/data/final/rl/` at Step 3.

### Transactional / Tool Data — Note

Three sources contain raw transactional records that downstream systems may also consume as agent-tool data, in addition to their SFT training role:

| Source | Why it's transactional | Where the structured rows go downstream |
|---|---|---|
| FinCEN Files (ICIJ) | CSV of suspicious transactions + narrative PDFs | Step 3 splits this: narratives → CPT L2; structured rows → agent tools |
| IBM AML Transactions HI-Small | Account- and transaction-level CSVs with labeled typologies | Used as SFT capability-D training; underlying CSV is also tool-ready |
| CFPB Consumer Complaints | Complaint narratives + product/issue/state metadata | SFT capability-D training; structured fields are tool-ready |

---

## Data Folder Organization

The script writes to the canonical layout from `revised_strategy.md` §"Canonical Folder Structure":

```text
1.data_download/data/
├── phase_stats.json              # per-source bytes, file count, filter pass-rate
└── raw/
    ├── cpt/
    │   ├── level_1/              # 7 sources: 6 Pile-of-Law subsets + EDGAR-CORPUS
    │   │   ├── pile_of_law_sec/
    │   │   ├── pile_of_law_cfr/
    │   │   ├── pile_of_law_federal_register/
    │   │   ├── pile_of_law_uscode/
    │   │   ├── pile_of_law_oig/
    │   │   ├── pile_of_law_doj_guidance/
    │   │   └── edgar_corpus/
    │   └── level_2/              # 9 sources
    │       ├── fincen_advisories/pdfs/
    │       ├── fincen_federal_register/pdfs/
    │       ├── fincen_sar_reviews/pdfs/
    │       ├── fincen_enforcement/{ckan_csv,scraped_pdfs,case_documents}/
    │       ├── fincen_files/{data,pdfs}/
    │       ├── fatf_publications/pdfs/
    │       ├── ofac_enforcement/
    │       ├── ofac_guidance/pdfs/
    │       └── caselaw_access_project/{raw_parquet,filtered}/
    └── sft/                      # 11 logical sources (legalbench expands to 16 dirs)
        ├── enterprise_financial_crime/
        ├── sarsum/
        ├── amlgentex/repo/
        ├── finance_instruct_500k/
        ├── finqa/default/
        ├── tat_qa/
        ├── financebench/
        ├── legalbench__rule_qa/
        ├── legalbench__sara_entailment/
        ├── legalbench__sara_numeric/
        ├── legalbench__hearsay/
        ├── legalbench__consumer_contracts_qa/
        ├── legalbench__contract_nli_confidentiality_of_agreement/
        ├── legalbench__contract_nli_explicit_identification/
        ├── legalbench__contract_nli_limited_use/
        ├── legalbench__contract_nli_no_licensing/
        ├── legalbench__contract_nli_notice_on_compelled_disclosure/
        ├── legalbench__contract_nli_return_of_confidential_information/
        ├── legalbench__supply_chain_disclosure_disclosed_accountability/
        ├── legalbench__supply_chain_disclosure_disclosed_training/
        ├── legalbench__supply_chain_disclosure_disclosed_verification/
        ├── legalbench__supply_chain_disclosure_best_practice_accountability/
        ├── legalbench__supply_chain_disclosure_best_practice_training/
        ├── ffiec_manual/html/
        ├── ibm_aml_transactions/
        └── cfpb_complaints/
```

There is **no** `data/raw/rl/` directory at Step 1 — per the strategy, RL splits are materialized at Step 3 (Data Preparation) from SFT artifacts.

The same `data/raw/<phase>/<source>/` layout is reused by Steps 2 and 3 — a PDF written here at `data/raw/cpt/level_2/fatf_publications/pdfs/<file>.pdf` produces its extracted text at `2.pdf_extraction/data/raw/cpt/level_2/fatf_publications/<file>.txt` and ends up contributing shards to `3.data_preparation/data/final/cpt/level_2/shards/`.
