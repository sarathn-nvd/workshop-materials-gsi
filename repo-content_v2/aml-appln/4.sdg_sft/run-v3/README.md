# AML SFT Synthetic Data Generation — v3 Pipeline

The v3 pipeline produces the SFT corpus for the AML investigation language
model used by the production agent in
`/data/swami/gsi-training/10.appln_buildout/backend/`.

**Read [`SDG_STRATEGY_SFT.md`](./SDG_STRATEGY_SFT.md) first** — it specifies
the SFT objective, the target agentic application that consumes the model,
the corpus shape, and what each pipeline stage does. Everything in this
README assumes you have that context.

---

## What v3 generates

| Component             | Records | Share |
|-----------------------|--------:|------:|
| `sar_judgment`        |  50,000 |  67%  |
| `auxiliary_numeric`   |   6,250 |   8%  |
| `auxiliary_citation`  |   6,250 |   8%  |
| `auxiliary_statutory` |   6,250 |   8%  |
| `auxiliary_behavioral`|   6,250 |   8%  |
| **Total**             |  75,000 | 100%  |

The pipeline oversamples ~1.5× (≈ 115K records) so the strict filter and
quota cap can drop sub-quality records and land at 75K.

### v3 corpus invariants

These are enforced by the per-record validators and the corpus-level audits
in Stage 9 (see `scripts/validators/`):

* **No label leakage in the user message.** The `sar_judgment` user message
  has *exactly* 7 keys — `task_type, transactions, kyc_profile,
  sanctions_pep_hits, policy_excerpts, sop_excerpts, auxiliary_findings`.
  No `_decision_target`, `_regulatory_frame`, or `_typology_inferred`.
  Auxiliary records carry only `{task_type, passage, question}` (or
  `{task_type, statute, fact_pattern, question}` for statutory).
* **Both classes have grounded narratives.** Every `is_suspicious=false`
  `sar_judgment` record carries a 400–800 char *disposition rationale*
  that names the surface red flag and cites the bundle disambiguator.
  Positive narratives follow the regulator-grade SAR style.
* **No frame is one-sided.** Every regulatory frame in the corpus has
  ≥ 30% minority-class representation. `benign` carries some positives
  (mixed-signal façade cases), `sanctions` carries some negatives
  (common-name PEP / OFAC-collision noise from `scripts/pools/sanctions_noise.py`).
* **`te` (terrorist exploit) frame is dropped.** It was 100% one-sided in
  earlier corpora; `terrorist_financing` typology now maps to the
  `trafficking` frame (FFIEC/FATF group them together).
* **Layering capped at 25%.** Enforced by `RULE-1-CAP-LAYERING` in Stage 9.
* **Class balance 45/55.** Positives/negatives, ±2pp tolerance.

---

## Pipeline layout

```
4.sdg_sft/run-v2/
├── README.md                    ← this file
├── SDG_STRATEGY_SFT.md          ← the strategy doc (read first)
├── data/                        ← outputs (generated; not committed)
│   ├── interim/                 (per-stage parquet)
│   ├── final/                   (the two deliverable JSONLs)
│   ├── manifests/               (audit reports, per-record verdicts)
│   └── logs/                    (per-stage subprocess logs)
├── tests/                       ← pytest suite — see "Tests" section below
│   ├── test_stage_7_prompt.py   (52 tests: prompt contract + LLM behaviour)
│   └── test_aux_pipeline.py     (25 tests: AML statute catalog + projectors + LLM)
└── scripts/
    ├── main.py                  ← entry point: spawns 2 pipelines in parallel
    ├── audit.py                 ← single audit (per-record / judge / corpus / all)
    ├── filter_and_cap.py        ← strict-judge filter + 75K quota cap
    ├── config.py                ← paths + endpoints + concurrency + distribution targets
    ├── schemas.py               ← Pydantic schemas (single source of truth)
    ├── non_auxiliary/           ← 9-stage SAR-judgment pipeline
    ├── auxiliary/               ← 4-stage auxiliary pipeline
    ├── pools/                   ← source-pool loaders + v3 additions:
    │                              `aml_statutes.py` (synthetic AML catalog)
    │                              `sanctions_noise.py` (common-name PEP seeds)
    ├── validators/              ← per-record + corpus-level rules
    └── common/                  ← shared infrastructure (DataDesigner factories,
                                   semantic_profile, etc.)
```

---

## The 3-command workflow

### 1. Generate the corpus

```bash
cd /data/swami/gsi-training/4.sdg_sft
source env/bin/activate
cd run-v2

# Smoke at N=300 (~20s with --dry-run, ~5 min real):
python -m scripts.main --total-records 300 --dry-run
python -m scripts.main --total-records 300

# Full corpus — 75K target, 1.5× oversample → 115K records:
python -m scripts.main --total-records 115000
```

The pipeline spawns two `multiprocessing.Process` children — one for the
non-auxiliary (`sar_judgment`) pipeline (9 stages), one for the auxiliary
pipeline (4 stages). They run in parallel and share the local gemma
cluster's concurrency budget.

**Background invocation** (recommended for full runs):

```bash
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
nohup python -u -m scripts.main --total-records 115000 \
    > data/logs/${RUN_ID}.out \
    2> data/logs/${RUN_ID}.err \
    < /dev/null &
echo $! > data/logs/${RUN_ID}.pid
```

**Outputs (raw):**

* `data/final/sar_judgment_non_auxillary_corpus.jsonl` (≈ 0.67 × N)
* `data/final/auxiliary_corpus.jsonl` (≈ 0.33 × N)

### 2. Audit

```bash
python -m scripts.audit all
```

Runs three audit modes in sequence: per-record (re-applies every per-record
validator and emits a JSONL of per-record verdicts), judge (LLM-judge pass
with per-bucket scoring), corpus (re-applies every corpus-level audit and
compares against the §4.2 distribution targets).

**Outputs:**

* `data/manifests/audit_per_record_{nonaux,aux}.jsonl`
* `data/manifests/audit_judge_<bucket>.jsonl`
* `data/manifests/audit_report.json`

### 3. Strict-filter + cap to 75K

```bash
python -m scripts.filter_and_cap --target-records 75000
```

Consumes the audit reports and the raw JSONLs, drops any record flagged by
the per-record or judge audits, dedups via MinHash, and samples down to
the 75K quota while preserving the §4.2 distribution targets.

**Outputs:**

* `data/final/sft_corpus_75k.nonaux.jsonl` (the deliverable)
* `data/final/sft_corpus_75k.aux.jsonl` (the deliverable)
* `data/manifests/filter_report.json`
* `data/manifests/missing_records.json`

---

## Tests

Run the static + stage-logic tests (fast, no LLM):

```bash
cd /data/swami/gsi-training/4.sdg_sft && source env/bin/activate
cd run-v2

pytest tests/ -v -m "not live"
```

Run the live-LLM prompt-validation tests against the local gemma cluster:

```bash
RUN_LIVE_PROMPT_TESTS=1 pytest tests/ -v
```

Test coverage:

| File | Tests | Layer |
|---|---:|---|
| `test_stage_7_prompt.py` | 38 static + 14 live | SAR-judgment prompt contract, bundle shape, extractor, per-record validators, end-to-end LLM behaviour on 4 canonical cases (elder-exploitation, laundromat cash, common-name PEP, sub-CTR near-miss) |
| `test_aux_pipeline.py`   | 20 static + 5 live | AML statute catalog completeness, synthetic-AML / sarsum / EFC / FFIEC projector shapes, Stage A2 routing, end-to-end LLM behaviour on synthetic AML statutes + Q/A generation |

The live-LLM tests are the most valuable signal — they confirm that the
new v3 prompt actually steers gemma to the expected verdicts and that
the model derives the correct entailment/contradiction/neutral label on
held-back synthetic AML statute records.

---

## Concurrency / deployment

* **LLM endpoint**: `http://localhost:8080/v1`, model `google/gemma-4-31b-it`,
  served by 4 vLLM replicas behind nginx (see `../model_deployment/`).
  Override via `LLM_ENDPOINT` / `LLM_MODEL` env vars.
* **Concurrency caps**: `scripts/config.py::Concurrency`. Defaults:
  `max_parallel_llm=256`, `max_cpu_workers=200`. When SAR + aux pipelines
  run in parallel each gets ~128 concurrent LLM slots.
* **Wall-clock estimate** at full N=115K: ~6h on the current cluster,
  dominated by Stage 7 (now generating narratives for *both* classes,
  about 2× the v2 cost in this stage) and Stage 6 + A2 (aux findings +
  aux generation).

---

## Failure recovery

* **Pipeline fail mid-run**: re-run with `--resume-from <stage_id>`.
  Each stage reads its predecessor's interim parquet/jsonl; no work
  is lost.
* **Audit conflict (concurrent runs)**: the judge audit holds a
  lockfile at `data/manifests/_audit_judge_full.lock`. Pass `--force`
  to `audit judge` if no other instance is active.
* **Disk pressure**: `data/_topup_runs/` (legacy) is safe to delete
  after a run completes. Interim parquets in `data/interim/` are also
  disposable once the final JSONLs are written.

---

## v3-only knobs you may want to tune

All in `scripts/config.py`:

| Knob | Default | What it controls |
|---|---|---|
| `RECORD_PATH_SHARES` | R1=14% R2=10% R3=10% R5=18% R6=40% R7=8% | Source-path mix in Stage 1 — lifts negative paths so the corpus lands near 45/55. |
| `AUX_TASK_SHARES` | 25%/25%/25%/25% | Aux task-type mix. |
| `audit_mix_label` target | 45/55 positive/negative | Adjusts how Stage 9 measures the class-balance audit. |
| `audit_cap_layering` cap | 25% | Maximum share of the corpus that `layering_passthrough` may occupy. |
| `audit_frame_label_balance` min_minority_share | 30% | Each frame must have ≥ this fraction in its minority polarity. |
| `audit_narrative_length_matched` median_tolerance_pct | 20% | Positive and negative narrative medians must be within this %. |
| `_SANCTIONS_NOISE_NEG_RATE` in `stage_1_drivers/stage.py` | 6% | Fraction of negatives flagged for sanctions-noise injection at Stage 4. |
| `POLICY_BROAD_K` in `stage_5_grounding/stage.py` | 2 | Number of typology-agnostic broad-search policy excerpts added as a safety net. |

---

## Strategy reference

All distribution rules, gate thresholds, and prompt contracts are in
[`SDG_STRATEGY_SFT.md`](./SDG_STRATEGY_SFT.md). Read it before changing
any prompt or validator.
