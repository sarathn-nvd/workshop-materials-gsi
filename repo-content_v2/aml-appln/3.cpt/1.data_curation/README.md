# Step 3.1 — CPT Data Curation

Curates the CPT JSONL files produced by Step 2 (`2.data_processing/data/cpt/level_{1,2}/<source>.jsonl`) into a clean, deduplicated per-source corpus ready for tokenization + packing in the next step.

The pipeline implements `approch.md` Section 6.2 (minus the validation split and Megatron tokenize+pack — those move to a future step).

### Joint dedup, layer-segregated output

Both CPT layers (`level_1` broad financial register + `level_2` AML-specific) are processed in **one run over the union of both layers**, with the layer split applied only at the final write step. This is the standard pattern used by every modern LLM pre-training corpus (FineWeb / RedPajama / Dolma / NeMo CC):

- **Dedup is a corpus-quality concern**: one canonical copy per content. Same FinCEN advisory appearing in both Pile-of-Law SEC and FinCEN Federal Register collapses to one survivor.
- **Curriculum is a training-time concern**: the L1 → L2 ordering from `approch.md` §2.2 is a *training scheduler* decision, not a curation decision.

Each Step 2 record carries a `layer` field; that field rides through every curation phase untouched. The final `WRITE_CURATED` phase is the only stage that reads it, grouping survivors by `(layer, source)` and writing to `cpt/level_1/<source>.jsonl` vs `cpt/level_2/<source>.jsonl`.

---

## Pipeline phases

The pipeline runs as 6 staged phases, each checkpointed to `<output_dir>/_work/checkpoint/meta.json`. Re-run the same command to resume from the last completed phase.

| # | Phase | What happens |
|---|---|---|
| 0 | `INGEST` | Read every `<input_dir>/level_1/*.jsonl` and `<input_dir>/level_2/*.jsonl`, map the Step 2 `content` field to `text`, force-tag every record with its `layer`, **split files >1000 records into chunks** for Ray-pipeline parallelism. → `_work/stages/00_ingest/` |
| 1 | `TEXT_CLEAN` | One Curator `Pipeline`: bytes-repr decode → HTML strip → English text cleanup → FastText `lid.176` (English ≥ 0.5) → length filter → `AddId` (prefix `cpt_`) → quality (alphanumeric / repeated-line / common-word) → boilerplate (per-source recurring-line + YAML denylist) → typed-tag PII redaction. → `_work/stages/01_clean/` |
| 2 | `EXACT_DEDUP` | SHA-256 over the `text` field via `ExactDeduplicationWorkflow` (uses our `id` column directly via `assign_id=False`), then `TextDuplicatesRemovalWorkflow`. → `_work/stages/02_exact/` |
| 3 | `FUZZY_DEDUP` | MinHash 128p (16 bands × 8 hashes), Jaccard ≈ 0.79 via `FuzzyDeduplicationWorkflow`, then `TextDuplicatesRemovalWorkflow`. Uses Curator's internal `_curator_dedup_id` (fuzzy has no `assign_id` knob). → `_work/stages/03_fuzzy/` |
| 4 | `XSOURCE_DEDUP` | Targeted source-pair dedup (richness-scored): for each configured pair (`pile_of_law_uscode ↔ uscode_house`, `pile_of_law_cfr ↔ derived_cfr_31_X`) we hash normalised text, find collisions across the pair, and drop the doc with the lower richness score. → `_work/stages/04_xsource/` |
| 5 | `WRITE_CURATED` | Read `_work/stages/04_xsource/*.jsonl`, group survivors by `(layer, source)`, and write to `cpt/<layer>/<source>.jsonl`. Then enrich `_work/summary.json` with chars, tokens, PII tallies, and layer totals. |

### Key contract notes (each was a runtime bug we hit)

- **INGEST chunks large source files**: Curator's `FilePartitioningStage` cannot intra-file split a JSONL; one 13 GB EDGAR file becomes one serial Ray task that bottlenecks the whole pipeline. We split at INGEST into 1000-record shards (~64 chunks for EDGAR).
- **FUZZY removal points at the `FuzzyDuplicateIds` SUBFOLDER**, not the parent `output_path`. If you give it the parent, pyarrow scans recursively into the MinHash/LSH cache parquets which have `list<int64>` columns and the schema merge fails with `ArrowNotImplementedError: Unsupported cast from list<int64> to int64`.
- **FUZZY removal `input_blocksize` MUST match identification's** (default `"1GiB"`), otherwise the IdGenerator's batch registry can't find the key for the new partitioning → `KeyError`.
- **FUZZY removal `ids_to_remove_duplicate_id_field="_curator_dedup_id"`**, not the SDK default `"id"` — the parquet that fuzzy writes is keyed by the IdGenerator's internal int64 ID, not our doc-level `id`.
- **EXACT removal uses `assign_id=False, id_field="id"`** to bypass the IdGenerator entirely and key on our doc-level `id`. Fuzzy doesn't expose this knob, hence the IdGenerator dance above.

### Bytes-repr decoder (first cleaning step)

Some upstream parquet files carry text columns serialized as `str(some_bytes)` instead of `some_bytes.decode()`, so the JSONL we receive contains literal `b'...\n...'` strings. Without a fix, lid would drop them all as gibberish. The `FixBytesReprModifier` runs as the very first cleaning step and decodes them back. Confirmed-affected source: **`pile_of_law_oig`** (~20K records). All other Step 2 sources are byte-clean and the modifier is a no-op for them.

### What we deliberately do NOT do here

- No validation split (moves to future tokenize/pack step).
- No tokenization or sequence packing (same).
- No NV-Ingest (Step 2 uses `pypdfium2`).
- No Data Designer / LLM generation (§6.2 is heuristic-only).
- No SFT-shape conversion (pipeline refuses to run if `--input_dir` resolves under `data/sft/`).

### PII strategy (locked-in)

- **Typed-tag redaction** (not masking). Tags follow §6.2 step 6: `[SSN]`, `[EIN]`, `[ACCOUNT_ID]`, `[PHONE]`, `[EMAIL]`, `[CREDIT_CARD]`, `[IBAN]`.
- **Detector**: regex-only in v1. PERSON detection NOT enabled — the public legal/regulatory CPT corpus contains case citations, public officials, and statute authors that should NOT be redacted.
- **Audit side-file** is written at `<output_dir>/_work/pii/<source>.audit.jsonl`. Each row records `{doc_id, source, entity_type, span_sha256, span_len, replacement}`. The original PII span is **never** persisted — only its salted SHA-256.

### Cross-source dedup ("richer wins")

```
richness_score(doc) = 0.5 * len(text)
                    + 0.3 * unique_section_headings(text)
                    + 0.2 * citation_density(text)
```

Within each cluster the highest scorer wins; the rest are dropped. Default pairs:
- `pile_of_law_uscode` ↔ `uscode_house`
- `pile_of_law_cfr` ↔ `derived_cfr_31_X`

---

## File layout

```text
3.cpt/1.data_curation/
├── config.py                 # dataclass-based config (joint dedup, single threshold set)
├── main.py                   # staged orchestration + checkpointing
├── utils.py                  # custom DocumentModifiers + IO + xsource helper
├── pii_patterns.py           # EIN / account-id / routing regex helpers
├── boilerplate/
│   └── denylists.yaml        # per-source hand-curated boilerplate denylists
└── README.md
```

Outputs (under `--output_dir`) mirror Step 2's folder layout:

```text
<output_dir>/
├── cpt/
│   ├── level_1/<source>.jsonl
│   └── level_2/<source>.jsonl
└── _work/
    ├── models/lid.176.bin
    ├── stages/
    │   ├── 00_ingest/{<layer>__<source>.jsonl, <layer>__<source>__chunkNNNN.jsonl}
    │   ├── 01_clean/*.jsonl
    │   ├── 02_exact/*.jsonl
    │   ├── 03_fuzzy/*.jsonl
    │   ├── 03_fuzzy_by_source/<source>.jsonl
    │   └── 04_xsource/<source>.jsonl
    ├── checkpoint/meta.json
    ├── cache/{exact, fuzzy/{cache, FuzzyDuplicateIds, fuzzy_id_generator.json}}
    ├── pii/{<source>.audit.jsonl, .salt}
    ├── log_data_curator.txt
    └── summary.json
```

### Logs and summary

| File | Content |
|---|---|
| `_work/log_data_curator.txt` | Running log (mirrors stdout). Per-phase timing, recurring-line counts, dedup-workflow timing, drop-rate logs. |
| `_work/summary.json` | Single rich per-(layer, source) report: per-phase record counts (drop trace), final chars, final tokens (Nemotron-3-Nano tokenizer), PII redaction tallies, plus `layer_totals`. Phase counts append after each phase completes; chars/tokens/PII/totals fill in at the end. |

Token counting requires HF auth for the gated Nemotron-3-Nano tokenizer:

```bash
export HF_TOKEN=hf_xxx_your_token   # same token Step 2 uses
```

If `HF_TOKEN` is unset, `tokens` is reported as `null` in the summary; char counts still populate.

`summary.json` shape:

```jsonc
{
  "tokenizer": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
  "by_layer": {
    "level_1": {
      "edgar_corpus": {
        "chars": 487213991,
        "tokens": 132417553,
        "phase_counts": {
          "INGEST":        63366,
          "TEXT_CLEAN":    60767,
          "EXACT_DEDUP":   60767,
          "FUZZY_DEDUP":   51019,
          "XSOURCE_DEDUP": 51019
        },
        "pii_redactions": { "PHONE": 4, "EMAIL": 12 }
      }
    },
    "level_2": { "...": "..." }
  },
  "layer_totals": {
    "level_1": { "final_records": 96213, "chars": 4_293_812_991, "tokens": 1_104_211_553, "sources": 8 },
    "level_2": { "final_records":  3018, "chars":   133_402_117, "tokens":    35_491_217, "sources": 9 }
  }
}
```

---

## How to run

The pipeline runs **inside the `nvcr.io/nvidia/nemo-curator:26.02` container** so the Ray-based `nemo_curator` SDK + GPU dedup workflows are available.

### Pre-flight (one-time)

```bash
export HF_TOKEN=hf_xxx_your_token       # for token-counting in summary.json (optional)
docker pull nvcr.io/nvidia/nemo-curator:26.02
```

### Full-corpus run (host invocation)

```bash
cd /data/swami/gsi-training/3.cpt/1.data_curation

nohup docker run --rm --gpus all \
  --name cpt-curator \
  --shm-size=64g \
  --user "$(id -u):$(id -g)" \
  --group-add 0 \
  -e HOME=/tmp -e HF_HOME=/tmp/hf_cache \
  -v /data/swami/gsi-training:/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/nemo-curator:26.02 \
  python -u /workspace/3.cpt/1.data_curation/main.py \
    --input_dir  /workspace/2.data_processing/data/cpt \
    --output_dir /workspace/3.cpt/1.data_curation/data \
  > curation.log 2>&1 &

echo "PID: $!"
sleep 2 && tail -f curation.log
```

`/workspace` inside the container maps to `/data/swami/gsi-training` on the host.

**Why each non-obvious flag:**

| Flag | Why |
|---|---|
| `--name cpt-curator` | Stable name so `docker stop cpt-curator` works without hunting for IDs. |
| `--shm-size=64g` | Docker's default `/dev/shm` is 64 MB; Ray's object store falls back to `/tmp` and the LSH shuffle in `FUZZY_DEDUP` becomes 2–3× slower. |
| `--user "$(id -u):$(id -g)"` | Output files under `data/` are owned by you, not root. Without this you'd need `sudo` to clean up. |
| `--group-add 0` | Image installs `nemo_curator` editable at `/opt/Curator/` with mode `0660` (group-readable, not other-readable). Adding the root group lets the non-root user import it. Without this you get `PermissionError: '/opt/Curator/nemo_curator/__init__.py'` immediately. |
| `-e HOME=/tmp -e HF_HOME=/tmp/hf_cache` | Non-root user inside the container has no real `$HOME`; `transformers` / `loguru` write caches to `/tmp` instead. |

### Resume / reset

Re-run the same command to resume from the last completed phase. Pass `--reset` to wipe the checkpoint and start over.

### Aborting / clean re-run

The Python process runs **inside the Docker container as the user you mapped**. Stop the container (not the host process):

```bash
cd /data/swami/gsi-training/3.cpt/1.data_curation

docker stop cpt-curator 2>/dev/null || true

# Confirm:
docker ps --filter ancestor=nvcr.io/nvidia/nemo-curator:26.02

# Wipe outputs (use sudo only if previous runs landed root-owned files):
rm -rf data curation.log
# OR if needed:
sudo chown -R "$USER:$USER" data && rm -rf data curation.log
```

To resume, just relaunch — the checkpoint preserves what's done.

### Expected runtime

Per `approch.md` §6.6 + observed throughput on this corpus:

| Phase | Wall-clock |
|---|---|
| INGEST | ~2 min |
| TEXT_CLEAN | ~10 min |
| EXACT_DEDUP | ~3 min |
| FUZZY_DEDUP | ~10–20 min (8 H100s; with `--shm-size=64g`) |
| XSOURCE_DEDUP | < 2 min |
| WRITE_CURATED + token summary | ~10–15 min |

End-to-end: **~30–60 min on 8×H100** for the full corpus (~14.7 GB raw text, ~101K records).

---

## Customisation

Everything tunable lives in `config.py`:

- **Length bounds**: `LengthConfig.{min_chars, max_chars}` (defaults 200 / `None`). `max_chars=None` disables the upper bound — needed for whole-statute corpora (USC/CFR/uscode_house) which have 2-10 MB single-record documents.
- **Quality**: `QualityConfig.{max_non_alphanumeric_ratio, min_unique_line_ratio, min_common_words}`.
- **Boilerplate**: edit `boilerplate/denylists.yaml`; tune `BoilerplateConfig.recurring_line_doc_ratio` (default 1%).
- **PII**: `PiiConfig.entity_tags`, `enable_ein_regex`, `enable_account_id_regex`, `write_audit`.
- **Fuzzy dedup**: `FuzzyDedupConfig.{char_ngrams, num_bands, minhashes_per_band, bands_per_iteration, input_blocksize}`. **`input_blocksize` MUST match between identification and removal**; both default to `"1GiB"`.
- **Cross-source pairs**: append to `XSourceDedupConfig.pairs`.
- **Layer subdirs**: `PipelineConfig.layer_dirs` (default `("level_1", "level_2")`).
