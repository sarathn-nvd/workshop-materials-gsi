# Step 2 — Data Processing

Turns the raw downloads from `1.data_download/data/raw/` into clean, training-ready JSONL under
`data/`. Single CPU-only pass — no RAG cluster, no GPUs.

## What it does

`run_extraction.py` walks every source under `1.data_download/data/raw/`, applies schema-aware
per-source extractors, copies transactional toolset data as-is, parallelises PDF text
extraction with **pypdfium2**, validates outputs, counts tokens with the Nemotron-3-Nano
tokenizer, and writes everything into `summary.json`.

Outputs:

```text
data/cpt/level_1/<source>.jsonl    # broad financial / regulatory corpora
data/cpt/level_2/<source>.jsonl    # AML-specific corpora
data/sft/<source>.jsonl            # text Q&A / instruction-response (Step 5 Data Designer reads this)
data/transactional/<source>/...    # raw files copied as-is (toolset reference)
summary.json                       # per-source stats + per-phase token totals
extraction.log                     # verbose stdout
```

CPU parallelism (240-core box):
- **PDF extraction** — `ProcessPoolExecutor` with `PDF_WORKERS = min(64, ncpu/4)` workers.
- **Tokenisation** — `ProcessPoolExecutor` with `TOK_WORKERS = min(16, ncpu/16)` workers, each
  loading the tokenizer once via the pool initializer.

Total runtime for ~3,000 PDFs and ~1.5 M JSONL records is **a few minutes**, not hours.

## How to run

```bash
cd /data/swami/gsi-training/2.data_processing

# HF auth -- needed for the gated Nemotron-3-Nano tokenizer
export HF_TOKEN=hf_xxx_your_token

PY=/data/swami/gsi-training/jupyter-env/bin/python3
nohup $PY -u run_extraction.py > extraction.log 2>&1 &
echo "PID: $!"
tail -f extraction.log
```

`pypdfium2` is auto-installed if missing (cell-1 of the script). `transformers`, `tokenizers`,
and `sentencepiece` are auto-installed before the token-counting step.

## When the run completes

```bash
# Single source of truth -- machine-readable
cat summary.json | jq '.totals'
```

Schema of `summary.json`:

```json
{
  "tokenizer": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
  "run_started_utc": "...",
  "discovery": {"total_units": 46, "by_phase": {...}, "skipped_files": [...]},
  "totals": {
    "records":  {"cpt_level_1": ..., "cpt_level_2": ..., "sft": ..., "transactional_files": ...},
    "tokens":   {"cpt_level_1": ..., "cpt_level_2": ..., "sft": ..., "grand": ...}
  },
  "durations_s": {"extraction": ..., "validate_and_tokenize": ..., "total": ...},
  "quality_gates": {...},
  "per_source": [
    {"source": "...", "phase": "cpt", "layer": "level_1",
     "files_count": ..., "files_mb": ..., "records": ..., "tokens": ...,
     "duration_s": ..., "output": "data/cpt/level_1/....jsonl", "copy_only": false},
    ...
  ]
}
```
