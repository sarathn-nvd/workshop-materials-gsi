# Stage 1 — Shuffle & chunk

Globally shuffles the partitioned `level_1` JSONL across all 8 source files using `terashuf`, splits the shuffled stream into N evenly-sized chunks, and pulls a held-out validation slice off the top of every chunk.

| | |
|---|---|
| Script | [`shuffle.py`](shuffle.py) (forked from `../../reference/1.shuffle_dataset/` — only change is `--val_pct` replacing the reference's hardcoded `head -n 10000` per chunk) |
| Input | [`../../data/final/level_1/`](../../data/final/level_1/) — 8 sources, ~9.25 GB, ~75 K records |
| Output | `data/level_1_shuffled/level_1.chunk.{00..31}.jsonl` + `level_1.val.jsonl` (~7.5 K records = 10% of corpus) |
| Wall clock | ~3 min on a typical machine |
| Dependencies | Python 3.10+ (stdlib only); `git`, `make`, `g++` for the auto-built `terashuf` binary |

## Why this stage exists

The Megatron `.bin/.idx` packed format that Stage 2 produces preserves document order — the dataloader at training time only **shuffles indices into** the packed file, not the underlying document sequence. If we don't pre-shuffle the text, the model would see all of EDGAR before any of `pile_of_law_oig` (or worse, the chunked `uscode_house` blocks back-to-back). A global shuffle across files breaks per-source autocorrelation so every shard is a representative mix of the full L1 corpus.

`terashuf` is the right tool here because it's a true global shuffle that uses bounded memory (it streams + binary-merges instead of loading everything). For our 9.25 GB input, 16 GB memory is more than enough.

## How to run

Pure host environment (no Docker — just Linux + Python).

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/1.shuffle_dataset

python3 shuffle.py \
    --input_dir   ../../data/final/level_1 \
    --output_dir  ./data/level_1_shuffled \
    --dataset_name level_1 \
    --extension   .jsonl \
    --memory      16 \
    --seed        42 \
    --nchunks     32 \
    --val_pct     5
```

First-run prerequisite: the script auto-clones `https://github.com/alexandres/terashuf` and `make`s it under `./terashuf/` if not present. Re-runs reuse the compiled binary.

### Why these specific args

| Arg | Value | Reason |
|---|---|---|
| `--input_dir` | `../../data/final/level_1` | Output of `prepare-data/split_pools.py`. The 8 `level_1/<source>.jsonl` files are picked up by the `find -name '*.jsonl'` step in shuffle.py. |
| `--dataset_name` | `level_1` | Output prefix. Matches the convention used by Stage 2's tokenizer invocation (it globs `level_1.chunk.*.jsonl`). |
| `--extension` | `.jsonl` | Our files are uncompressed (the curator emits them that way). For `.jsonl.zst` or `.jsonl.gz` the script auto-routes through `zstdcat` / `zcat`. |
| `--memory` | `16` | GB given to terashuf's internal buffer. 9.25 GB input fits comfortably; can drop to 8 if RAM constrained. |
| `--seed` | `42` | Same seed as `prepare-data/split_pools.py` for consistency. Re-runs are bit-identical. |
| `--nchunks` | `32` | Sets parallelism granularity for Stage 2 (each chunk → one tokenizer worker batch) and Stage 3's dataloader (NeMo globs all 32 `.bin/.idx` shards). 32 is a good fit for our 64-CPU host and ~300 MB chunk size. |
| `--val_pct` | `10` | Percentage of total records to pull off the chunks for `level_1.val.jsonl`. ~7.5 K records of our 75 K-record corpus → ~940 records per source on average, plenty for `compare_perplexity.py --max_samples 200..2000`. |

## Validation extraction

Implementation: the script does a one-pass record count over the input directory before invoking terashuf, then derives `val_per_chunk = round(total_records × val_pct / 100) / nchunks`. After splitting, `head -n val_per_chunk` is pulled off each chunk and `sed -i '1,val_per_chunk d'` removes those lines from the chunk. Net effect: a global, post-shuffle, deterministic 10% holdout that's evenly distributed across chunks.

For our corpora at the recommended `--nchunks 32 --val_pct 10`:

| Corpus | Total records | val_per_chunk | Total val | Train remainder |
|---|---:|---:|---:|---:|
| level_1 (this script's input) | ~75,391 | ~235 | ~7,520 | ~67,871 |
| level_2 (used in `../../run-v2/`) | ~8,046 | ~25 | ~800 | ~7,246 |

There's a guardrail: if the resulting val slice would exceed 50% of the corpus, the script aborts with a clear error before invoking terashuf, so misconfigured runs (`--nchunks 100 --val_pct 50` etc.) fail loudly rather than silently producing empty training files.

**Why a percentage instead of a fixed per-chunk count?** The reference's hardcoded `head -n 10000` was sized for its 10 M-doc translation corpus where 320 K val (= 32 × 10 K) is a clean ~3% slice. On our 75 K-record level_1, the same hardcoded value would consume the entire corpus into the val file. A percentage scales correctly across run-v1 (75 K records) and run-v2 (8 K records) without needing per-run tuning.

## Output layout

```
data/level_1_shuffled/
├── level_1.chunk.00.jsonl
├── level_1.chunk.01.jsonl
├── ...
├── level_1.chunk.31.jsonl   # 32 training chunks, ~290 MB each, globally shuffled
└── level_1.val.jsonl        # validation set: top 10K lines pulled off every chunk = 320K records
```

The validation file is stripped from the chunks via `sed -i '1,10000d'`, so there is **zero overlap** between training and val.

## What you should NOT do

- **Don't shuffle within `prepare-data/`.** The partition step ([`../../prepare-data/split_pools.py`](../../prepare-data/split_pools.py)) is intentionally write-in-source-order so chunked-doc neighbors stay adjacent and the partition is reproducible / inspectable. Cross-source shuffling is this stage's job.
- **Don't change `--seed`.** It pins the shuffle for reproducibility. Different seeds → different chunk contents → invalidates Stage 2's `.bin/.idx` and Stage 3's dataloader index cache.
- **Don't drop `--nchunks` below 8.** With 8 GPUs + FSDP, a per-GPU file count below 1 hurts I/O parallelism. 16 or 32 is the sweet spot.
