# Stage 1 — Shuffle & chunk (native + replay independently)

Globally shuffles the partitioned `level_2` JSONL across native AML sources and EDGAR-replay **separately**, producing two distinct shard sets so Stage 3's dataloader can apply a 25/75 weighted blend at training time.

Reuses [`../../run-v1/1.shuffle_dataset/shuffle.py`](../../run-v1/1.shuffle_dataset/shuffle.py) unchanged — invoked twice with different `--input_dir` / `--output_dir` / `--dataset_name`.

| | |
|---|---|
| Script | `../../run-v1/1.shuffle_dataset/shuffle.py` (no local copy needed) |
| Input | `data/level_2_split/{native,replay}/` (symlinks into [`../../data/final/level_2/`](../../data/final/level_2/)) |
| Output | `data/level_2_native_shuffled/` + `data/level_2_replay_shuffled/` |
| Wall clock | ~30 s native + ~4 min replay |
| Dependencies | Python 3.10+ (stdlib only); `git`, `make`, `g++` for the auto-built `terashuf` binary (run-v2 will re-build under `data/terashuf/` on first run if not present) |

> **Note (May 8, 2026)**: `shuffle.py` was patched at line 348 to use `find -L` (was `find`) so it dereferences the symlinks under `data/level_2_split/{native,replay}/`. Without that patch, `find -type f` skips symlinks and terashuf reads zero bytes — chunks come out empty even though `Scanned: N records` looks fine. If you ever see a re-run produce 0-byte chunks, verify `shuffle.py` line 348 still has `find -L`.

## Why two shuffles instead of one

Phase 1's `shuffle.py` globs every file in `--input_dir` and cats them together before terashuf. Correct for a single-corpus run (L1) but it **loses per-source identity**. Phase 2 needs two separate `.bin/.idx` shard sets so Megatron's `BlendedDataset` can sample them at the 25/75 ratio set in [`../3.run_cpt/recipe_a100sxm-8.yaml`](../3.run_cpt/recipe_a100sxm-8.yaml). That separation must happen at this stage — once tokenized into a single shard, the weight knob is gone.

See [`../README.md`](../README.md) §"The conceptual gap from run-v1 — routing vs weighting" for the full rationale.

## Pre-stage: split level_2 into native vs replay (already done)

`prepare-data/split_pools.py` co-located 8 native L2 sources and the 1 EDGAR-replay file in `data/final/level_2/`. The shuffle step wants each shard-set in its own directory so `terashuf` globs only the intended files. Symlinks (already created in `data/level_2_split/`) keep the original partition untouched and reproducible:

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v2

mkdir -p 1.shuffle_dataset/data/level_2_split/native
mkdir -p 1.shuffle_dataset/data/level_2_split/replay

# EDGAR-replay: the single file routed to level_2 by split_pools (13% of EDGAR)
ln -sf "$(realpath ../data/final/level_2/edgar_corpus.jsonl)" \
       1.shuffle_dataset/data/level_2_split/replay/edgar_corpus.jsonl

# Native L2: everything else in level_2/
for f in ../data/final/level_2/*.jsonl; do
  [ "$(basename "$f")" = "edgar_corpus.jsonl" ] && continue
  ln -sf "$(realpath "$f")" \
         "1.shuffle_dataset/data/level_2_split/native/$(basename "$f")"
done
```

Sanity check (8 native, 1 replay):

```bash
ls 1.shuffle_dataset/data/level_2_split/native/
# courtlistener.jsonl  fatf_publications.jsonl  fincen_advisories.jsonl
# fincen_enforcement.jsonl  fincen_federal_register.jsonl  fincen_files.jsonl
# fincen_sar_reviews.jsonl  ofac_guidance.jsonl
ls 1.shuffle_dataset/data/level_2_split/replay/
# edgar_corpus.jsonl
```

## How to run

Pure host environment (no Docker — just Linux + Python).

### Stage 1a — Shuffle native L2 (~30 s, ~10 M tokens over 8 sources)

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v2/1.shuffle_dataset

python3 ../../run-v1/1.shuffle_dataset/shuffle.py \
    --input_dir    data/level_2_split/native \
    --output_dir   data/level_2_native_shuffled \
    --dataset_name level_2_native \
    --extension    .jsonl \
    --memory       8 \
    --seed         42 \
    --nchunks      1 \
    --val_pct      10
```

Produces:
- `data/level_2_native_shuffled/level_2_native.chunk.0.jsonl` (~1,900 records, ~10 M tokens)
- `data/level_2_native_shuffled/level_2_native.val.jsonl` (~190 records = 10% holdout, per-source stratified by the global shuffle)

### Stage 1b — Shuffle EDGAR-replay (~4 min, ~300 M tokens, single source)

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v2/1.shuffle_dataset

python3 ../../run-v1/1.shuffle_dataset/shuffle.py \
    --input_dir    data/level_2_split/replay \
    --output_dir   data/level_2_replay_shuffled \
    --dataset_name level_2_replay \
    --extension    .jsonl \
    --memory       16 \
    --seed         42 \
    --nchunks      4 \
    --val_pct      1
```

Produces:
- `data/level_2_replay_shuffled/level_2_replay.chunk.{0..3}.jsonl` (~6,200 records total, ~300 M tokens)
- `data/level_2_replay_shuffled/level_2_replay.val.jsonl` (~60 records = 1% holdout)

### Why these specific args

| Arg | Native value | Replay value | Reason |
|---|---|---|---|
| `--memory` | 8 | 16 | Native is ~40 MB raw → 8 GB buffer is overkill but safe; replay is ~1.2 GB → bump to 16 GB. |
| `--seed` | 42 | 42 | Same seed as `prepare-data/split_pools.py` for reproducibility. |
| `--nchunks` | 1 | 4 | Native is small enough to fit in one chunk (simplifies the Stage-3 blend to exactly 2 paths). Replay is ~30× bigger and benefits from 4 chunks for Stage-2 tokenization parallelism. |
| `--val_pct` | 10 | 1 | Native val is the **primary post-training metric** (per-source L2 PPL). Replay val is only used to catch L1-register drift during Phase 2 — small holdout is enough. |

## Output layout

```
data/
|-- level_2_split/
|   |-- native/   (8 symlinks to native AML JSONLs)
|   `-- replay/   (1 symlink to edgar_corpus.jsonl)
|-- level_2_native_shuffled/
|   |-- level_2_native.chunk.0.jsonl
|   `-- level_2_native.val.jsonl
`-- level_2_replay_shuffled/
    |-- level_2_replay.chunk.{0..3}.jsonl
    `-- level_2_replay.val.jsonl
```

Stage 2 picks up these chunks, tokenizes them into two separate `.bin/.idx` shard sets (`native_processed_data_*` and `replay_processed_data_*`), and Stage 3's recipe `dataset.paths` weights them 25/75.

## What you should NOT do

- **Don't combine native + replay into a single shuffle.** That collapses the per-side identity required for the Stage-3 weighted blend. The whole point of running `shuffle.py` twice is to keep them separate.
- **Don't use a different `--seed` for native vs replay.** Reproducibility hinges on bit-identical shuffle output across re-runs. Seed `42` matches Phase 1 and `prepare-data/split_pools.py`.
- **Don't drop `--val_pct` for replay below 1.** A val pool of zero records breaks the Stage-3 in-training val pass.
