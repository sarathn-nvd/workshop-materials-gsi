# Stage 1 — Shuffle SFT JSONL globally + split train / val / test

## What this does

1. **Normalize**: extract only the `messages` field from every JSONL record (matches the `ChatDataset` schema in stage 4). Discards `id`, `source`, `metadata`, etc.
2. **Globally shuffle** every record across every input file using `terashuf` (memory-bounded external shuffle).
3. **Split into chunks** (`sft_mixed.chunk.NN.jsonl`) plus dedicated `sft_mixed.val.jsonl` and `sft_mixed.test.jsonl`.

The validation set goes to stage 4 (`validation_dataset` in the recipe). The test set is held back for **post-training evaluation** (whichever harness you use; we don't use it during training to avoid leakage).

The chunk count default is 1 (sufficient for SFT — typically 100K-1M chat records, no need to parallelize tokenization across shards the way CPT does).

## Prerequisites

- `terashuf` is auto-built on first run (clones https://github.com/alexandres/terashuf into `./terashuf/` and `make`s it). Requires `gcc`/`make` available — run **inside the nemo-automodel container** if the host doesn't have them:
  ```bash
  docker run --rm \
    -v /sadata/swaminathanb/gsi-training:/workspace \
    --workdir /workspace/4.run_sft/1.shuffle_dataset \
    nvcr.io/nvidia/nemo-automodel:26.04 \
    python3 shuffle.py --input_dir ../data/raw --dataset_name sft_mixed --nchunks 1 --memory 64
  ```
  Or natively if `gcc` is on the host:
  ```bash
  cd /sadata/swaminathanb/gsi-training/4.run_sft/1.shuffle_dataset
  python3 shuffle.py --input_dir ../data/raw --dataset_name sft_mixed --nchunks 1 --memory 64
  ```

- Input files at `--input_dir` must end in `.jsonl` (or `.jsonl.zst` / `.jsonl.gz` if you pass `--extension`).

## Usage

```bash
cd /sadata/swaminathanb/gsi-training/4.run_sft/1.shuffle_dataset

python3 shuffle.py \
  --input_dir            ../data/raw \
  --dataset_name         sft_mixed \
  --nchunks              1 \
  --val_samples_per_chunk  2000 \
  --test_samples_per_chunk 2000 \
  --memory               64 \
  --seed                 42
```

(If you ran `check_format_fix.py` first, swap `../data/raw` → `../data/fixed`.)

## Output

Files are written under `./data/`:

| File | Purpose |
|---|---|
| `sft_mixed.chunk.00.jsonl` | Training set — 1 chunk by default; passed to stage 3 then to recipe `dataset.path_or_dataset_id` |
| `sft_mixed.val.jsonl` | Validation set (`val_samples_per_chunk` records per chunk) — recipe `validation_dataset.path_or_dataset_id` |
| `sft_mixed.test.jsonl` | Held-out test set — for post-training eval, not loaded by the recipe |

A line-count summary lands in `./shuffle_log.txt`.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--input_dir` | required | Directory of source `.jsonl` files |
| `--dataset_name` | `sft_dataset` | Output filename prefix; use `sft_mixed` to match the rest of this pipeline |
| `--extension` | `.jsonl` | File extension to glob for. Supports `.jsonl.zst` / `.jsonl.gz` |
| `--nchunks` | `1` | Number of training chunks. `1` is fine unless you have >100 GB raw |
| `--val_samples_per_chunk` | `500` | Records pulled out of each chunk into the val file |
| `--test_samples_per_chunk` | `500` | Records pulled out of each chunk into the test file |
| `--memory` | `8` | Memory in GB for `terashuf`. **Bump this if your raw data > 10 GB** |
| `--seed` | `42` | Reproducibility |

## Pitfalls observed in the reference

- **Last message must be `assistant`**. The shuffle stage *doesn't* enforce this. Use `data/check_format_fix.py` upstream, or rely on `3.filter_dataset/rebuild_sft_jsonl.py --ensure-last-assistant` downstream.
- **Unicode line separators (`U+2028`, `U+2029`)** can break ChatDataset's `splitlines()` reader. The shuffle's normalizer doesn't escape these — `3.filter_dataset/rebuild_sft_jsonl.py` does. Always run rebuild after this stage if your raw data has Unicode noise.
- **`terashuf` MEMORY env var matters** — too low causes excessive on-disk spilling. Pass `--memory 64` or higher on the A100 box.
