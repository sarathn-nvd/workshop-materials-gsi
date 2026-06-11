# Stage 2 — Tokenize to Megatron `.bin/.idx` (native + replay independently)

Tokenizes the two shuffled JSONL shard sets from Stage 1 into Megatron-Core indexed `.bin/.idx`, producing **two separate output prefixes** (`native_processed_data_*` and `replay_processed_data_*`) so Stage 3's weighted blend can target each side independently.

Reuses [`../../run-v1/2.prepare_megatroncore_dataset/preprocess_megatron_dataset.py`](../../run-v1/2.prepare_megatroncore_dataset/preprocess_megatron_dataset.py) unchanged — invoked twice with different `--input` glob, `--output-prefix`, and `--workers`.

| | |
|---|---|
| Script | `../../run-v1/2.prepare_megatroncore_dataset/preprocess_megatron_dataset.py` (no local copy needed) |
| Input | `../1.shuffle_dataset/data/level_2_{native,replay}_shuffled/*.chunk.*.jsonl` |
| Output | `data/native_processed_data_0_text_document.{bin,idx}` (~20 MB) + `data/replay_processed_data_{0..3}_text_document.{bin,idx}` (~600 MB) |
| Tokenizer | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` (HuggingFace, requires `HF_TOKEN`) |
| Wall clock | ~1 min native + ~4 min replay |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` |

## Why two tokenization passes

Stage 1 produced two distinct shard sets so the dataloader can apply per-side weights. Tokenization must preserve that separation: a single mixed `.bin/.idx` shard has no knob for the dataloader to weight against. Two `--output-prefix` invocations produce two named shard sets that the recipe's `dataset.paths` blend points at.

See [`../README.md`](../README.md) §"The conceptual gap from run-v1 — routing vs weighting" for the full rationale.

## How to run

Inside the nemo-automodel container so the tokenizer + `nemo_automodel.indexed_dataset` are available.

### Stage 2a — Tokenize native L2 shard (~1 min)

> **`--workdir` is required.** `preprocess_megatron_dataset.py` writes a `corpus_details.log` to its current working directory. If `--workdir` is unset, the container's cwd defaults to `/workspace` (= the bind-mount root), which the unprivileged `--user $(id -u):$(id -g)` cannot write to → `PermissionError: [Errno 13] Permission denied: 'corpus_details.log'`. Pointing `--workdir` at the run-v2 tokenize `data/` dir fixes this and keeps the log next to the outputs.

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v2/2.prepare_megatroncore_dataset

export HF_TOKEN=hf_xxx_your_token

docker run --rm \
  --user "$(id -u):$(id -g)" --group-add 0 \
  --workdir /workspace/3.cpt/2.run_cpt/run-v2/2.prepare_megatroncore_dataset/data \
  -e HOME=/tmp -e HF_HOME=/tmp/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v /data/swami/gsi-training:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 /workspace/3.cpt/2.run_cpt/run-v1/2.prepare_megatroncore_dataset/preprocess_megatron_dataset.py \
    --input  "/workspace/3.cpt/2.run_cpt/run-v2/1.shuffle_dataset/data/level_2_native_shuffled/level_2_native.chunk.*.jsonl" \
    --json-keys text \
    --output-prefix native_processed_data \
    --output-path  /workspace/3.cpt/2.run_cpt/run-v2/2.prepare_megatroncore_dataset/data \
    --pretrained-model-name-or-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --tokenizer-name-or-path        nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --append-eod \
    --trust-remote-code \
    --workers 16
```

Produces: `data/native_processed_data_0_text_document.{bin,idx}` (~20 MB, ~10 M tokens) + `data/native_processed_data_0_stats.json` + `data/corpus_details.log` (informational, can be deleted post-run).

### Stage 2b — Tokenize EDGAR-replay shards (~4 min)

Same command, different `--input` glob, different `--output-prefix`, more workers (4 input files → 4× the parallelism budget). Same `--workdir` fix applies.

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" --group-add 0 \
  --workdir /workspace/3.cpt/2.run_cpt/run-v2/2.prepare_megatroncore_dataset/data \
  -e HOME=/tmp -e HF_HOME=/tmp/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v /data/swami/gsi-training:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 /workspace/3.cpt/2.run_cpt/run-v1/2.prepare_megatroncore_dataset/preprocess_megatron_dataset.py \
    --input  "/workspace/3.cpt/2.run_cpt/run-v2/1.shuffle_dataset/data/level_2_replay_shuffled/level_2_replay.chunk.*.jsonl" \
    --json-keys text \
    --output-prefix replay_processed_data \
    --output-path  /workspace/3.cpt/2.run_cpt/run-v2/2.prepare_megatroncore_dataset/data \
    --pretrained-model-name-or-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --tokenizer-name-or-path        nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --append-eod \
    --trust-remote-code \
    --workers 64
```

Produces: `data/replay_processed_data_{0..3}_text_document.{bin,idx}` (~600 MB total, ~300 M tokens) + 4 `_stats.json` files.

### Why these specific args

| Arg | Native value | Replay value | Reason |
|---|---|---|---|
| `--input` | glob over 1 chunk | glob over 4 chunks | Matches Stage 1's `--nchunks`. |
| `--output-prefix` | `native_processed_data` | `replay_processed_data` | **Critical** — different prefixes prevent collision and let the Stage-3 recipe target each side. |
| `--workers` | 16 | 64 | One worker per input file × workers-per-file. Native has 1 chunk so 16 workers is plenty; replay's 4 chunks each get 16 workers. |
| `--append-eod` | on | on | EOS between docs so packed sequences carry document boundaries. **Mandatory** for both. |

The tokenizer (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`) **must match** the model the recipe loads (cpt-l1-final, which was itself trained from the BF16 base in Phase 1). Token IDs are keyed to this exact tokenizer.

## Verify both sides before Stage 3

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v2/2.prepare_megatroncore_dataset

python3 -c "
import json, glob
for side in ('native', 'replay'):
    total = 0; shards = 0
    for p in sorted(glob.glob(f'data/{side}_processed_data_*_stats.json')):
        s = json.load(open(p)); total += sum(s['per_key']['text']); shards += 1
    print(f'{side:7s}: {shards} shard(s), {total:>14,} tokens')
"
# Expected:
# native : 1 shard(s),    ~10,000,000 tokens
# replay : 4 shard(s),   ~300,000,000 tokens
```

If native is off by > 15% or replay by > 20%, reconcile against [`../../data/final/split_summary.json`](../../data/final/split_summary.json) before proceeding. Mismatched token counts mean wrong blend weights at training time.

## Output layout

```
data/
|-- native_processed_data_0_text_document.bin
|-- native_processed_data_0_text_document.idx
|-- native_processed_data_0_stats.json
|-- replay_processed_data_0_text_document.bin
|-- replay_processed_data_0_text_document.idx
|-- replay_processed_data_0_stats.json
|-- replay_processed_data_1_text_document.{bin,idx}
|-- replay_processed_data_1_stats.json
|-- replay_processed_data_2_text_document.{bin,idx}
|-- replay_processed_data_2_stats.json
|-- replay_processed_data_3_text_document.{bin,idx}
`-- replay_processed_data_3_stats.json
```

Stage 3's `dataset.paths` references each shard explicitly with its weight (see [`../3.run_cpt/recipe_a100sxm-8.yaml`](../3.run_cpt/recipe_a100sxm-8.yaml) `dataset:` block).

## What you should NOT do

- **Don't tokenize native + replay together.** Collapses the per-side identity required for the weighted blend. Use separate `--output-prefix` invocations.
- **Don't change the tokenizer.** It must match the model. Phase 2 loads `cpt-l1-final` (which was tokenized with `Base-BF16` in Phase 1); a different tokenizer here means the model sees garbage IDs.
- **Don't drop `--append-eod`.** Without it, packed sequences silently bleed across documents and the model learns spurious cross-document attention patterns.
- **Don't tokenize the `*.val.jsonl` files.** Those are for the post-training `compare_perplexity.py` evaluator, which loads them fresh as JSONL — they should not enter the training shards.
