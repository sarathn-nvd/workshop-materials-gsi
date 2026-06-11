# Stage 2 — Tokenize to Megatron `.bin/.idx`

Tokenizes the shuffled JSONL chunks from Stage 1 with the FP8 base-model tokenizer, appends an `<eod>` token between every document, and writes Megatron-Core indexed `.bin/.idx` shards that Stage 3's dataloader memory-maps directly.

| | |
|---|---|
| Script | [`preprocess_megatron_dataset.py`](preprocess_megatron_dataset.py) (copied from `../../reference/2.prepare_megatroncore_dataset/`, unchanged) |
| Input | `../1.shuffle_dataset/data/level_1_shuffled/level_1.chunk.*.jsonl` (32 chunks, ~9.25 GB) |
| Output | `data/processed_data_{0..31}_text_document.{bin,idx}` (~4.7 GB packed; 2.31 B uint32 token IDs + index) |
| Tokenizer | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (HuggingFace, requires `HF_TOKEN`) |
| Wall clock | ~10 min on 64 CPU cores |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` |

## Why this stage exists

Two non-negotiable training-time properties motivate offline tokenization:

1. **GPU starvation.** Tokenizing 9 GB of text on the fly with HF tokenizer is ~30 MB/s on CPU; a single H100 doing FP8 matmul at 1.5 K tokens/s/GPU asks for ~2 GB/s of input. Offline tokenization decouples them.
2. **Memory-mapped reads.** Indexed `.bin/.idx` lets the dataloader page in only the sequences it needs via OS mmap with zero-copy. Random access into a 2.3 B-token corpus would be cost-prohibitive otherwise.

The `--append-eod` flag is **critical**: it inserts an EOS token between every document so the model can learn document boundaries when the dataloader packs multiple short documents into a single `seq_length=4096` sequence (which is the common case for our corpus).

## How to run

Inside the nemo-automodel container so the FP8 tokenizer + `nemo_automodel.indexed_dataset` are available.

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/2.prepare_megatroncore_dataset

export HF_TOKEN=hf_xxx_your_token

docker run --rm \
  --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp -e HF_HOME=/tmp/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v /data/swami/gsi-training:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 /workspace/3.cpt/2.run_cpt/run-v1/2.prepare_megatroncore_dataset/preprocess_megatron_dataset.py \
    --input  "/workspace/3.cpt/2.run_cpt/run-v1/1.shuffle_dataset/data/level_1_shuffled/level_1.chunk.*.jsonl" \
    --json-keys text \
    --output-prefix processed_data \
    --output-path  /workspace/3.cpt/2.run_cpt/run-v1/2.prepare_megatroncore_dataset/data \
    --pretrained-model-name-or-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --tokenizer-name-or-path        nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --append-eod \
    --trust-remote-code \
    --workers 64
```

### Why these specific args

| Arg | Value | Reason |
|---|---|---|
| `--input` | glob over 32 chunks | The script's `Partition` divides workers across matched files (`workers // n_files`), so passing all 32 at once gives true cross-file parallelism. |
| `--json-keys` | `text` | Our split_pools output schema is `{text, source, id}` — only `text` is the training input. (Reference used `translation transliteration`.) |
| `--output-prefix` | `processed_data` | Per-file output names become `processed_data_<idx>_text_document.{bin,idx}`. |
| `--output-path` | `data/` | Keeps the script's intermediate `_ss` files and final outputs in one tree. |
| `--pretrained-model-name-or-path` + `--tokenizer-name-or-path` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | The tokenizer that the training model uses. **Both args must be the FP8 model** so token IDs match what the model expects at training time. |
| `--append-eod` | on | Inserts the EOS token (Nemotron-3 uses `<extra_id_1>` as EOD; the script reads `tokenizer.eos_token_id` automatically) at the end of every document, so packed sequences carry document boundaries. **Mandatory for multi-doc packing.** |
| `--trust-remote-code` | on | Required for Nemotron-3's custom tokenizer code. |
| `--workers` | 64 | One worker per CPU core, distributed across 32 input files (`64 // 32 = 2` workers per file inside the script's `Pool`). Tune down if your host has fewer cores. |

The `HF_TOKEN` env is needed because the FP8 model is a gated HuggingFace repo. Set it before invoking; do **not** bake it into the script.

## Output layout

```
data/
├── processed_data_0_text_document.bin     # token IDs (uint32 for 256K vocab)
├── processed_data_0_text_document.idx     # document offsets + lengths
├── processed_data_1_text_document.bin
├── processed_data_1_text_document.idx
├── ...
├── processed_data_31_text_document.bin
├── processed_data_31_text_document.idx
└── processed_data_*_stats.json            # per-file token counts (auto-emitted)
```

Stage 3's recipe globs `processed_data_*_text_document*` to pick all 32 shards as a single dataset. NeMo's `BlendedMegatronDatasetBuilder` will treat them as one logical dataset (no per-shard weights — the shuffle in Stage 1 already balanced per-source mix).

## Verification

After tokenization, sanity-check total token count against the design target (~2.31 B from the partition summary):

```bash
python3 -c "
import json, glob
total = 0
for p in sorted(glob.glob('data/processed_data_*_stats.json')):
    s = json.load(open(p))
    total += sum(s['per_key']['text'])
print(f'Total tokens (text key, all shards): {total:,}')
print(f'Expected from prepare-data/split_summary.json: ~2.31 B')
"
```

Expected: 2.2–2.4 B tokens (the curator's char→token ratio for our corpus is ~4.0; minor variance from EOD tokens and chunk boundaries is normal).

## What you should NOT do

- **Don't change tokenizer mid-run.** The training model's embedding table is keyed to this exact tokenizer. Re-tokenizing with anything else means the model will see garbage IDs.
- **Don't drop `--append-eod`.** Without it, packed sequences silently bleed across documents — the model learns spurious cross-document attention patterns and val PPL goes up.
- **Don't add `--split-sentences`.** That's NLTK-based and adds an unnecessary intermediate `_ss.jsonl` pass. The base model already handles sentence boundaries via its tokenizer.
- **Don't tokenize the val.jsonl into the same output-prefix.** It would land in `data/processed_data_32_text_document.{bin,idx}` and the dataloader would silently include val records in training. The val file is for the post-training `compare_perplexity.py`, which loads it fresh.
