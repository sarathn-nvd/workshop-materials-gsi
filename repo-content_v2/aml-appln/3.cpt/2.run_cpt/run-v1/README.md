# CPT run-v1 — Phase 1 (level_1 broad-register pretrain)

End-to-end Phase 1 continued-pretraining of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` on the partitioned `level_1` corpus produced by [`../prepare-data/`](../prepare-data/).

| | |
|---|---|
| Phase | 1 of 2 (broad financial / regulatory register) |
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |
| Input data | [`../data/final/level_1/`](../data/final/level_1/) — 8 sources, ~9.25 GB JSONL, ~75 K docs, ~2.31 B tokens |
| Output checkpoint | `cpt-l1-final/` (consumed by `run-v2/` Phase 2) |
| Hardware | 8 × H100 NVL (94 GB / GPU) |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` |
| Wall-clock estimate | ~40 hours / ~1.7 days |
| Loss target | per-source val PPL drop ≥ 15% on the L1 holdout vs base; expert load balance in [0.5%, 5%] |

Detailed recipe rationale lives in [`../README.md`](../README.md) §4–5. This document is the **operational runbook** — the three commands you actually execute, in order.

---

## Pipeline overview

```
data/final/level_1/                   ../prepare-data/ produced this
        |
        |  Stage 1: terashuf-based global shuffle + chunking + val extraction
        v
1.shuffle_dataset/data/level_1_shuffled/{custom.chunk.NN.jsonl, custom.val.jsonl}
        |
        |  Stage 2: Megatron .bin/.idx tokenization (FP8 base tokenizer + EOD)
        v
2.prepare_megatroncore_dataset/data/processed_data_*_text_document.{bin,idx}
        |
        |  Stage 3: FSDP2 + EP=8 + FP8 training (recipe_h100nvl-8.yaml)
        v
3.run_cpt/checkpoints/{phase1/, LOWEST_VAL/} -> cpt-l1-final/
```

Each stage has its own README in its subdirectory with the exact invocation. The summary commands below are the happy-path; consult the per-stage READMEs for tuning.

---

## How to run (end-to-end)

### Stage 1 — Shuffle & chunk (~3 min)

Pure stdlib Python + `terashuf` (auto-installed on first run). Runs on the host, no Docker.

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
    --val_pct     10
```

Produces 32 shuffled training chunks + a `level_1.val.jsonl` (~7.5 K records = 10% of the ~75 K-record corpus, pulled off the top of all chunks for held-out perplexity eval). The reference's hardcoded `head -n 10000` per chunk has been replaced by a percentage-based pull because our corpus is much smaller than the reference's — see the Stage 1 README "Validation extraction" section for details.

### Stage 2 — Tokenize to Megatron `.bin/.idx` (~10 min)

Runs inside the nemo-automodel container so the FP8 tokenizer + `nemo_automodel.indexed_dataset` are available.

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/2.prepare_megatroncore_dataset

export HF_TOKEN=hf_xxx_your_token
docker run --rm \
  --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp -e HF_HOME=/workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v /data/swami/gsi-training:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 /workspace/3.cpt/2.run_cpt/run-v1/2.prepare_megatroncore_dataset/preprocess_megatron_dataset.py \
    --input  "/workspace/3.cpt/2.run_cpt/run-v1/1.shuffle_dataset/data/level_1_shuffled/level_1.chunk.*.jsonl" \
    --json-keys text \
    --output-prefix processed_data \
    --output-path  /workspace/3.cpt/2.run_cpt/run-v1/2.prepare_megatroncore_dataset/data \
    --pretrained-model-name-or-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --tokenizer-name-or-path        nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --append-eod \
    --trust-remote-code \
    --workers 64
```

Produces 32 `.bin` + 32 `.idx` shards in `data/`. Append-eod adds the EOD token between every document so the model can learn document boundaries when the dataloader packs multiple docs into a single 4096-token sequence.

### Stage 3 — Train (~40-50 hours; central estimate ~42 h)

Background launch with stdout+stderr piped to `run.log` on the host (the redirect happens host-side, so the path must be the host path).

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/3.run_cpt

export HF_TOKEN=hf_xxx_your_token

nohup docker run --rm --gpus all \
  --name cpt-phase1 \
  --shm-size=64g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" --group-add 0 \
  --workdir /workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt \
  -e HOME=/tmp -e HF_HOME=/workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v /data/swami/gsi-training:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  torchrun --nproc-per-node=8 \
    /workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/pretrain.py \
    --config /workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/recipe_h100nvl-8.yaml \
  > /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/3.run_cpt/run.log 2>&1 &
echo "PID: $!"

# Watch progress
tail -f /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/3.run_cpt/run.log

# Stop cleanly
docker stop cpt-phase1
```

Strongly recommended: run a 30-step smoke test first (copy the recipe with `max_steps: 30, ckpt_every_steps: 999` and the same `--name cpt-phase1-smoke`) to measure real `tps/GPU` before committing to the multi-day run. See [`./3.run_cpt/README.md`](3.run_cpt/README.md) for the throughput → wall-clock table.

### Post-training — Compare perplexity (~30 min)

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/3.run_cpt

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp -e HF_HOME=/workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v /data/swami/gsi-training:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 /workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/compare_perplexity.py \
    --input     /workspace/3.cpt/2.run_cpt/run-v1/1.shuffle_dataset/data/level_1_shuffled/level_1.val.jsonl \
    --base_model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --ft_model   /workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated \
    --batch_size 16 --max_len 1024 --warmup 2 --measure 50 --max_samples 10000
```

Pass criterion (per [`../README.md`](../README.md) §4.5 evaluation gates):
- Per-source val PPL drop ≥ 15% on AML-specific holdout vs base
- MMLU regression ≤ 5 pp
- Per-expert token fraction in [0.5%, 5%]

---

## Differences from `../reference/`

The reference ran on **8 × H200 (140 GB) with BF16** on a translation/transliteration corpus. This run targets **8 × H100 NVL (94 GB) with FP8** on AML/financial register. The deltas are:

| | Reference | run-v1 (this) | Why |
|---|---|---|---|
| Base model | `Nemotron-3-Nano-30B-A3B-BF16` | `Nemotron-3-Nano-30B-A3B-FP8` | Locked decision (`../README.md` §1) |
| `torch_dtype` | `bfloat16` | `auto` (honors FP8 native) | FP8 weights |
| `seq_length` | 2048 | 4096 | Long EDGAR / FinCEN docs (`../README.md` §4.3) |
| `local_batch_size` | 2 | 1 | Memory-driven (94 GB H100 NVL vs 140 GB H200) |
| `ep_size` | 1 | **8** | Critical fit on H100 NVL (`../README.md` §4.1) |
| Peak LR | 5e-5 cosine | **2e-5 WSD** | Conservative for narrow corpus + Phase 2 resume |
| Tokenizer json-keys | `translation transliteration` | `text` | Our schema (`../prepare-data/`) |
| Input data | translation pairs | partitioned `level_1` corpus (`../data/final/level_1/`) | Different task |
| `shuffle.py` val extraction | hardcoded `head -n 10000` per chunk (= 320 K val on a 10 M-doc corpus) | `--val_pct 10` (= ~7.5 K val on our 75 K-doc corpus) | Reference's hardcoded value would consume our entire corpus |
| `max_steps` | 12750 | ~1800 | Sized for 1 epoch over 2.31 B tokens at GBS 256 / seq 4096 |
| `compare_perplexity.py` keys | `translation`, `transliteration` | `text` | Schema match |

Everything else (FSDP2, AdamW betas, gradient clipping, MoE knobs, checkpoint cadence) follows the reference shape with the parameter values from `../README.md` §4.4.

---

## Directory layout

```
run-v1/
├── README.md                                        (this file)
├── 1.shuffle_dataset/
│   ├── README.md
│   ├── shuffle.py                                   (copied from ../reference/, unchanged)
│   ├── terashuf/                                    (auto-cloned + compiled on first run)
│   └── data/level_1_shuffled/                       (Stage 1 output)
│       ├── level_1.chunk.{00..31}.jsonl
│       └── level_1.val.jsonl
├── 2.prepare_megatroncore_dataset/
│   ├── README.md
│   ├── preprocess_megatron_dataset.py               (copied from ../reference/, unchanged)
│   └── data/processed_data_{0..31}_text_document.{bin,idx}    (Stage 2 output)
└── 3.run_cpt/
    ├── README.md
    ├── pretrain.py                                  (copied from ../reference/, unchanged)
    ├── recipe_h100nvl-8.yaml                        (NEW — Phase 1 recipe)
    ├── compare_perplexity.py                        (NEW — adapted for `text` field)
    ├── training.log                                 (Stage 3 stdout/stderr)
    ├── mapping_dir/                                 (NeMo dataloader index cache)
    └── checkpoints/                                 (every 250 steps + LOWEST_VAL)
```
