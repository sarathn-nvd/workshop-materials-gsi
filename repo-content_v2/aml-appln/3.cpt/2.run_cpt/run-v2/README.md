# CPT run-v2 — Phase 2 (level_2 AML-specific domain adaptation)

End-to-end Phase 2 continued-pretraining on the `cpt-l1-final` checkpoint produced by [`../run-v1/`](../run-v1/). Trains on the partitioned `level_2` pool from [`../prepare-data/`](../prepare-data/) — native AML sources **blended** with EDGAR-replay to prevent catastrophic forgetting while avoiding per-token over-memorization.

| | |
|---|---|
| Phase | 2 of 2 (AML-specific domain adaptation) |
| Base checkpoint | `../run-v1/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated/` (= `cpt-l1-final`); on the A100 box this is synced to `<HOST_WORKSPACE>/cpt-l1-final/` |
| Input data | [`../data/final/level_2/`](../data/final/level_2/) — 9 sources (~310 M tokens): ~10 M native AML + ~300 M EDGAR-replay |
| Output checkpoint | `cpt-l2-final/` (consumed by SFT — next pipeline) |
| Hardware | 8 × A100 SXM (80 GB / GPU, NVSwitch full mesh) — same node used for Phase 1 |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` |
| Wall-clock estimate | **~10–14 h** (extrapolated from Phase-1 measurement of ~37 sec/step; Phase 2 halves GBS so per-step time drops to ~18-25 sec, × 2056 steps ≈ 10-14 h) |
| Loss target | per-source L2 val PPL drop ≥ 15% vs `cpt-l1-final`; L1 replay PPL drift ≤ 10% (catastrophic-forgetting guardrail); MMLU drop ≤ 5pp cumulative vs base |

Detailed recipe rationale lives in [`../README.md`](../README.md) §4.5 and §3.3. This document is the **operational runbook** — the exact commands you execute, in order.

---

## The conceptual gap from run-v1 — routing vs weighting

`prepare-data/split_pools.py` did **routing**: which documents live in `level_1/` vs `level_2/` on disk. It did **not** decide how often training samples each source. Those are separate concerns:

| Concept | Controlled by | Value in run-v2 |
|---|---|---|
| **Routing** (which docs live where) | `prepare-data/split_pools.py` | 13% of EDGAR + all native L2 → `level_2/` |
| **Weighting** (how often training samples each source) | Stage 3 dataloader (`dataset.paths` blend weights) | 25% native / 75% replay |

**The natural token ratio inside `data/final/level_2/` is ~3% native AML / ~97% EDGAR-replay** (10 M : 300 M). If Phase 2 used run-v1's pipeline unchanged — one shuffle, one tokenize, one single-glob `dataset.paths`, just `num_epochs` bumped — the dataloader would sample at that natural ratio. The model would spend 97% of its Phase-2 gradient steps on EDGAR-replay and 3% on AML content. **No AML domain shift would land.** Phase 2 would effectively be "more of Phase 1 at a lower LR" — not what we're after.

**The weighted blend at Stage 3 is what promotes native AML from 3% to 25%.** To apply weights, the tokenized shards must exist *separately* per side — a single mixed shard has no knob for the dataloader to weight against. Every pipeline delta between run-v1 and run-v2 exists to make that weight application possible.

So the short answer to "is it just more epochs?" is **no**. The next section enumerates the six items that actually change, in two categories.

---

## Summary of changes vs run-v1 — six items, two categories

### Category A — Mandatory pipeline changes (to enable the weighted blend)

| # | What | Why |
|---|---|---|
| 1 | **Stage 0 (new)**: symlink `data/final/level_2/` into `native/` + `replay/` subdirs | Stage 1's `shuffle.py` globs every file under `--input_dir`; without the split, native and replay would cat together and shuffle into one mixed stream (the routing-vs-weighting trap above). |
| 2 | **Stage 1 runs twice** (native + replay, separate output dirs + `dataset_name`s) | Produces two JSONL chunk sets — one per side of the future blend. |
| 3 | **Stage 2 runs twice** (different `--output-prefix`) | Produces two `.bin/.idx` shard sets (`native_processed_data_*` and `replay_processed_data_*`). These are the two sides the blend weights point at. |
| 4 | **Stage 3 recipe `dataset.paths`** uses weighted-blend syntax, not a single glob | Without weights, Megatron-Core treats all shards equally → natural ~3% AML share → no domain shift. |

### Category B — Training-behavior deltas (Phase 2 is a fine-tune on top of Phase 1)

| # | What | Why |
|---|---|---|
| 5 | **Resume from `cpt-l1-final`** instead of HF base | Phase 2 nudges Phase 1's weights; it doesn't start from scratch. |
| 6 | **Lower LR + cosine + smaller GBS + fewer opt steps** (`lr` 2e-5 → 5e-6, WSD → cosine, GBS 256 → 128, `max_steps` 1800 → 2056) | Standard fine-tune-on-top recipe. A high LR here over-narrows the register and hurts L1 retention. |

**Note on `num_epochs: 2`.** That knob is **decorative under bounded `max_steps` mode** — same behavior as run-v1's `num_epochs: 1` (see `../run-v1/3.run_cpt/recipe_h100nvl-8.yaml` line ~271). The real training-volume lever is `max_steps: 2056`. The "2 epochs" label is a human-readable intent, not what the dataloader reports; Megatron will log `> total number of epochs: ~27` against the 10 M native-L2 shard (= internal replay rate), not `2`.

The "Recipe deltas from Phase 1" section below spells out the exact YAML for Category B. The "Summary of every delta from run-v1" table at the bottom enumerates every individual knob across both categories.

---

## The one thing that makes Phase 2 different from Phase 1

Phase 1 trains on one corpus (L1) for one epoch. Phase 2 trains on **two sources blended at each microbatch**: native L2 (small, AML-specific) and EDGAR-replay (large, L1 anti-forgetting tail). Sizing the blend correctly is what prevents over-memorization.

**The risk.** Native L2 is ~10 M tokens — smaller than the `prepare-data/` design assumed (~40 M). If we naively apply the original `"44"/"56"` blend weights + 4 epochs from [`../README.md`](../README.md) §3.3, each native-L2 token would be replayed ~95 times, and small sources (`fatf_publications`, `fincen_enforcement`, `fincen_files`) would exceed 300×. That's pure memorization territory for a 30 B MoE.

**The fix, in three numbers.** All three are set in Stage 3's recipe — no data-prep changes:

| Knob | Phase 1 value | Phase 2 value | Why |
|---|---|---|---|
| Blend ratio (native L2 : EDGAR-replay) | n/a (single corpus) | **25 : 75** (was planned 44 : 56) | Cuts native-L2 draw from 44% to 25% of the stream |
| `num_epochs` | 1 | **2** (was planned 4) | Halves total native-L2 token demand |
| Per-source `fincen_files` cap | n/a | **keep 8× cap from `../README.md` §3.2** | Its 35 K-token pool would otherwise be replayed ~2000× |

**Net effect**: each native-L2 token is replayed ~25–30× on average (safe for a 30 B MoE); EDGAR-replay tokens get ~3× (comfortable anti-forgetting signal); `fincen_files` stays at 8×. No source memorizes.

The rest of this README is how to actually run that.

---

## Pipeline overview

```
../data/final/level_2/                               ../prepare-data/ produced this
    │  (9 JSONL files: 8 native AML + 1 EDGAR-replay)
    │
    │  Stage 1a/1b: split native vs replay, then shuffle each independently
    │  (two terashuf runs -- NOT one combined like Phase 1, so the blend has
    │   two distinct .bin/.idx shards in Stage 3)
    │   ▼
1.shuffle_dataset/data/
    ├── level_2_native_shuffled/   (chunks 0..N of 8 native AML sources, global-shuffled)
    └── level_2_replay_shuffled/   (chunks 0..M of EDGAR-replay, global-shuffled)
    │
    │  Stage 2a/2b: tokenize each shard-set separately
    │   ▼
2.prepare_megatroncore_dataset/data/
    ├── native_processed_data_*_text_document.{bin,idx}
    └── replay_processed_data_*_text_document.{bin,idx}
    │
    │  Stage 3: blended dataloader (25% native L2 / 75% EDGAR-replay),
    │  resumed from cpt-l1-final, 2 epochs, cosine LR, FSDP2+EP=8+FP8
    │   ▼
3.run_cpt/checkpoints/{step_*/, LOWEST_VAL/}  ->  cpt-l2-final/
```

Key difference from run-v1: **Stages 1 and 2 run twice** (once for native, once for replay), producing two separate shard sets. The weighted blend lives in the Stage-3 recipe `dataset.paths` arg. See [Why two shuffle passes?](#why-two-shuffle-passes) below for the rationale.

> **In-training val vs post-training per-source eval — don't conflate them.** The `validation_dataset` block in the recipe (Stage 3) uses the **same 25/75 blend weights** as training, so in-training val loss is a weighted mixture of native AML + EDGAR-replay. A flat in-training val curve does NOT mean "L2 not improving" — it can also mean "EDGAR-replay dominating the mix stays flat". The actual Phase-2 objective — per-source L2 PPL drop ≥ 15% — is measured **post-training** via `compare_perplexity.py` run on `level_2_native.val.jsonl` (native-only, no replay mixed in). In-training val is a sanity signal; the post-training eval is the pass/fail gate.

---

## Prerequisite — wait for Phase 1 to produce `cpt-l1-final`

Phase 2 cannot start until `../run-v1/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated/` exists and the post-training `compare_perplexity.py` reports per-source L1 PPL drops ≥ 15%. If those gates are not met, do **not** proceed — per [`../README.md`](../README.md) §4.5, fall back to the mid-snapshot at `checkpoints/phase1/step_900/` and re-launch Phase 2 from there.

Verify the checkpoint is present:

```bash
ls /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated/
# should show config.json, model-*.safetensors, tokenizer*.json
```

---

## How to run (end-to-end)

### Stage 1a — Split level_2 into native vs replay (~30 s)

`prepare-data/split_pools.py` already co-located the 8 native L2 files and the 1 EDGAR-replay file in `data/final/level_2/`. The shuffle step wants each shard-set in its own directory so `terashuf` globs only the intended files. Use symlinks so the original partition stays untouched (reproducible + inspectable):

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

# Sanity check
ls 1.shuffle_dataset/data/level_2_split/native/
# expect: courtlistener.jsonl fatf_publications.jsonl fincen_advisories.jsonl
#         fincen_enforcement.jsonl fincen_federal_register.jsonl fincen_files.jsonl
#         fincen_sar_reviews.jsonl ofac_guidance.jsonl
ls 1.shuffle_dataset/data/level_2_split/replay/
# expect: edgar_corpus.jsonl
```

### Stage 1b — Shuffle native + replay independently (~4 min total)

Two separate `shuffle.py` invocations — identical invocation pattern to run-v1, but pointed at the two split subdirs. `shuffle.py` is copied from `../run-v1/1.shuffle_dataset/` unchanged; we don't need a local copy (call it by full path).

**Native L2 shuffle** (~10 M tokens over 8 sources; small, 1 chunk is fine):

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

**EDGAR-replay shuffle** (~300 M tokens, single source, 4 chunks for tokenization parallelism):

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
- `data/level_2_replay_shuffled/level_2_replay.chunk.{0,1,2,3}.jsonl` (~6,200 records total, ~300 M tokens)
- `data/level_2_replay_shuffled/level_2_replay.val.jsonl` (~60 records = 1% holdout — this side is the forgetting guardrail, not a primary metric, so a small holdout is enough)

**Why the different `--nchunks` and `--val_pct`?** Native L2 is small enough to fit in one chunk (simplifies the Stage-3 blend to exactly 2 paths). Replay is ~30× bigger and benefits from 4 chunks for Stage-2 tokenization parallelism. Replay `val_pct=1` because the replay val is only used to catch L1-register drift during Phase 2 — we don't need a large statistical sample, we just need trend.

### Stage 2a — Tokenize native L2 shard (~1 min)

Runs inside the nemo-automodel container. Note the **different `--output-prefix`** (`native_processed_data`) so this doesn't collide with the replay tokenize step that follows.

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/run-v2/2.prepare_megatroncore_dataset

export HF_TOKEN=hf_xxx_your_token

docker run --rm \
  --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp -e HF_HOME=/workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/hf_cache -e HF_TOKEN=$HF_TOKEN \
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

Produces: `data/native_processed_data_0_text_document.{bin,idx}` (~20 MB, ~10 M tokens).

### Stage 2b — Tokenize EDGAR-replay shards (~4 min)

Same command, different `--input` glob, different `--output-prefix`:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp -e HF_HOME=/workspace/3.cpt/2.run_cpt/run-v1/3.run_cpt/hf_cache -e HF_TOKEN=$HF_TOKEN \
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

Produces: `data/replay_processed_data_{0..3}_text_document.{bin,idx}` (~600 MB total, ~300 M tokens).

**Verify both sides before Stage 3:**

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

If native is off by > 15% or replay by > 20%, go back to [`../prepare-data/split_summary.json`](../prepare-data/split_summary.json) and reconcile against the chars × (chars/tok ratio from `../1.data_curation/summary.json`) expected values. Don't proceed to training with mismatched token counts.

### Stage 3 — Phase 2 training (~10–14 h on A100 SXM)

**Wall-clock estimate.** Phase 1 measured **~37 sec/step → ~18.5 h** on this same A100 SXM node at GBS=256 / max_steps=1800 (see `../run-v1/3.run_cpt/A100_SXM_OPTIMIZATION.md`). Phase 2 halves GBS (128) and tops up max_steps (2056 vs 1800), so per-step time roughly halves to ~18-25 sec:

| Phase | Per-step time | Total steps | Wall clock |
|---|---:|---:|---:|
| Phase 1 (A100 SXM, measured) | ~37 sec | 1800 | **~18.5 h** |
| Phase 2 (A100 SXM, projected) | ~18-25 sec | 2056 | **~10-14 h** |

The per-step time should drop because MoE all-to-all traffic and per-micro compute both scale with GBS. If Phase 2 step time exceeds ~30 sec, something is off (likely the blend mis-weighting causing extra dataset rebuilds) — stop and investigate before burning the full run.

**Checkpoint disk budget.** `ckpt_every_steps: 100` × `max_steps: 2056` = ~20 checkpoints × ~60 GB each = **~1.2 TB**. Ensure the A100 host has that much free space at `<HOST_WORKSPACE>/run-v2/3.run_cpt/checkpoints/` before launch. The `LOWEST_VAL/` symlink is additive — it's a hardlink target, not a copy, so it costs almost no extra disk.

**Resume from crash.** Phase 2 is short enough on A100 SXM that mid-run SSH disconnects are unlikely to matter, but the recipe is detached-launch-friendly anyway. NeMo auto-detects the newest checkpoint in `checkpoint.checkpoint_dir` on startup — to resume after `docker stop cpt-phase2 && docker rm cpt-phase2`, just re-run the launch command from the recipe header.

**Prerequisites on the A100 box** (see `recipe_a100sxm-8.yaml` header for the full list):

1. SSH in, set `HOST_WORKSPACE` to the parent directory containing both `run-v2/` and `cpt-l1-final/`:
   ```bash
   ssh swaminathanb@exp-blr-dgxa100-02
   cd /sadata/swaminathanb/gsi-training-cpt
   export HOST_WORKSPACE=/sadata/swaminathanb/gsi-training-cpt
   ```
2. Sync the Phase-1 LOWEST_VAL checkpoint as `<HOST_WORKSPACE>/cpt-l1-final/`:
   ```bash
   # On the BASE host (where run-v1 lives):
   rsync -avh --progress \
     /data/swami/gsi-training/3.cpt/2.run_cpt/run-v1/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated/ \
     swaminathanb@exp-blr-dgxa100-02:/sadata/swaminathanb/gsi-training-cpt/cpt-l1-final/
   ```
3. Sync the `run-v2/` tree (recipe, dataset, scripts) to `<HOST_WORKSPACE>/run-v2/`:
   ```bash
   rsync -avh --progress \
     /data/swami/gsi-training/3.cpt/2.run_cpt/run-v2/ \
     swaminathanb@exp-blr-dgxa100-02:/sadata/swaminathanb/gsi-training-cpt/run-v2/
   ```
4. `docker login nvcr.io` and `export HF_TOKEN=hf_xxx_your_token`.

The Phase-2 recipe `3.run_cpt/recipe_a100sxm-8.yaml` is already populated with all four deltas (no hand-editing needed). Launch:

```bash
docker run -d --gpus all \
  --name cpt-phase2 \
  --shm-size=64g --ipc=host \
  --ulimit memlock=-1 \
  --workdir /workspace/run-v2/3.run_cpt \
  -e HOME=/tmp \
  -e HF_HOME=/workspace/run-v2/3.run_cpt/hf_cache \
  -e HF_TOKEN=$HF_TOKEN \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $HOST_WORKSPACE:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  torchrun --nproc-per-node=8 \
    /workspace/run-v2/3.run_cpt/pretrain.py \
    --config /workspace/run-v2/3.run_cpt/recipe_a100sxm-8.yaml

# Stream container stdout/stderr to a file (run from $HOST_WORKSPACE):
nohup docker logs -f cpt-phase2 > run-v2/3.run_cpt/run.log 2>&1 &
disown

# Watch progress (filtered)
tail -f $HOST_WORKSPACE/run-v2/3.run_cpt/run.log \
  | grep -E 'step|loss|val|ppl|expert|total number of epochs'

# Stop cleanly
docker stop cpt-phase2 && docker rm cpt-phase2
```

**Critical early-run log line to watch for**: on first init, Megatron prints `> total number of epochs: N` per blended shard. Expected values are:

| Shard | Expected `N` (internal epochs) | If actually reports |
|---|---:|---|
| `native_processed_data` | ~27 | far > 30 → blend weights too high; kill + tighten |
| `replay_processed_data_*` | ~3 | far > 5 → max_steps too large OR replay share too high |

If either exceeds the expected range by 2×, stop and re-check the recipe before burning ~48 h of compute.

### Post-training — Per-source eval + forgetting check (~45 min)

Runs on the A100 box (or any GPU host). Two evals — L2 val (primary Phase-2 metric) and L1 val (forgetting guardrail).

```bash
# L2 per-source PPL — the primary Phase-2 metric
# (compares Phase-2 LOWEST_VAL against the Phase-1 base = cpt-l1-final)
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp -e HF_HOME=/workspace/run-v2/3.run_cpt/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v $HOST_WORKSPACE:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 /workspace/run-v2/3.run_cpt/compare_perplexity.py \
    --input     /workspace/run-v2/1.shuffle_dataset/data/level_2_native_shuffled/level_2_native.val.jsonl \
    --base_model /workspace/cpt-l1-final \
    --ft_model   /workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated \
    --batch_size 16 --max_len 1024 --warmup 2 --measure 50 --max_samples 5000 --use_bf16

# L1 forgetting guardrail — Phase-2 ckpt vs Phase-1 ckpt on L1 val
# Requires the L1 val.jsonl synced to the A100 box at <HOST_WORKSPACE>/level_1.val.jsonl,
# OR run this eval on the BASE host where run-v1 is fully populated.
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp -e HF_HOME=/workspace/run-v2/3.run_cpt/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v $HOST_WORKSPACE:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 /workspace/run-v2/3.run_cpt/compare_perplexity.py \
    --input     /workspace/level_1.val.jsonl \
    --base_model /workspace/cpt-l1-final \
    --ft_model   /workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated \
    --batch_size 16 --max_len 1024 --warmup 2 --measure 50 --max_samples 10000 --use_bf16
```

**Pass criteria** (per [`../README.md`](../README.md) §4.5):

| Eval | Pass | Fail action |
|---|---|---|
| L2 per-source PPL drop vs Phase-1 checkpoint | ≥ 15% on OFAC / FinCEN cluster; ≥ 5% on FATF / CourtListener | Raise L2 share to 30% and re-run Phase 2 with 3 epochs |
| L1 per-source PPL drift vs Phase-1 checkpoint | ≤ 10% increase on any source | Raise `replay` blend share from 75% to 85% and re-run |
| MMLU sample (1K Qs) | Total drop vs base model ≤ 5pp | Restore `checkpoints/step_1000/` (Phase-2 midpoint at `ckpt_every_steps: 100` × 2056 steps) and halve `lr` (5e-6 → 2.5e-6) in a re-run |
| MoE per-expert load | Every expert in [0.5%, 5%] of tokens | Raise `load_balancing_loss_coeff` 1e-2 → 5e-2 in re-run |

If all four pass, rename the Phase-2 checkpoint as the final deliverable. On the A100 box:

```bash
cp -r $HOST_WORKSPACE/run-v2/3.run_cpt/checkpoints/LOWEST_VAL \
      $HOST_WORKSPACE/cpt-l2-final
```

Then sync `cpt-l2-final/` back to the BASE host (alongside `cpt-l1-final/`) for the SFT pipeline:

```bash
# On the BASE host:
rsync -avh --progress \
  swaminathanb@exp-blr-dgxa100-02:/sadata/swaminathanb/gsi-training-cpt/cpt-l2-final/ \
  /data/swami/gsi-training/3.cpt/2.run_cpt/cpt-l2-final/
```

That directory is the input to SFT (next pipeline, `4.sft/...`).

---

## Recipe deltas from Phase 1 (already applied in `3.run_cpt/recipe_a100sxm-8.yaml`)

Phase 2's recipe was created by copying `../run-v1/3.run_cpt/recipe_a100sxm-8.yaml` and applying these four deltas. Everything else (FSDP2, EP=8, BF16 dispatcher, activation checkpointing, dist_env, rng, checkpoint dir schema) stays identical to Phase 1 — same hardware, same model, same fit. The block below documents what changed and why.

### Delta 1 — Model: resume from Phase-1 checkpoint (`cpt-l1-final`)

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  # CHANGED: resume from cpt-l1-final (synced to <HOST_WORKSPACE>/cpt-l1-final/
  # on the A100 box) instead of the HF base model.
  pretrained_model_name_or_path: /workspace/cpt-l1-final
  trust_remote_code: true
  torch_dtype: auto
  use_mamba_kernels: true
  backend:
    # unchanged from Phase 1
    _target_: nemo_automodel.components.models.common.BackendConfig
    linear: te
    rms_norm: torch_fp32
    experts: torch_mm
    dispatcher: torch
    enable_hf_state_dict_adapter: true
    enable_fsdp_optimizations: true
```

### Delta 2 — Dataset: weighted blend (the overfitting fix)

This is the heart of Phase 2. Replace Phase 1's single-glob `paths:` with an interleaved `[weight, path, ...]` list.

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining
  # CHANGED: 2-side weighted blend. Megatron normalizes weights to 1.0, so each
  # replay shard takes 18.75 (4 * 18.75 = 75) to give exactly 25% native / 75% replay.
  paths:
    - "25"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/native_processed_data_0_text_document
    - "18.75"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/replay_processed_data_0_text_document
    - "18.75"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/replay_processed_data_1_text_document
    - "18.75"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/replay_processed_data_2_text_document
    - "18.75"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/replay_processed_data_3_text_document
  index_mapping_dir: /workspace/run-v2/3.run_cpt/mapping_dir
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
    trust_remote_code: true
  seq_length: 4096
  split: "0.99, 0.01, 0.00"
  splits_to_build: "train"
```

> **Weight-arithmetic gotcha.** Naively writing `"25", native, "75", replay_0, "75", replay_1, ..., "75", replay_3` gives `25 + 4*75 = 325` total, which Megatron normalizes to `25/325 ≈ 7.7%` native and `75/325 ≈ 23.1%` **per replay shard** (× 4 = 92.3% replay total). That's not 25/75. The fix above splits the 75 evenly across the 4 replay shards (`4 × 18.75 = 75`) so the total `25 + 75 = 100` normalizes to exactly the intended ratio. If your replay tokenize step produced a different number of shards, redistribute 75 evenly across them.

Validation dataset mirrors the training blend byte-for-byte — same weights, same paths, same order. If you diverge, in-training val loss becomes uninterpretable.

```yaml
validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining
  paths:
    - "25"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/native_processed_data_0_text_document
    - "18.75"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/replay_processed_data_0_text_document
    - "18.75"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/replay_processed_data_1_text_document
    - "18.75"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/replay_processed_data_2_text_document
    - "18.75"
    - /workspace/run-v2/2.prepare_megatroncore_dataset/data/replay_processed_data_3_text_document
  index_mapping_dir: /workspace/run-v2/3.run_cpt/mapping_dir
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
    trust_remote_code: true
  seq_length: 4096
  split: "0.99, 0.01, 0.00"
  splits_to_build: "validation"
  num_val_samples: 2048
```

Remember: **in-training val loss here is the weighted mix, not per-source L2 PPL.** Per-source L2 PPL is measured post-training via `compare_perplexity.py` (see Stage-3 Post-training section). Don't read too much into a flat in-training val curve; the signal you actually care about lives in the post-training eval.

### Delta 3 — Step scheduler: halve total training, smaller GBS

```yaml
step_scheduler:
  # CHANGED: halved global_batch_size and topped up max_steps from Phase 1.
  #
  # Token math:
  #   tokens / opt step = global_batch_size * seq_length = 128 * 4096 = 524,288
  #   Total training tokens = max_steps * tokens_per_step = 2056 * 524 K = 1.08 B
  #   With 25/75 blend:
  #     native L2 drawn = 1.08 B * 25% = 270 M from 10 M pool = ~27x replay [safe]
  #     replay drawn    = 1.08 B * 75% = 810 M from 300 M pool = ~2.7x replay [normal]
  #   Epochs over native-L2 side ≈ 27 (what Megatron will actually log);
  #   "num_epochs: 2" below is a human-readable intent, NOT what the dataloader reports.
  global_batch_size: 128            # was 256 in Phase 1 -- smaller corpus, more opt steps / epoch
  # 80 GB ceiling -- Phase 1 measured ~67 GiB peak at LBS=2 (steady state).
  # LBS=4 estimated ~78-82 GiB which is right at the OOM edge during val/ckpt
  # spikes. Stay at 2; the GBS halving already cuts wall clock in half.
  local_batch_size: 2               # unchanged from Phase 1 A100 recipe
  ckpt_every_steps: 100
  val_every_steps: 100              # was 190 in Phase 1 -- smaller run, more frequent val is cheap
  num_epochs: 2                     # DECORATIVE under bounded max_steps mode (same as run-v1's
                                    # num_epochs: 1). The real training-volume lever is max_steps.
                                    # Megatron will log "> total number of epochs: ~27" for the
                                    # 10M native-L2 shard -- that's the internal replay rate,
                                    # not a recipe bug.
  max_steps: 2056                   # ~2x per-epoch-worth of steps (540M tokens / 524K ≈ 1028 steps/epoch)
```

### Delta 4 — LR + schedule: lower peak, cosine

```yaml
optimizer:
  _target_: torch.optim.AdamW
  betas: [0.9, 0.95]
  # CHANGED: quarter of Phase 1 peak
  # Phase 1 already moved the model toward the financial register; Phase 2 fine-tunes
  # on top. Higher LR here over-narrows and hurts L1 retention.
  lr: 5.0e-6
  weight_decay: 0.1
  eps: 1.0e-8

lr_scheduler:
  # CHANGED: cosine (was WSD in Phase 1)
  # Phase 1 finished at min_lr=2e-6 (WSD endpoint). Phase 2 resumes with a short
  # warmup back up to 5e-6, then cosine-decays to 5e-7 over the rest of the run.
  lr_decay_style: cosine
  lr_warmup_steps: 50               # was 100 in Phase 1 -- just a short ramp from 2e-6 to 5e-6
  lr_decay_steps: 2000              # most of max_steps=2056 decays to min_lr
  min_lr: 5.0e-7                    # peak / 10
```

Everything else in the recipe (`fp8:` (disabled on A100), `distributed:`, `loss_fn:`, `dataloader:`, `validation_dataloader:`, `dist_env:`, `rng:`, `checkpoint:`) stays **exactly as Phase 1's A100 recipe** — only the `checkpoint.checkpoint_dir` path changes from `run-v1` to `run-v2`:

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: /workspace/run-v2/3.run_cpt/checkpoints/
  model_save_format: safetensors
  save_consolidated: true
```

---

## Why two shuffle passes?

Phase 1's `shuffle.py` globs every file in `--input_dir` and cats them together before terashuf — this is correct for a single-corpus training run (L1), but it **loses per-source identity**. For Phase 2 we need two distinct `.bin/.idx` shard sets so Megatron's `BlendedDataset` can sample them at 25/75 ratio; that's only possible if tokenization sees them as separate inputs.

Three alternatives were considered:

1. **Modify `shuffle.py` to preserve per-source grouping.** ~20 lines of change (group by `rec["source"]` after scan, emit N chunks per source). Enables the full 9-shard blend spec from [`../prepare-data/README.md`](../prepare-data/README.md) §3.2 with square-root-tempered per-source weights. Over-engineered for our 10 M native-L2 volume — each source already gets a fair share of the 27× average replay without tempering.
2. **Do per-source upsampling at the JSONL level before shuffle** (duplicate documents in small-source JSONLs to match target upsample factor, then treat everything as one corpus). Works with existing scripts but wastes disk (10 M tokens × ~6× tempering = 60 M on-disk), and bakes the upsample factors into the tokenized shards so tuning them post-tokenize is impossible.
3. **Two-shard blend (chosen).** One global shuffle across the 8 native L2 sources, one across replay. Two `.bin/.idx` sets. Native-L2 sources get **natural-share weighting** (OFAC dominates at ~42%, FATF gets ~2%). For our volume that's acceptable — after 27× average replay every source sees enough passes to contribute meaningfully. If post-run per-source L2 PPL shows FATF / courtlistener under-trained relative to OFAC, upgrade to alternative 1 in a re-run.

---

## Directory layout

```
run-v2/
├── README.md                                        (this file)
├── 1.shuffle_dataset/
│   └── data/
│       ├── level_2_split/                           (Stage 1a -- symlinks into ../data/final/)
│       │   ├── native/   (8 symlinks to native AML JSONLs)
│       │   └── replay/   (1 symlink to edgar_corpus.jsonl)
│       ├── level_2_native_shuffled/                 (Stage 1b output -- native)
│       │   ├── level_2_native.chunk.0.jsonl
│       │   └── level_2_native.val.jsonl
│       └── level_2_replay_shuffled/                 (Stage 1b output -- replay)
│           ├── level_2_replay.chunk.{0..3}.jsonl
│           └── level_2_replay.val.jsonl
├── 2.prepare_megatroncore_dataset/
│   └── data/                                        (Stage 2a+2b output)
│       ├── native_processed_data_0_text_document.{bin,idx}
│       ├── native_processed_data_0_stats.json
│       ├── replay_processed_data_{0..3}_text_document.{bin,idx}
│       └── replay_processed_data_{0..3}_stats.json
└── 3.run_cpt/
    ├── pretrain.py                                  (copied from run-v1/, unchanged)
    ├── compare_perplexity.py                        (copied from run-v1/, unchanged)
    ├── recipe_a100sxm-8.yaml                        (NEW -- Phase 2 recipe, 4 deltas from run-v1's A100 recipe)
    ├── run.log                                      (Stage 3 stdout/stderr; created at launch)
    ├── mapping_dir/                                 (NeMo dataloader index cache; populated at launch)
    └── checkpoints/                                 (every 100 steps + LOWEST_VAL -> cpt-l2-final)
```

---

## Summary of every delta from run-v1

For anyone reading this who just wants the diff:

| Stage | run-v1 (Phase 1) | run-v2 (Phase 2) |
|---|---|---|
| Input data | 1 directory (`data/final/level_1/`, 8 files mixed) | 2 subdirs (`data/final/level_2/{native, replay}`, split by Stage 1a) |
| `shuffle.py` invocations | 1 | **2** (native + replay, separate output dirs + dataset_names) |
| `shuffle.py` `--nchunks` | 32 | 1 (native) + 4 (replay) |
| `shuffle.py` `--val_pct` | 10 | 10 (native) + 1 (replay) |
| `preprocess_megatron_dataset.py` invocations | 1 | **2** (different `--output-prefix`: `native_processed_data` vs `replay_processed_data`) |
| Recipe filename | `recipe_a100sxm-8.yaml` | `recipe_a100sxm-8.yaml` (same name, Phase-2 contents) |
| Recipe `model.pretrained_model_name_or_path` | HF Base-BF16 | **`/workspace/cpt-l1-final`** (synced from run-v1 LOWEST_VAL) |
| Recipe `dataset.paths` | single glob | **weighted blend `[25, native, 18.75×4, replay]`** |
| Recipe `step_scheduler.global_batch_size` | 256 | **128** |
| Recipe `step_scheduler.local_batch_size` | 2 | 2 (unchanged — A100 80 GB ceiling) |
| Recipe `step_scheduler.num_epochs` | 1 | **2** (decorative; max_steps is the real lever) |
| Recipe `step_scheduler.max_steps` | 1800 | **2056** |
| Recipe `step_scheduler.val_every_steps` | 190 | **100** |
| Recipe `optimizer.lr` | 2e-5 | **5e-6** |
| Recipe `lr_scheduler.lr_decay_style` | WSD | **cosine** |
| Recipe `lr_scheduler.lr_warmup_steps` | 100 | **50** |
| Recipe `lr_scheduler.lr_decay_steps` | 1800 | **2000** |
| Recipe `lr_scheduler.wsd_decay_steps` | 430 | (removed — cosine doesn't use this) |
| Recipe `lr_scheduler.min_lr` | 2e-6 | **5e-7** |
| Recipe `checkpoint.checkpoint_dir` | `/workspace/3.run_cpt/checkpoints/` | `/workspace/run-v2/3.run_cpt/checkpoints/` |
| Container bind-mount | `-v $HOST_WORKSPACE:/workspace` (was repo root) | `-v $HOST_WORKSPACE:/workspace` (parent of `run-v2/` and `cpt-l1-final/`) |
| Post-training eval runs | 1 (L1 val vs base) | **2** (L2 val vs Phase-1 ckpt + L1 val vs Phase-1 ckpt for forgetting check) |

Everything not on this list is literally copy-paste from `run-v1/3.run_cpt/recipe_a100sxm-8.yaml`. FP8 stays disabled, dispatcher stays `torch`, EP stays 8, AC stays on, FSDP2 stays unchanged.
