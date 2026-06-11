# CPT Training — Nemotron-3-Nano-30B-A3B on 8×H100 NVL

Continued Pre-Training (CPT) plan for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` on the curated AML/financial corpus produced by [`../1.data_curation/`](../1.data_curation/).

This document covers three things, in order:

1. **The 2-level training strategy** — what runs, in what sequence, and why.
2. **How to split the L1 and L2 data** — the disjoint pool partition produced once at data-prep time.
3. **Recommended training parameters** — parallelism, sequence/batch sizing, optimizer, schedule, and the per-phase deltas.

Companion docs:
- [`../../training_strategy.md`](../../training_strategy.md) — overall pipeline (CPT → SFT → RL)
- [`../../2.data_processing/summary.json`](../../2.data_processing/summary.json) — token/record stats per source
- [`../reference/`](../reference/) — prior-work `nemo-automodel` recipe used as the empirical baseline

---

## Table of Contents

1. [Inputs at a Glance](#1-inputs-at-a-glance)
2. [Two-Level Training Strategy](#2-two-level-training-strategy)
3. [Data Split Strategy](#3-data-split-strategy)
   - 3.1 [L1 — three-pool partition](#31-l1--three-pool-partition)
   - 3.2 [L2 — train + holdout split](#32-l2--train--holdout-split)
   - 3.3 [Phase 2 batch composition](#33-phase-2-batch-composition)
   - 3.4 [Pre-prep filters that apply to both](#34-pre-prep-filters-that-apply-to-both)
4. [Recommended Training Parameters](#4-recommended-training-parameters)
   - 4.1 [Hardware fit vs the reference recipe](#41-hardware-fit-vs-the-reference-recipe)
   - 4.2 [Parallelism](#42-parallelism)
   - 4.3 [Sequence length, batch sizing, throughput](#43-sequence-length-batch-sizing-throughput)
   - 4.4 [Phase 1 hyperparameters](#44-phase-1-hyperparameters)
   - 4.5 [Phase 2 hyperparameters (deltas only)](#45-phase-2-hyperparameters-deltas-only)
   - 4.6 [MoE-specific knobs](#46-moe-specific-knobs)
   - 4.7 [Combined budget summary](#47-combined-budget-summary)
5. [How Training Happens on Long-Sequence Records](#5-how-training-happens-on-long-sequence-records)
   - [5.1 The packing algorithm in one paragraph](#5-1)
   - [5.2 Worked example — a 32,000-token EDGAR 10-K at seq_length=4096](#5-2)
   - [5.3 Where each guarantee is enforced](#5-3)
   - [5.4 Three NeMo defaults we inherit but never set explicitly](#5-4)
   - [5.5 Information-loss accounting](#5-5)
6. [Concrete Phase 1 Recipe (drop-in YAML)](#6-concrete-phase-1-recipe-drop-in-yaml)
7. [Risks to Pre-mitigate](#7-risks-to-pre-mitigate)
8. [Open Questions](#8-open-questions)
9. [Directory Layout](#9-directory-layout)

---

## 1. Inputs at a Glance

| | |
|---|---|
| Hardware | 8 × NVIDIA H100 NVL (94 GB / GPU; 752 GB total VRAM) |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` |
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |
| Architecture | Hybrid Mamba-2 + Transformer MoE — 31.6B total / 3.2B active per token; 128 routed + 1 shared experts, 6 active |
| Precision | FP8 (Transformer Engine); FP32 master weights + FP32 AdamW moments |
| L1 raw tokens | **3,157 M** across 8 sources (`edgar_corpus` 89.4%, rest 10.6%) |
| L2 raw tokens | **40.3 M** across 9 sources (`ofac_guidance` 45%, FinCEN 49%, FATF + courtlistener 6%) |
| Wall-clock budget for CPT | ~6 days (leaves ~8 days for SFT + RL inside the 14-day total) |

**Precision note.** Training is performed end-to-end in FP8 using NVIDIA Transformer Engine kernels (matmul inputs/outputs in FP8 E4M3, accumulation in FP32, master weights and AdamW moments in FP32). FP8 attention/MLP compute roughly halves activation memory and improves H100 throughput; the FP32 master weights preserve update precision. The released `-FP8` checkpoint is loaded natively — no offline conversion step.

---

## 2. Two-Level Training Strategy

Two sequential CPT runs. **No third run.** The second run inherits the first's checkpoint and adds the AML-specific signal while protecting against catastrophic forgetting via L1 replay.

```
                ┌────────────────────────────┐
                │  Base: Nemotron-3-Nano-30B │
                │  -A3B-FP8                  │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │  Phase 1 (run-v1)          │
                │  L1 only, 1 epoch          │
                │  ~1.82 B tokens trained    │
                │  Goal: broad financial /   │
                │  regulatory register       │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │  cpt-l1-final/             │
                │  + mid-snapshot (rollback) │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │  Phase 2 (run-v2)          │
                │  L2 upsampled + L1 replay  │
                │  3–4 epochs, interleaved   │
                │  ~2.16 B tokens trained    │
                │  Goal: AML-specific        │
                │  domain specialization     │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │  cpt-l2-final/             │
                │  → consumed by SFT (next)  │
                └────────────────────────────┘
```

**Why two phases instead of one mixed run:**
- Phase 1 lets the model absorb the 3 B-token broad register without being pulled toward AML-specific minutiae too early. With LR at peak, the broad signal lands cleanly.
- Phase 2 lowers the LR and concentrates compute on the small (40 M) AML corpus, which is too small to dominate gradients in a one-shot mix. Upsampling + replay achieves the right effective ratio without throwing away L1's diversity.
- The mid-Phase-1 checkpoint is the rollback insurance against Phase 2 over-narrowing the model (per `training_strategy.md` §10 item 2).

**Curriculum direction:** broad → narrow. The reverse (narrow → broad) is well-known to wash out the specialized signal.

---

## 3. Data Split Strategy

Both layers get partitioned **once**, at data-prep time, before any training starts. After the split, every training run consumes only the pool(s) it owns. **No token appears in two pools.** This is the contract the data-prep script (TBD) must enforce.

### 3.1 L1 — three-pool partition

The 3.16 B L1 corpus is routed deterministically (hash on `doc_id`) into three disjoint pools. EDGAR is the only source large enough to spare meaningful volume for the replay pool; the seven smaller sources go almost entirely into Phase-1 train (we need every token of them).

| Source | Total | → `L1-train-phase1` | → `L1-replay-pool` | → `L1-holdout-eval` |
|---|---:|---:|---:|---:|
| edgar_corpus | 2,821 M | 1,500 M (53%) | 1,250 M (44%) | 71 M (2.5%) |
| pile_of_law_oig | 251 M | 240 M (95.6%) | 0 | 11 M (4.4%) |
| pile_of_law_federal_register | 31 M | 30 M | 0 | 1.4 M |
| pile_of_law_sec | 31 M | 30 M | 0 | 1.4 M |
| pile_of_law_cfr | 11 M | 10.5 M | 0 | 0.5 M |
| uscode_house | 5.9 M | 5.6 M | 0 | 0.3 M |
| pile_of_law_doj_guidance | 5.6 M | 5.4 M | 0 | 0.2 M |
| pile_of_law_uscode | 1.3 M | 1.25 M | 0 | 0.05 M |
| **TOTAL** | **3,157 M** | **~1,823 M** | **~1,250 M** | **~85 M** |

**Effective Phase 1 source mix** (1.82 B tokens consumed in 1 epoch):

| Source | Share of Phase 1 |
|---|---:|
| edgar_corpus | ~82.3% |
| pile_of_law_oig | ~13.2% |
| pile_of_law_federal_register | ~1.6% |
| pile_of_law_sec | ~1.6% |
| pile_of_law_cfr | ~0.6% |
| uscode_house, doj_guidance, uscode | ~0.7% combined |

EDGAR is still dominant — that's intentional given its scale and fit for "broad financial register." The cap brings it from 89% (raw) down to 82% (post-cap), which restores enough room for the 7 supporting sources to actually move the model. If post-Phase-1 per-source val PPL shows EDGAR over-fitting (PPL drop >50%) while the 7 supporting sources barely move (<10% drop), tighten the EDGAR cap to 1.0 B in a re-run.

**Per-doc 5% cap** (from the planning todo): mostly affects `uscode_house` — 3 documents totalling 5.9 M, so 5% of source = ~295 K per doc. Each doc is chunked and capped at ~295 K contiguous tokens; the dropped tail goes to `L1-holdout-eval`. For EDGAR (avg doc 45 K, source 2.82 B), 5% = 141 M — non-binding. Not enforced for the smaller sources where it would be a no-op.

### 3.2 L2 — train + holdout split

L2 is small (40 M). All of it goes into training; only a thin holdout is reserved for per-source eval.

| Source | Total | → `L2-train-pool` | → `L2-holdout-eval` |
|---|---:|---:|---:|
| ofac_guidance | 18,137 K | 17,230 K (95%) | 907 K |
| fincen_federal_register | 5,717 K | 5,431 K | 286 K |
| fincen_sar_reviews | 5,080 K | 4,826 K | 254 K |
| fincen_advisories | 4,641 K | 4,409 K | 232 K |
| fincen_enforcement | 4,261 K | 4,048 K | 213 K |
| ofac_enforcement | (in `ofac_guidance`) | — | — |
| courtlistener | 1,310 K | 1,244 K | 66 K |
| fatf_publications | 1,055 K | 1,002 K | 53 K |
| fincen_files | 107 K | 102 K | 5 K |
| enterprise_financial_crime | 0.4 K | 0 (drop, too small) | 0 |
| **TOTAL** | **40,308 K** | **~38.3 M** | **~2.0 M** |

`L2-train-pool` is the input to **per-source weighted upsampling** for Phase 2 (so OFAC's natural 45% share doesn't drown the smaller sources). Target: each source's share in the upsampled stream ≈ `sqrt(natural_share)` normalized — a standard square-root tempering that keeps OFAC dominant but gives FATF and courtlistener a fighting share.

| Source | Natural share | Tempered share | Upsample factor |
|---|---:|---:|---:|
| ofac_guidance | 45.0% | 26.0% | 3.5× |
| fincen_federal_register | 14.2% | 14.6% | 6.2× |
| fincen_sar_reviews | 12.6% | 13.8% | 6.6× |
| fincen_advisories | 11.5% | 13.2% | 6.9× |
| fincen_enforcement | 10.6% | 12.6% | 7.2× |
| courtlistener | 3.2% | 6.9% | 13× |
| fatf_publications | 2.6% | 6.2% | 14× |
| fincen_files | 0.3% | 6.7% | (cap to 8×; drop excess) |
| **Effective L2 stream** | 100% | 100% | **~6.3× avg** |

Total upsampled L2 stream ≈ **~240 M tokens** per Phase-2 epoch — squarely inside the 200–320 M target from the planning todo. Capping `fincen_files` at 8× (instead of the 100×+ that pure tempering would request) avoids near-pathological repetition of 107 K tokens.

### 3.3 Phase 2 batch composition

At every microstep, the dataloader samples one of two streams:

| Stream | Source pool | Per-step share | Per-epoch tokens |
|---|---|---:|---:|
| L2 (upsampled) | `L2-train-pool` × per-source factors above | **44%** | ~240 M |
| L1 (replay) | `L1-replay-pool` (~1,250 M unseen EDGAR + small-source eval-tail) | **56%** | ~300 M |
| **Per-epoch total** | | 100% | **~540 M** |

Critical property: across **4 Phase-2 epochs**, total replay draw = 1,200 M ≈ size of the replay pool. **The model never sees an EDGAR document twice across the entire CPT pipeline** — every token in `L1-replay-pool` is held back from Phase 1 specifically so Phase 2 has fresh distribution-matched anti-forgetting signal.

L1:L2 effective ratios:

| Period | L1 tokens | L2 tokens | L1:L2 |
|---|---:|---:|---:|
| Phase 1 alone | 1,820 M | 0 | ∞ |
| Phase 2 (4 epochs) | 1,200 M | 960 M | 1.25 : 1 |
| **Lifetime CPT** | **3,020 M** | **960 M** | **3.1 : 1** |

The lifetime 3:1 is more aggressive (more AML-narrow) than the 10:1 in the planning todo. That's deliberate because **AML is the actual deployment domain**. If post-Phase-2 evals show MMLU regression > 5pp (per `training_strategy.md` §6.7), re-balance toward 5:1 by raising the L1-replay share from 56% → 70% in a Phase 2 re-run.

### 3.4 Pre-prep filters that apply to both

These run before the partition, on the curated `.jsonl` outputs from [`../1.data_curation/`](../1.data_curation/) — the partition then sees clean text only.

1. **Document-level shuffle** seeded once, recorded in `data_prep_manifest.json` for reproducibility.
2. **Length floor 200 chars / ceiling 1 M chars** (matches `training_strategy.md` §6.2).
3. **Pack to 4096-token sequences with `<eod>` separators**, using the base-model tokenizer (Megatron `.bin/.idx` format, via `preprocess_megatron_dataset.py` from [`../reference/2.prepare_megatroncore_dataset/`](../reference/2.prepare_megatroncore_dataset/)).
4. **Manifest deliverable:** `data_prep_manifest.json` records, per pool: source list, doc count, token count, SHA-256 of each `.bin` file. Required for audit and for Phase-2 reproducibility.

---

## 4. Recommended Training Parameters

### 4.1 Hardware fit vs the reference recipe

The reference recipe ([`../reference/3.run_cpt/recipe_h200-8.yaml`](../reference/3.run_cpt/recipe_h200-8.yaml)) ran on **8×H200 (140 GB)** at **mem ≈ 110 GiB / GPU**. Our hardware is **8×H100 NVL (94 GB / GPU)** — the reference will OOM as-is. Two changes restore headroom:

| Knob | Reference (H200) | This recipe (H100 NVL) | Why |
|---|---|---|---|
| `local_batch_size` | 2 | **1** | Halves activation memory |
| `ep_size` | 1 (FSDP only) | **8** | Distributes 128 routed experts (16/GPU) instead of FSDP-gathering all of them every microstep |
| `activation_checkpointing` | true | **true** (keep) | Mandatory at this size |
| `sequence_parallel` | false | **false** (TP=1, no benefit) | — |

**Estimated per-GPU footprint** at `seq=4096, lbs=1, FSDP2+EP=8, act-ckpt=on, FP8`:

| Component | Size |
|---|---:|
| Sharded forward weights (FP8) | ~4 GB |
| Sharded optimizer (FP32 master + AdamW m, v) | ~45 GB |
| Activations (act-ckpt, lbs=1, seq=4096; FP8 matmul tensors) | 8–12 GB |
| TE FP8 amax history + scale buffers | ~1 GB |
| Comm buffers + workspace | ~5 GB |
| **Total** | **~63–67 GB / GPU** (~27–31 GB headroom) |

FP8 saves ~4 GB on forward weights and ~3 GB on activations vs the same recipe in higher precision. The optimizer state is unchanged (master copies and moments stay FP32 for update precision). Headroom can be re-invested into a longer sequence (`seq_length=6144`) or higher `local_batch_size=2` if the smoke test (§6 risk #2) shows stable FP8 loss; default recipe stays conservative.

### 4.2 Parallelism

```
World size = 8
FSDP2: full shard across all 8 ranks
EP    = 8   (one expert-parallel group of size 8; 16 routed experts/GPU; shared expert replicated)
TP    = 1   (no benefit on 8 GPUs at this size)
PP    = 1   (would hurt MoE routing across only 8 ranks)
CP    = 1   (seq_length 4096 doesn't justify the overhead)
DP    = 8   (implicit from FSDP shard count)
```

### 4.3 Sequence length, batch sizing, throughput

| Knob | Value | Rationale |
|---|---|---|
| `seq_length` | **4096** | Reference used 2048 — wasteful for our long docs (EDGAR avg 45 K, FinCEN PDFs avg 13 K). 4096 captures statute → subsection → application without paying the quadratic cost of 8K. Mamba layers absorb long context cheaply; attention layers stay tractable. |
| `local_batch_size` | **1** | Memory-driven (§4.1). |
| **Tokens per microstep** | 4,096 | seq × lbs |
| `global_batch_size` (sequences) | **256 (Phase 1) / 128 (Phase 2)** | Phase 1 → ~1.05 M tok/step (sweet spot for 30B CPT); Phase 2 halved → ~525 K tok/step (smaller corpus needs more optimizer steps per epoch to learn). |
| Grad accumulation (implied) | **32 / 16** | gbs / (lbs × world_size) |

**Throughput projection.** Reference logged **~1.67 K t/s/GPU** on H200:

```16329:16329:/data/swami/gsi-training/3.cpt/reference/3.run_cpt/training.log
2025-12-27 16:05:15 | INFO | root | step 4 | epoch 0 | loss 3.9796 | grad_norm 6.2927 | lr 7.25e-06 | mem 109.89 GiB | tps 15014.28(1876.79/gpu) | num_label_tokens 524288
```

H100 NVL has ~70% of H200's HBM bandwidth (3.35 vs 4.8 TB/s), but FP8 compute on H100 doubles peak matmul throughput vs the reference's higher-precision baseline (1979 vs 989 TFLOPS). Net expectation: **~1.4–1.7 K t/s/GPU**, i.e. **~11–14 K t/s aggregate** — comparable to or slightly above the reference. **Validate with a 30-step smoke test** before the full run (§6 risk #2).

### 4.4 Phase 1 hyperparameters

| Setting | Value | Why |
|---|---|---|
| Data | `L1-train-phase1` (~1.82 B tokens) | §3.1 |
| Epochs | 1 | Strategy doc §6.6 |
| `seq_length` | 4096 | §4.3 |
| `local_batch_size` | 1 | §4.1 |
| `global_batch_size` | 256 | §4.3 |
| Optimizer steps | ~1,734 | 1.82 B / 1.05 M |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1, eps=1e-8) | Strategy doc §6.6; standard for MoE CPT |
| Peak LR | **2.0e-5** | Conservative for full-param 30B CPT on a corpus close to base distribution. Reference's 5e-5 was justified by translation/transliteration domain shift; AML/finance is much closer to base. |
| Min LR | 2.0e-6 | peak / 10 |
| Schedule | **WSD** (Warmup-Stable-Decay) | Better than cosine for staged CPT — Phase 2 can resume from the WSD decay endpoint without an LR reset. |
| Warmup steps | 100 (~6%) | Standard for resumed-pretraining |
| Stable steps | ~1,200 (~70%) | Held at peak |
| Decay steps | ~430 (~24%) | Linear or 1-sqrt to min_lr |
| Gradient clip | 1.0 | Standard |
| Precision | FP8 (Transformer Engine, E4M3 forward / E5M2 backward); FP32 master + FP32 optimizer moments; FP32 reductions | FP8 throughput on H100 + FP32 master for update precision |
| FP8 recipe | `delayed_scaling` with 16-step amax history, margin=0; first 50 steps in higher precision then promote to FP8 | Standard TE pattern; warmup avoids amax instability while loss is large |
| Checkpointing | every 250 steps (~7 ckpts total) | Mid-snapshot at step ~900 is the rollback fallback |
| Validation | every 200 steps on `L1-holdout-eval`, stratified per-source | §3.1 |
| **Wall clock estimate** | **~40 hours ≈ 1.7 days** | 1.82 B / 12.5 K t/s |

### 4.5 Phase 2 hyperparameters (deltas only)

| Knob | Phase 1 | **Phase 2** | Why |
|---|---|---|---|
| Base | FP8 base checkpoint | **`cpt-l1-final`** | Resume |
| Data | `L1-train-phase1` | **`L2-train-pool` (upsampled) + `L1-replay-pool` mix per §3.3** | |
| `global_batch_size` | 256 | **128** | Halve to ~525 K tok/step — small corpus needs more optimizer steps/epoch to learn (you want hundreds, not dozens) |
| Grad accum | 32 | **16** | Falls out from above |
| Peak LR | 2.0e-5 | **5.0e-6** | Quarter of Phase 1. Phase 1 already moved the model; Phase 2 fine-tunes on top. Higher LR over-narrows. |
| Min LR | 2.0e-6 | **5.0e-7** | peak / 10 |
| Schedule | WSD | **Cosine, restart from Phase-1 final LR** | Resume from WSD decay endpoint, cosine through Phase 2 |
| Warmup | 100 steps | **50 steps** | Just to ramp from Phase-1 final LR to Phase-2 peak |
| `num_epochs` | 1 | **4** (early-stop, patience=1) | OK because per-epoch corpus is small; monitor val PPL — stop if L1-replay PPL climbs >10% (forgetting signal) |
| Optimizer steps | ~1,734 | ~4,114 | (4 × 540 M) / 525 K |
| Validation | `L1-holdout-eval` | **`L1-holdout-eval` + `L2-holdout-eval`** | Both layers tracked separately |
| **Wall clock estimate** | ~40 h | **~48 hours ≈ 2.0 days** | 2.16 B / 12.5 K t/s |

**Phase 2 evaluation gates** (per `training_strategy.md` §6.7):

| Holdout | Watch for | Action |
|---|---|---|
| `L2-holdout-eval` PPL | Should drop monotonically | If it plateaus before epoch 3, early-stop |
| `L1-holdout-eval` PPL | Should not rise > 10% from Phase-2 start | If it rises, raise L1-replay share to 70% next epoch |
| MMLU sample (1K Qs) | Drop ≤ 5pp vs base | If exceeded, restore mid-Phase-2 ckpt and reduce LR to 2.5e-6 |
| MoE expert-load balance | Each expert in [0.5%, 5%] of tokens | If unbalanced, raise `load_balancing_loss_coeff` 1e-2 → 5e-2 |

### 4.6 MoE-specific knobs

```yaml
moe:
  router_z_loss_coeff: 1.0e-3       # discourages large router logits → stability
  load_balancing_loss_coeff: 1.0e-2 # NeMo default
  shared_expert_overlap: true       # overlap shared-expert compute with routed dispatch
  token_dispatcher: alltoall        # standard for EP
  permute_fusion: true              # if available in 26.04
```

Monitor every 100 steps: per-expert token-fraction histogram. Per `training_strategy.md` §6.7, alarm if any expert receives <0.5% or >5% of tokens.

### 4.7 Combined budget summary

| | **Phase 1** | **Phase 2** |
|---|---|---|
| Base | `Nemotron-3-Nano-30B-A3B-FP8` | `cpt-l1-final` |
| Data pools | `L1-train-phase1` | `L2-train-pool` (upsampled) + `L1-replay-pool` |
| Data tokens / run | ~1.82 B | ~2.16 B (4 × 540 M) |
| Epochs | 1 | 3–4 (early-stop) |
| Seq length | 4096 | 4096 |
| Local batch | 1 | 1 |
| Global batch (seqs) | 256 | 128 |
| Tokens / opt step | ~1.05 M | ~525 K |
| Grad accum | 32 | 16 |
| Parallelism | FSDP2 + EP=8 + ckpt | same |
| Peak LR | 2e-5 | 5e-6 |
| Schedule | WSD | Cosine |
| Optimizer | AdamW (0.9, 0.95, wd=0.1) | same |
| Optimizer steps | ~1,734 | ~4,114 |
| Wall clock | ~40 h | ~48 h |
| Per-GPU mem | ~65 GB | ~65 GB |
| Precision | FP8 (TE) + FP32 master + FP32 moments | same |
| Checkpoints | every 250 steps | every 250 steps |
| Deliverable | `cpt-l1-final/` + mid-snapshot | `cpt-l2-final/` |

**Total CPT compute: ~88 hours ≈ 3.7 days.** Leaves **~10 days** for SFT + RL + eval inside the 14-day budget.

---

## 5. How Training Happens on Long-Sequence Records

A natural question once you look at the corpus: **what happens to a 146 KB EDGAR 10-K — about 32,000 tokens after the FP8 tokenizer — when training is configured with `seq_length=4096`?** Is it truncated? Replayed in 8 windows? Concatenated with neighboring docs? This section pins down the answer with code citations so the behavior isn't a guess.

<a id="5-1"></a>
### 5.1 The packing algorithm in one paragraph

Megatron-LM's `GPTDataset` (which NeMo's `MegatronPretraining` wraps unchanged) does **concat-and-chunk packing**. It treats every `.bin` shard as one giant flat token stream — laid out in the order given by a per-epoch-shuffled `document_index` — and slices that stream into back-to-back, non-overlapping `seq_length`-token windows. Long documents naturally span multiple consecutive samples. Short documents get co-packed into a single sample, separated only by the EOS token that `--append-eod` wrote into the `.bin` at Stage 2. There is **no truncation**, **no stride/overlap**, and **no per-document zero-padding**. The only loss is at most one trailing partial window of the *whole shuffled corpus per epoch* (controlled by `drop_last_partial_sequence=True`, hardcoded for the train split). The 1-token "extra" you see in the algorithm is the standard input/label shift, not a stride.

Authoritative sources:
- [`gpt_dataset.py:296-377`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py) — `_query_document_sample_shuffle_indices`: the per-step assembly that does the (doc_idx, offset) lookup and `np.concatenate` of multi-document samples.
- [`helpers.cpp:143-229`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/helpers.cpp) — `build_sample_idx`: the C++ inner loop that slices the flat token stream into `seq_length` windows.
- [`gpt_dataset.py:504-507`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py) — `drop_last_partial_sequence=True` for train, configurable for valid.
- [`megatron_dataset.py:253-266`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/llm/megatron_dataset.py) — NeMo-Automodel's `MegatronPretraining` hardcodes `reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False`.

<a id="5-2"></a>
### 5.2 Worked example — a 32,000-token EDGAR 10-K at seq_length=4096

After packing into the `.bin`, our document `D` lives at some position `k` in `document_index`. Assume the previous sample ended cleanly at `(k-1, end)` so the next sample starts at `(k, 0)`. `build_sample_idx` walks forward, advancing `doc_offset` by `seq_length` per iteration:

| Sample # | enters with `doc_offset` | tokens covered (within `D`) | exits at |
|---:|---:|---|---:|
| 1 | 0     | `D[0:4096]`     — first 4,096 tokens | 4,096 |
| 2 | 4,096 | `D[4096:8192]`                       | 8,192 |
| 3 | 8,192 | `D[8192:12288]`                      | 12,288 |
| 4 | 12,288| `D[12288:16384]`                     | 16,384 |
| 5 | 16,384| `D[16384:20480]`                     | 20,480 |
| 6 | 20,480| `D[20480:24576]`                     | 24,576 |
| 7 | 24,576| `D[24576:28672]`                     | 28,672 |
| 8 | 28,672| `D[28672:32000]` (3,328 tokens incl. EOS) **+** `D'[0:769]` (next doc) | 769 in `D'` |

So our 32,000-token document is consumed **without loss** across 8 consecutive entries in `sample_index`. The `shuffle_index` (a NumPy random permutation seeded by `rng.seed`) then permutes the *visiting order*, so during training those 8 chunks appear at 8 random global steps within the epoch — **not** as 8 adjacent training steps. Every token is seen exactly once per epoch.

<a id="5-3"></a>
### 5.3 Where each guarantee is enforced

The behavior above is delivered by the **combination** of Stage 2 + Stage 3 in [`run-v1/`](run-v1/). One missing piece would silently break things:

| Behavior | Enforced by | Where exactly |
|---|---|---|
| Tokenize each doc once, no truncation at preprocess time | Stage 2 | [`preprocess_megatron_dataset.py`](run-v1/2.prepare_megatroncore_dataset/preprocess_megatron_dataset.py) calls `IndexedDatasetBuilder.add_document(token_ids)` per line. There is no `seq_length` argument at this stage — every doc is stored at its full token length in the `.bin`. |
| EOS appended at end of every document | Stage 2 (`--append-eod` flag) | The flag in our [`run-v1/README.md`](run-v1/README.md) Stage 2 invocation. The script has `if append_eod: doc_ids.append(tokenizer.eos_token_id)` per record. **Without `--append-eod`, document boundaries silently disappear when the dataloader concatenates docs and the model learns garbage cross-doc transitions.** |
| Variable-length docs preserved in `.bin/.idx` | Stage 2 | `IndexedDatasetBuilder` records `(byte_offset, length)` per document in the `.idx` — no fixed-size assumption. |
| Concat-and-chunk packing into `seq_length` samples | Stage 3 (recipe + NeMo wrapper) | `seq_length: 4096` in [`recipe_h100nvl-8.yaml`](run-v1/3.run_cpt/recipe_h100nvl-8.yaml) is what `build_sample_idx` slices on. The packing algorithm itself comes from the `_target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining` line in the recipe — that class wraps Megatron-Core's `GPTDataset`, which is where the `_query_document_sample_shuffle_indices` + `build_sample_idx` code cited above lives. |
| Long docs split across consecutive samples without loss | Stage 3 (algorithmic) | Nothing in the recipe controls this — it's the only thing `GPTDataset` knows how to do. |
| Per-epoch shuffle of sample visiting order | Stage 3 | The recipe's `rng.seed: 1111` seeds `_build_shuffle_index`. |
| At most ~3,000 tokens dropped from the *whole corpus per epoch* | Stage 3 (default) | `drop_last_partial_sequence=True` is hardcoded for the train split in `gpt_dataset.py:504-507`. |

<a id="5-4"></a>
### 5.4 Three NeMo defaults we inherit but never set explicitly

These come from `MegatronPretraining.__init__` hardcoding them — they are inherited automatically when you use `_target_: ...MegatronPretraining`, and **there is no YAML key to override them today**:

| Behavior | Default | Effect on our run | If we wanted to change |
|---|---|---|---|
| `reset_attention_mask=False` | inherited | Within a sample like `[D_tail, EOS, D'_head]`, `D'` can causally attend across the EOS to `D_tail`. Standard GPT-style behavior; matches GPT-3 / Llama / the reference recipe. | Subclass `MegatronPretraining` (no YAML key today). |
| `reset_position_ids=False` | inherited | Position IDs are `arange(0, 4096)` over the whole sample — `D'`'s first token gets position 3,500-ish, not 0. | Same — subclass required. |
| `eod_mask_loss=False` | inherited | Model trains on EOS predictions (good — it learns when to emit a stop). Loss is only masked on PAD positions. | Same — but you probably don't want to change this one. |

For broad-register CPT on a corpus where document boundaries are mostly clean (10-K filings, OIG reports, statutes), enabling cross-doc attention is the standard choice — it's not a bug. Worth knowing about so the behavior isn't surprising in the perplexity logs.

<a id="5-5"></a>
### 5.5 Information-loss accounting

For Phase 1 (`level_1` corpus, 1 epoch):

- 1.82 B tokens in train (post-tokenize)
- 1.82 B / 4,096 ≈ **444,580 samples** per epoch
- `drop_last_partial_sequence=True` drops at most one trailing partial window of < 4,096 tokens **from the whole shuffled corpus**, not from any single document
- **Worst-case loss: 0.000165% of training tokens.** Effectively zero.

For Phase 2 (`level_2` corpus, 4 epochs with replay): same accounting per epoch; the per-epoch shuffle re-seats which tokens fall in the dropped tail, so over 4 epochs effectively every token in `level_2` is seen at least once.

**Bottom line: no per-document truncation, no padding waste, no information loss beyond a sub-0.001% trailing-window drop. The 32,000-token EDGAR 10-K (and every other long doc) is presented to the model as N consecutive training samples spread across N random global steps within the epoch — every byte intact.**

---

## 6. Concrete Phase 1 Recipe (drop-in YAML)

```yaml
# recipe_phase1_h100nvl.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  trust_remote_code: true
  torch_dtype: auto                 # honor the FP8 checkpoint's native dtype
  use_mamba_kernels: true

# Transformer Engine FP8 training recipe (verify exact key names in 26.04)
fp8:
  enabled: true
  recipe: delayed_scaling           # E4M3 fwd / E5M2 bwd
  amax_history_len: 16
  amax_compute_algo: max
  margin: 0
  warmup_steps: 50                  # train first 50 steps in higher precision, then promote

dataset:
  _target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining
  paths: /workspace/3.cpt/2.run_cpt/data/phase1/L1_train_phase1_*_document*
  index_mapping_dir: /workspace/3.cpt/2.run_cpt/mapping_dir
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
    trust_remote_code: true
  seq_length: 4096
  split: "1.0, 0.0, 0.0"          # holdout is a separate dataset (see below)
  splits_to_build: "train"

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining
  paths: /workspace/3.cpt/2.run_cpt/data/eval/L1_holdout_eval_*_document*
  index_mapping_dir: /workspace/3.cpt/2.run_cpt/mapping_dir
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
    trust_remote_code: true
  seq_length: 4096
  split: "0.0, 1.0, 0.0"
  splits_to_build: "validation"
  num_val_samples: 2048

step_scheduler:
  global_batch_size: 256
  local_batch_size: 1
  ckpt_every_steps: 250
  val_every_steps: 200
  num_epochs: 1
  max_steps: 1800                  # safety cap above estimated ~1734

dist_env:
  backend: nccl
  timeout_minutes: 15

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  tp_size: 1
  cp_size: 1
  pp_size: 1
  ep_size: 8                       # critical change vs reference
  activation_checkpointing: true
  sequence_parallel: false

optimizer:
  _target_: torch.optim.AdamW
  betas: [0.9, 0.95]
  lr: 2.0e-5
  weight_decay: 0.1
  eps: 1.0e-8

lr_scheduler:
  lr_decay_style: wsd              # if 26.04 supports it; else cosine
  lr_warmup_steps: 100
  lr_decay_steps: 430
  min_lr: 2.0e-6

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

checkpoint:
  enabled: true
  checkpoint_dir: /workspace/3.cpt/2.run_cpt/checkpoints/phase1/
  model_save_format: safetensors
  save_consolidated: true

# MoE-specific keys per nemo-automodel:26.04 schema (verify exact key names):
# moe.router_z_loss_coeff: 1.0e-3
# moe.load_balancing_loss_coeff: 1.0e-2
```

**Launch command** (analogous to the reference's invocation):

```bash
docker run --rm --gpus all \
  --name cpt-phase1 \
  --shm-size=64g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data/swami/gsi-training:/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  torchrun --nproc-per-node=8 \
    /workspace/3.cpt/2.run_cpt/pretrain.py \
    --config /workspace/3.cpt/2.run_cpt/recipe_phase1_h100nvl.yaml \
  > /workspace/3.cpt/2.run_cpt/phase1.log 2>&1
```

The Phase 2 YAML is a delta of the above (different `paths`, blended dataset, lower LR, cosine schedule, gbs=128) and will be drafted once the data-prep script lands.

---

## 7. Risks to Pre-mitigate

1. **FP8 training stability on hybrid Mamba-Transformer MoE.** Transformer Engine FP8 is well-validated on dense transformers and standard MoE; the Mamba state-space layers and the MoE router are the unknowns. Mitigations: (a) the 50-step higher-precision warmup before promoting to FP8 (see YAML §5); (b) per-step `loss` and `grad_norm` watchdog — if loss spikes >2× the running mean or grad_norm exceeds 5.0, auto-disable FP8 for the next 100 steps then re-enable; (c) keep MoE router compute outside FP8 (router logits are small but precision-sensitive — set `fp8_router: false` in the TE config).
2. **EP=8 + Mamba-layer interaction.** The reference disabled the MoE parallelizer (`# parallelizer: ...` in [`recipe_h200-8.yaml`](../reference/3.run_cpt/recipe_h200-8.yaml)) because it was incompatible with `NemotronH` (uses `backbone` not `model`). Verify in `nemo-automodel:26.04` release notes that this was fixed; if not, fall back to FSDP-only and accept tighter memory (mitigate with CPU offload of optimizer state — slower but fits).
3. **Throughput uncertainty.** The 1.4–1.7 K t/s/GPU estimate combines H100 NVL bandwidth ratios and FP8 speedup expectations — neither has been measured for this exact stack. Run a **30-step smoke test** with `max_steps=30, ckpt=disabled` first; if measured tps < 1.0 K t/s/GPU, drop `seq_length` to 3072 (still better than the reference's 2048) and re-budget wall clocks accordingly.
4. **EDGAR-CORPUS dominance.** Even with the cap, EDGAR is 82% of Phase 1. If per-source val PPL shows EDGAR over-fitting (>50% PPL drop) while supporting sources barely move (<10%), tighten the EDGAR cap to 1.0 B in a re-run.
5. **L2 over-fitting in Phase 2.** With 6× upsampling, each L2 token is seen ~24 times across 4 epochs. Watch for L2 train loss collapsing toward 0 while L2 holdout PPL stagnates — that's memorization. If observed, reduce L2 upsampling to 4× and extend to 5 epochs.
6. **Catastrophic forgetting.** The L1-replay design mitigates this, but the 3:1 lifetime ratio is more aggressive than the strategy doc's 10:1 starting point. MMLU regression > 5pp is the abort signal — see §4.5 actions.

---

## 8. Open Questions

1. Confirm `nemo-automodel:26.04` exposes the FP8 / Transformer Engine config keys assumed in §5 — exact key names may differ; verify against the container's example recipes before launch.
2. Phase 2 L1-replay share: **56% (default) or 70% (forgetting-resistant)**?
3. Phase 2 epoch count: **3 or 4**? (Default 4 with early-stop is the safer choice.)
4. Should I draft (a) the **Phase 2 recipe YAML**, (b) the **data-prep script** that produces all five `.bin/.idx` pools (`L1-train-phase1`, `L1-replay-pool`, `L1-holdout-eval`, `L2-train-pool`, `L2-holdout-eval`) per §3, and (c) the manifest writer?

---

## 9. Directory Layout

```
3.cpt/2.run_cpt/
├── README.md                          # this file
├── pretrain.py                        # copied from ../reference/3.run_cpt/
├── recipe_phase1_h100nvl.yaml         # §5
├── recipe_phase2_h100nvl.yaml         # to be drafted
├── data_prep.py                       # to be drafted (produces 5 pools per §3)
├── data_prep_manifest.json            # auto-emitted by data_prep.py
├── data/
│   ├── phase1/                        # L1-train-phase1 .bin/.idx shards
│   ├── phase2/                        # L2-train-pool + L1-replay-pool shards
│   └── eval/                          # L1-holdout-eval + L2-holdout-eval shards
├── mapping_dir/                       # NeMo dataloader index cache
├── checkpoints/
│   ├── phase1/                        # 250-step ckpts + cpt-l1-final + mid-snapshot
│   └── phase2/                        # 250-step ckpts + cpt-l2-final
├── phase1.log
└── phase2.log
```
