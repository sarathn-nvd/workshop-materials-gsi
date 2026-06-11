# Stage 3 — CPT Phase 1 training

Full-parameter continued pretraining of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` on the tokenized `level_1` shards from Stage 2. Produces `cpt-l1-final/` — the checkpoint that `../run-v2/` (Phase 2) resumes from.

| | |
|---|---|
| Entry point | [`pretrain.py`](pretrain.py) (copied from `../../reference/3.run_cpt/`, unchanged) |
| Recipe | [`recipie_a100sxm-8.yaml`](recipie_a100sxm-8.yaml) (Phase 1 / BF16 / A100 SXM 80 GB) |
| Eval | [`compare_perplexity.py`](compare_perplexity.py) (adapted for our `text` schema, per-source stratified) |
| Input | `../2.prepare_megatroncore_dataset/data/processed_data_*_text_document.{bin,idx}` (32 shards, ~2.31 B tokens) |
| Output | `checkpoints/{epoch_0_step_*/, LATEST/, LOWEST_VAL/}` — `LOWEST_VAL/model/consolidated/` is the Phase-2 input |
| Wall clock | ~18-20 hours measured on A100 SXM (37 sec/step × 1636 effective steps) |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` |

The recipe YAML has inline comments explaining every non-obvious knob. The H100-NVL FP8 sibling recipe (`recipe_h100nvl-8.yaml`, if it exists in this directory) is the FP8-tier reference; A100 has no FP8 hardware (Ampere SM_80) so this directory ships the BF16 A100 recipe as the working configuration.

## Recipe at a glance (A100 SXM, BF16)

| Setting | Value | Source of truth |
|---|---|---|
| Base model | `Nemotron-3-Nano-30B-A3B-Base-BF16` | gated NVIDIA HF repo |
| Precision | BF16 (TE backend, FlashAttention BF16 path) + FP32 master + FP32 AdamW moments | `fp8.enabled: false` — A100 has no FP8 tensor cores |
| Sequence length | 4096 | balances long doc context vs activation memory |
| Local batch size | 2 | memory-driven for 80 GB A100 SXM (~57 GiB measured peak at LBS=2) |
| Global batch size | 256 (~1.05 M tokens / opt step) | sweet spot for 30 B CPT |
| Grad accum | 16 (= 256 / 2×8) | falls out from above |
| Parallelism | FSDP2 + EP=8, no TP/PP/CP | `ep_size=8` forced by MoE optimizer-state memory at 80 GB |
| Activation checkpointing | on | mandatory at this size; ~125 GiB peak without |
| Optimizer | AdamW (β=0.9, 0.95; wd=0.1; eps=1e-8) | standard MoE CPT |
| Peak LR | **2.0e-5** | conservative for 30B CPT close to base distribution |
| Min LR | 2.0e-6 (peak / 10) | |
| LR schedule | **WSD** (warmup 100, stable 1270, decay 430) | enables clean Phase 2 resume |
| WSD trailing decay style | `cosine` (`lr_wsd_decay_style`) | gentle annealing onto sharp minimum |
| `max_steps` | 1800 (= 1734 + safety; actual one-epoch cap is 1636) | sized for 1 epoch over 2.31 B tokens |
| Validation cadence | every 190 steps (in-training NeMo split) | + post-training `compare_perplexity.py` on the Stage 1 holdout |
| Checkpoint cadence | every 100 steps | mid-snapshots are rollback fallbacks |
| MoE knobs | (default — router + load balancing handled inside the model) | + monitoring for per-expert load in [0.5%, 5%] |

Estimated per-GPU footprint: **~57 GiB** measured peak (~23 GiB headroom on 80 GB).

## How to run

The full launch sequence with all prerequisites (SSH to A100 node, NGC docker login, `HF_TOKEN` export) is documented inline at the top of [`recipie_a100sxm-8.yaml`](recipie_a100sxm-8.yaml). Operationally:

```bash
ssh swaminathanb@exp-blr-dgxa100-02
cd /sadata/swaminathanb/gsi-training
export HOST_WORKSPACE=/sadata/swaminathanb/gsi-training
export HF_TOKEN=hf_xxx_your_token

# (One-time on this node) Authenticate against NGC
docker login nvcr.io
#   Username: $oauthtoken
#   Password: <NGC_API_KEY>

# Launch detached. (Rootless Docker: omit --user / --group-add 0;
# container UID 0 maps to your host UID via the user namespace.)
docker run -d --gpus all \
  --name cpt-phase1 \
  --shm-size=64g --ipc=host \
  --ulimit memlock=-1 \
  --workdir /workspace/3.run_cpt \
  -e HOME=/tmp \
  -e HF_HOME=/workspace/3.run_cpt/hf_cache \
  -e HF_TOKEN=$HF_TOKEN \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $HOST_WORKSPACE:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  torchrun --nproc-per-node=8 \
    /workspace/3.run_cpt/pretrain.py \
    --config /workspace/3.run_cpt/recipie_a100sxm-8.yaml

# Stream container stdout/stderr to a file (run from $HOST_WORKSPACE)
nohup docker logs -f cpt-phase1 > 3.run_cpt/run.log 2>&1 &
disown

# Watch progress (filtered for signal)
tail -f 3.run_cpt/run.log | grep -E 'step|loss|val|ppl|expert|Saving|Updated LOWEST'

# Stop cleanly
docker stop cpt-phase1 && docker rm cpt-phase1
```

Why each flag matters:

| Flag | Purpose |
|---|---|
| `docker run -d --name cpt-phase1` | Detached named container so `docker logs` / `docker stop` can target it. |
| `--gpus all` | Expose all 8 A100 SXMs to the container. Requires nvidia-container-toolkit configured for rootless if running rootless. |
| `--shm-size=64g --ipc=host` | NCCL + PyTorch DataLoader shared-memory needs. |
| `--ulimit memlock=-1` | NCCL pinned-memory buffers. The companion `--ulimit stack=67108864` was REMOVED for rootless Docker — `RLIMIT_STACK` (rlimit type 3) cannot be raised above the host's `ulimit -Hs` ceiling under rootless, and the default 8 MiB stack is fine for this workload. |
| `--workdir /workspace/3.run_cpt` | CWD inside the container is a writable bind-mount path so any cwd-relative writes succeed. |
| `-e HF_HOME=...` `-e HF_TOKEN=...` | HF download cache lives on the bind mount (so re-launches are near-instant); HF_TOKEN is required for the gated Nemotron-3 repo. |
| `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces fragmentation pressure during val/ckpt memory spikes. |
| `-v $HOST_WORKSPACE:/workspace` | Bind the project root so paths like `/workspace/3.run_cpt/...` resolve. Under rootless Docker, files written via this mount are owned by your real host UID (because container UID 0 = your host UID via the user namespace), so checkpoint output is yours by default. |

### Smoke test first

Before committing to the full run, validate throughput with a 30-step dry run. Set `step_scheduler.max_steps: 30` and `checkpoint.enabled: false` in a copy of the recipe, then launch as above. Expected `tps` in the log on A100 SXM 80GB BF16:

| Tier | Per-GPU tps | Aggregate tps | Verdict |
|---|---:|---:|---|
| Healthy (steady state) | ~3500 | ~28,000 | proceed with full run |
| Acceptable | 2,500–3,500 | 20,000–28,000 | proceed; expect ~22-25 h instead of ~18-20 h |
| Slow | < 2,000 | < 16,000 | drop `seq_length` to 3072 and re-smoke; check for noisy-neighbor on the node |

Step 0 reports anomalously low tps (~990/GPU in the measured run) because cold-start overhead (FSDP first all-gather, allocator warmup, dataset index touches) is included in the integrated wall clock. Step 1 onward returns to steady state.

## Watch for during training

| Signal | Healthy | Action if violated |
|---|---|---|
| Train loss trajectory | smooth, no spikes > 2× running mean | if persistent spike → kill, re-launch from latest ckpt |
| `grad_norm` | < 0.5 (measured 0.18–0.22 in stable phase) | raise log warning; abort if persistent at > 1.0 |
| Per-GPU memory | 55–62 GiB (measured 57.41 GiB at LBS=2) | if OOM near step 100 (first val pass): drop `local_batch_size: 2 → 1` |
| In-training val loss (every 190 steps) | monotonically descending; first val ≈ 1.09 | early-stop if plateau for 3 consecutive evals AND no decay-phase acceleration |
| LOWEST_VAL symlink updates | every val pass that improves on prior best | (informational — auto-managed) |
| LR trajectory | warmup 0–99 (linear 2e-6 → 2e-5); stable 100–1369 (locked at 2e-5); decay 1370–1799 (cosine 2e-5 → 2e-6) | if step 100's `lr` is not exactly `2.00e-05` → scheduler misconfigured |

The recipe was tuned to the token budget, not to per-step time. Do not modify `lr_warmup_steps`, `wsd_decay_steps`, or `lr_decay_steps` based on observed step time alone.

### Known one-epoch truncation

`num_epochs: 1` + the actual dataloader length (1636 steps for the 32-shard L1 corpus) means training terminates at **step 1636**, not at `max_steps: 1800`. The WSD decay phase therefore runs only 266 of its planned 430 steps (62%), and final LR ends at ~7.7e-6 instead of `min_lr=2e-6`. The `max_steps: 1800` value remains as a safety ceiling.

Effect on Phase 2 resume: Phase 2 must short-warmup back up from the Phase-1 endpoint LR (~7.7e-6), not assume it starts from `min_lr=2e-6`. See `../run-v2/3.run_cpt/recipe_a100sxm-8.yaml` for the Phase-2 configuration that handles this.

## Post-training — Compare perplexity

Once `checkpoints/LOWEST_VAL/model/consolidated/` exists, run the per-source eval against the Stage 1 holdout. The `LOWEST_VAL` symlink resolves to whichever checkpoint achieved the lowest val loss during training (typically the late-decay or final checkpoint).

```bash
docker run --rm --gpus all \
  -e HOME=/tmp -e HF_HOME=/workspace/3.run_cpt/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -v $HOST_WORKSPACE:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 /workspace/3.run_cpt/compare_perplexity.py \
    --input     /workspace/<your-stage-1-validation-jsonl> \
    --base_model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --ft_model   /workspace/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated \
    --batch_size 16 --max_len 1024 --warmup 2 --measure 50 --max_samples 10000 --use_bf16
```

This prints a table like:

```
Source                            Base PPL       FT PPL      Δ %
--------------------------------------------------------------------
edgar_corpus                        16.420        9.110     -44.5% <-- WIN
pile_of_law_oig                     14.880       11.205     -24.7% <-- WIN
pile_of_law_federal_register        15.110       12.660     -16.2% <-- WIN
pile_of_law_sec                     14.550       11.890     -18.3% <-- WIN
pile_of_law_cfr                     12.330       10.140     -17.8% <-- WIN
uscode_house                        11.205        9.730     -13.2%
pile_of_law_doj_guidance            13.880       11.470     -17.4% <-- WIN
pile_of_law_uscode                  10-890        9.620     -11.7%
====================================================================
WIN  : per-source PPL drop >= 15%
BAD  : per-source PPL increase > 5% (catastrophic forgetting on a held-back source)
```

The pass criterion is **per-source PPL drop ≥ 15%** for the dominant sources (EDGAR, OIG, federal_register, SEC), with no `BAD` markers anywhere. If `BAD` appears, the run over-narrowed and Phase 2 should resume from a **mid-snapshot** (e.g. `epoch_0_step_999/` or `epoch_0_step_899/`) instead of `LOWEST_VAL`.

Aggregate val_loss (the in-training metric) is a sanity gauge, not the verdict. The measured run achieved val_loss 1.0911 → 1.0467 (~4% aggregate drop); the per-source breakdown above is what determines whether Phase 2 is safe to start.

## Differences from the reference's H100/H200 recipes

| | Reference (H100 NVL FP8) | This (A100 SXM BF16) | Why |
|---|---|---|---|
| Base | `-FP8` | **`-BF16`** | A100 has no FP8 tensor cores (Ampere SM_80) |
| `torch_dtype` | `auto` | `auto` | honor model native dtype |
| `fp8:` block | `enabled: true` (delayed_scaling + warmup 50) | **`enabled: false`** | locked decision for A100 hardware tier |
| `seq_length` | 4096 | 4096 | identical (long EDGAR / FinCEN docs) |
| `local_batch_size` | 1 | **2** | A100 80 GB SXM has more headroom than H100 NVL 94 GB after dropping FP8 dense savings |
| `ep_size` | 8 | 8 | distributes 128 routed experts; required to fit in 80 GB |
| `lr` | 2e-5 | 2e-5 | identical |
| `lr_decay_style` | `WSD` | `WSD` | identical |
| `lr_decay_steps` | 1800 | 1800 | identical |
| `wsd_decay_steps` | 430 | 430 | identical |
| `lr_wsd_decay_style` | `cosine` | `cosine` | identical |
| `lr_warmup_steps` | 100 | 100 | identical |
| `max_steps` | 1800 | 1800 | identical |
| `dataset.split` | `0.99, 0.01, 0.0` | `0.99, 0.01, 0.0` | identical |
| `dist_env.timeout_minutes` | 15 | **30** | absorbs initial dataset index build on cold launch |
| Per-step wall-clock | ~22-25 sec (FP8 + H100) | ~37 sec (BF16 + A100) | hardware tier difference |
| Total wall-clock | ~22-30 h | ~18-20 h | A100 finishes faster because `num_epochs: 1` caps at ~1636 steps regardless of `max_steps` |
