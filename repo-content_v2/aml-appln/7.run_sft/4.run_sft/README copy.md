# Stage 5 - SFT training

Supervised fine-tuning of the Phase-2 CPT endpoint (`run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated/`, = `cpt-l2-final`) on the leak-stripped chat dataset under `4.run_sft/run-v2/4.remove_field/final_data/` (stage 3 filter -> stage 4.remove_field).

| | |
|---|---|
| Entry point | [`finetune.py`](finetune.py) (copied from `../../3.run_cpt/pretrain.py`, unchanged - same `TrainFinetuneRecipeForNextTokenPrediction` recipe class works for both CPT and SFT) |
| Recipe | [`recipe_a100sxm-8.yaml`](recipe_a100sxm-8.yaml) (Phase-1 SFT / BF16 / A100 SXM 80 GB; structurally identical to the H200 reference except for the A100-specific overrides documented in the recipe header) |
| Input model | `/workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated/` (= cpt-l2-final, ~63 GB) |
| Input data | `/workspace/4.run_sft/run-v2/4.remove_field/final_data/{sft_mixed.chunk.00,sft_mixed.val}.jsonl` (leak-stripped output of stage 4.remove_field; see [`../4.remove_field/README.md`](../4.remove_field/README.md)) |
| Output | `checkpoints/{epoch_N_step_*/, LATEST, LOWEST_VAL}` (the `LOWEST_VAL/model/consolidated/` is the final SFT artifact) |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` (same as CPT) |

## Recipe at a glance

| Setting | Value | Source |
|---|---|---|
| Base model | `cpt-l2-final` (= run-v2 LOWEST_VAL) | local consolidated dir |
| Precision | BF16 (`torch_dtype: bfloat16`, TE backend BF16) | `fp8.enabled: false` (no FP8 on Ampere) |
| Mamba kernels | **off** (`use_mamba_kernels: false`) | reference convention; ChatDataset packing path interacts poorly with mamba state-passing |
| Packing | `packed_sequence_size: 5350` | reference; tune via stage 2 percentile report |
| Local batch size | 1 | conservative; relies on packing for throughput |
| Global batch size | 16 (gradient accum = 2) | reference; small batches are standard for SFT |
| Parallelism | FSDP2 (dp_size=8) + EP=8, no TP/PP/CP | EP=8 mandatory on A100 80 GB (vs reference's `dp_size: none` on H200 140 GB) |
| Activation checkpointing | on | mandatory on 80 GB |
| Optimizer | AdamW, betas=(0.9, **0.999**), wd=0.1 | SFT convention -- different from CPT's beta2=0.95 |
| Peak LR | 1e-5 | SFT convention; ~2x CPT Phase-2 peak |
| Min LR | 1e-6 (peak / 10) | reference |
| LR schedule | cosine, warmup 100, decay auto-computed (= total_steps - 100) | reference |
| Epochs | 1 | reference |
| Val cadence | every 500 steps | reference |
| Ckpt cadence | every 1000 steps | reference |

## How to run

The full launch sequence (SSH, NGC login, HF_TOKEN export, docker run, log tail) is documented inline at the top of [`recipe_a100sxm-8.yaml`](recipe_a100sxm-8.yaml). Operationally:

```bash
ssh swaminathanb@exp-blr-dgxa100-02
cd /sadata/swaminathanb/gsi-training
export HOST_WORKSPACE=/sadata/swaminathanb/gsi-training
export HF_TOKEN=hf_xxx_your_token

docker login nvcr.io   # one-time per node

docker run -d --gpus all \
  --name sft-phase1 \
  --shm-size=64g --ipc=host \
  --ulimit memlock=-1 \
  --workdir /workspace/4.run_sft/run-v2/5.run_sft \
  -e HOME=/tmp \
  -e HF_HOME=/workspace/4.run_sft/run-v2/5.run_sft/hf_cache \
  -e HF_TOKEN=$HF_TOKEN \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $HOST_WORKSPACE:/workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  torchrun --nproc-per-node=8 \
    /workspace/4.run_sft/run-v2/5.run_sft/finetune.py \
    --config /workspace/4.run_sft/run-v2/5.run_sft/recipe_a100sxm-8.yaml

nohup docker logs -f sft-phase1 > 4.run_sft/run-v2/5.run_sft/run.log 2>&1 &
disown

tail -f 4.run_sft/run-v2/5.run_sft/run.log | grep -E 'step|loss|val|ppl|expert|Saving|Updated LOWEST'
```

## Watch for during training

| Signal | Healthy | Action if violated |
|---|---|---|
| Train loss trajectory | smooth descent, starts ~0.5-1.5 (much lower than CPT's ~1.1 because SFT loss is masked to assistant tokens only) | If first step reports loss > 5, model didn't load correctly -- kill, check `pretrained_model_name_or_path` |
| `grad_norm` | < 1.0 (typical SFT) | abort if persistent at > 5 |
| Per-GPU memory | comparable to CPT (~57-65 GiB at LBS=1 + packed_sequence=5350) | if OOM during val: drop `packed_sequence_size` |
| Throughput | ~10-20K tokens/sec aggregate (SFT is slower than CPT due to chat-template tokenization on the fly + ChatDataset's variable-length packing) | If < 5K aggregate, suspect dataloader bottleneck -- check `num_workers` is not deadlocking on chat-template overhead |
| `[val]` loss every 500 steps | monotonically descending | early-stop if plateau for 3 consecutive evals AND no further LOWEST_VAL updates |
| `LOWEST_VAL` symlink updates | tracks each best val | (informational, auto-managed) |

## Pitfalls observed during CPT that may also bite SFT

1. **`num_workers` and dataloader pickling**. CPT Phase 2 hit `TypeError: cannot pickle 'BufferedReader'` with the BlendedDataset + spawn workers. SFT's ChatDataset doesn't hold file handles (loads JSONL eagerly into memory), so `num_workers: 4` should be safe. If you see the same crash here, drop to `num_workers: 0`.

2. **Decay style casing**. SFT uses `cosine` (lowercase) which the scheduler matches at `Automodel/nemo_automodel/components/optim/scheduler.py:224`. Don't accidentally write `Cosine` or `COSINE` -- silent fall-through to the unsupported-style raise (the same trap WSD-vs-wsd hit in CPT Phase 1).

3. **Rootless Docker memlock cap**. The CPT recipes dropped `--ulimit stack=67108864` because rootless can't raise `RLIMIT_STACK` above the host hard cap. SFT here keeps only `--ulimit memlock=-1`; if you see a `failed to set rlimit type 8: operation not permitted` error, ask the cluster admins to raise `memlock` in `/etc/security/limits.d/` for your user.

## Post-training - evaluate

The CPT-style `compare_perplexity.py` measures next-token prediction PPL, which is a fine sanity check but not the right metric for SFT (where what matters is whether the model produces useful instruction-following completions, not whether each next token has low PPL).

For SFT, the meaningful evals are:
- **Held-out chat completion** on `sft_mixed.test.jsonl` (stage-1 output that wasn't seen by training or in-training val). Generate completions, compare to ground-truth assistant turns via BLEU/ROUGE/exact-match if you have references, or qualitative review for tasks without canonical answers.
- **Task-specific harnesses** (MMLU, GSM8K, your AML benchmarks). Use `lm-eval-harness` or equivalent.
- **Qualitative inspection** on hand-curated AML prompts.

Whichever you choose, the model artifact lives at:

```
/sadata/swaminathanb/gsi-training/4.run_sft/run-v2/5.run_sft/checkpoints/LOWEST_VAL/model/consolidated/
```

Promote it as the final deliverable:

```bash
cp -r /sadata/swaminathanb/gsi-training/4.run_sft/run-v2/5.run_sft/checkpoints/LOWEST_VAL \
      /sadata/swaminathanb/gsi-training/sft-final
```

## Differences from the H200 reference

| | Reference (H200) | This (A100 SXM) | Why |
|---|---|---|---|
| `model.use_mamba_kernels` | `false` | `false` | identical (reference convention) |
| `model.torch_dtype` | `bfloat16` | `bfloat16` | identical |
| `fp8:` block | absent | `enabled: false` (explicit) | A100 has no FP8 hardware; explicit form prevents TE FP8-promotion surprises |
| `model.backend` | absent (uses defaults) | explicit `linear: te`, `experts: torch_mm`, `dispatcher: torch` | A100 needs torch dispatcher; DeepEP is Hopper-only |
| `distributed.dp_size` | `none` (auto-pick) | `8` (explicit) | H200 140 GB lets the framework skip EP; A100 80 GB needs explicit EP=8 |
| `distributed.ep_size` | absent | `8` | MoE 128 routed experts won't fit in 80 GB without sharding across the 8 GPUs |
| `distributed.activation_checkpointing` | `true` | `true` | identical (mandatory on A100 80 GB) |
| `optimizer.lr` | 1e-5 | 1e-5 | identical |
| `optimizer.betas` | `[0.9, 0.999]` | `[0.9, 0.999]` | identical (SFT convention) |
| `lr_scheduler` | cosine, warmup 100, min_lr 1e-6 | identical | identical |
| `packed_sequence_size` | 5350 | 5350 | identical |
| `dataset` | ChatDataset on path | identical | identical |
| `dataloader.num_workers` | 4 | 4 | identical (drop to 0 if pickle errors recur) |
| Container | `nvcr.io/nvidia/nemo-automodel:25.11` | `nvcr.io/nvidia/nemo-automodel:26.04` | matches the version we used for CPT |
| Checkpoint path | `/home/data/3.sft/run-v2/4.run_sft/checkpoints/` | `/workspace/4.run_sft/run-v2/5.run_sft/checkpoints/` | our workspace layout |
| Input model | `/home/data/2.cpt/run-v5/3.run_cpt/checkpoints/LOWEST_VAL/...` | `/workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/...` | our cpt-l2-final |
