# 4. SFT pipeline

End-to-end supervised fine-tuning of the Phase-2 CPT endpoint (`run-v2/3.run_cpt/checkpoints/LOWEST_VAL/` = `cpt-l2-final`) on instruction-format chat data, producing the SFT-final checkpoint that downstream (RLHF / deployment) pipelines consume.

| | |
|---|---|
| Phase | 1 of 1 (single-pass SFT for now; can be extended to multi-pass like CPT was if needed) |
| Base checkpoint | `../run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated/` (= `cpt-l2-final`) |
| Input data | Raw JSONL chat records under `data/raw/` (or `data/fixed/` after running `data/check_format_fix.py`) |
| Output checkpoint | `4.run_sft/checkpoints/LOWEST_VAL/model/consolidated/` (= `sft-final`) |
| Hardware | 8 x A100 SXM (80 GB / GPU, NVSwitch full mesh) -- same node used for CPT |
| Container | `nvcr.io/nvidia/nemo-automodel:26.04` |
| Wall-clock estimate | depends on corpus size; for a typical 100K-record SFT corpus with `packed_sequence_size: 5350` and 1 epoch, ~6-10 h on A100 SXM |

The H200 reference at `reference/run-v1/4.run_sft/` and `reference/run-v2/4.run_sft/` was the structural starting point. This pipeline mirrors the v2 reference's stage layout (shuffle -> analyse -> filter -> train) and applies A100-specific overrides documented in [`4.run_sft/recipe_a100sxm-8.yaml`](4.run_sft/recipe_a100sxm-8.yaml) (FP8 disabled, EP=8, explicit dp_size=8, torch_mm experts, torch dispatcher).

---

## Pipeline overview

```
   data/raw/   <-- you put raw chat-format JSONL here
       |
       |  (optional but recommended)
       |  data/check_format_fix.py   -- drops malformed records, enforces last-message=assistant,
       |                                truncates by token-count ceiling
       v
   data/fixed/
       |
       |  1.shuffle_dataset/shuffle.py   -- normalize to {messages: [...]}, terashuf globally,
       |                                    split into chunks + val + test
       v
   1.shuffle_dataset/data/sft_mixed.{chunk.NN, val, test}.jsonl
       |
       +-- 2.analyse_dataset/analyse_dataset.py   -- token percentile report
       |    (read-only; informs --max_length and packed_sequence_size choices)
       |
       v
   3.filter_dataset/filter_data.py     -- drop records exceeding --max_length tokens
       |
       v
   3.filter_dataset/final_data/sft_mixed.{chunk.NN, val, test}.jsonl
       |
       |  3.filter_dataset/rebuild_sft_jsonl.py  -- escape Unicode line separators,
       |                                             enforce last-message=assistant
       v
   3.filter_dataset/final_data_clean/sft_mixed.{chunk.NN, val, test}.jsonl
       |
       v
   4.run_sft/recipe_a100sxm-8.yaml + finetune.py   -- the actual SFT training
       |
       v
   4.run_sft/checkpoints/{epoch_N_step_*/, LOWEST_VAL/}  ->  sft-final
```

## Directory layout

```
4.run_sft/
|-- README.md                                        (this file)
|-- reference/                                       (the H200 reference -- left untouched)
|   |-- data/check_format_fix.py
|   |-- run-v1/{1.shuffle_dataset, 2.analyse_dataset, 3.filter_dataset, 4.run_sft}/
|   `-- run-v2/{1.shuffle_dataset, 2.analyse_dataset, 3.filter_dataset, 4.run_sft}/
|-- data/                                            (raw + format-fix area)
|   |-- README.md
|   |-- check_format_fix.py
|   |-- raw/                                         (you create this; drop raw JSONL here)
|   `-- fixed/                                       (auto-created by check_format_fix.py)
|-- 1.shuffle_dataset/
|   |-- README.md
|   |-- shuffle.py                                   (copied from reference/run-v2, unchanged)
|   |-- terashuf/                                    (auto-cloned + built on first run)
|   `-- data/                                        (output: sft_mixed.{chunk.NN, val, test}.jsonl)
|-- 2.analyse_dataset/
|   |-- README.md
|   |-- analyse_dataset.py                           (copied from reference/run-v2, unchanged)
|   `-- corpus_details.log                           (auto-created on run)
|-- 3.filter_dataset/
|   |-- README.md
|   |-- filter_data.py                               (adapted: --model_path now CLI flag, default points at our cpt-l2-final)
|   |-- rebuild_sft_jsonl.py                         (copied from reference/run-v2, unchanged)
|   |-- final_data/                                  (auto-created by filter_data.py)
|   |-- final_data_clean/                            (auto-created by rebuild_sft_jsonl.py)
|   `-- filtered_data.log                            (auto-created on run)
`-- 4.run_sft/
    |-- README.md
    |-- recipe_a100sxm-8.yaml                        (NEW -- A100 SFT recipe, adapted from H200 reference)
    |-- finetune.py                                  (copied from ../../3.run_cpt/pretrain.py, unchanged
    |                                                 -- same TrainFinetuneRecipeForNextTokenPrediction class works for SFT)
    |-- run.log                                      (auto-created at launch)
    |-- mapping_dir/                                 (NeMo dataloader index cache; auto-populated)
    |-- hf_cache/                                    (HF tokenizer cache; auto-populated)
    `-- checkpoints/                                 (auto-created at first ckpt write)
        |-- epoch_0_step_999/
        |-- epoch_0_step_1999/
        |-- ...
        |-- LATEST           -> epoch_N_step_X
        `-- LOWEST_VAL       -> epoch_N_step_X
```

## End-to-end run order

1. **Drop your raw SFT JSONL files** into `data/raw/`. Each line must be a JSON object with a `messages` array (`role`/`content` per message). Other fields are ignored by the shuffle stage.

2. **(Optional) `data/check_format_fix.py`** -- catches malformed records, last-message-not-assistant, and token-overrun upfront. Produces `data/fixed/`. See [`data/README.md`](data/README.md) for the command. Skip this only if your raw data is already clean.

3. **`1.shuffle_dataset/shuffle.py`** -- normalizes to bare `{messages: [...]}`, globally shuffles, splits chunk + val + test. See [`1.shuffle_dataset/README.md`](1.shuffle_dataset/README.md). Output lands in `1.shuffle_dataset/data/`.

4. **`2.analyse_dataset/analyse_dataset.py`** -- read-only token-length percentile report. Use the output to confirm or tune `packed_sequence_size` (in stage 4 recipe) and `--max_length` (in stage 3). See [`2.analyse_dataset/README.md`](2.analyse_dataset/README.md).

5. **`3.filter_dataset/filter_data.py`** then **`3.filter_dataset/rebuild_sft_jsonl.py`** -- length filtering and JSON sanitation. Output lands in `3.filter_dataset/final_data_clean/`. See [`3.filter_dataset/README.md`](3.filter_dataset/README.md).

6. **`4.run_sft/recipe_a100sxm-8.yaml`** + `finetune.py` -- the actual SFT training. See [`4.run_sft/README.md`](4.run_sft/README.md) for the launch block. Output lands in `4.run_sft/checkpoints/LOWEST_VAL/`.

## Comparison to CPT Phase 1 + Phase 2

| | CPT Phase 1 (`3.run_cpt/`) | CPT Phase 2 (`run-v2/3.run_cpt/`) | SFT (`4.run_sft/4.run_sft/`) |
|---|---|---|---|
| Dataset class | `MegatronPretraining` (binary `.bin`/`.idx`) | `MegatronPretraining` weighted blend | `ChatDataset` (line-oriented JSONL) |
| Pre-stages | tokenize via `2.prepare_megatroncore_dataset/` | weighted blend in recipe `dataset.paths` | shuffle + analyse + filter (4 stages total here) |
| Sequence length | fixed 4096 | fixed 4096 | variable (up to `packed_sequence_size: 5350`, with packing) |
| Batch size | GBS=256, LBS=2 | GBS=128, LBS=2 | GBS=16, LBS=1 |
| LR schedule | WSD (warmup 100 + stable 1270 + decay 430) | cosine (warmup 50, decay 2000) | cosine (warmup 100, decay auto) |
| Peak LR | 2e-5 | 5e-6 | 1e-5 |
| Optimizer betas | (0.9, **0.95**) | (0.9, **0.95**) | (0.9, **0.999**) <-- SFT convention |
| `use_mamba_kernels` | true | true | **false** <-- packing path interaction |
| Wall-clock | ~11.5 h | ~11.5 h | ~6-10 h (depends on data size) |
| Output gate | per-source PPL drop >= 15% (CPT objective) | per-source PPL drop on AML + L1 forgetting <= 10% | held-out chat completion eval (qualitative + harnesses; not in scope here) |

## Notes carried forward from CPT runs

The lessons we learned in CPT also apply here. From experience:

- **Run from `/sadata/swaminathanb/gsi-training/`**, not from the GPU node's home dir. The bind-mount `-v $HOST_WORKSPACE:/workspace` requires this exact path.
- **`HF_TOKEN` must be exported** in the launching shell. Even though SFT loads model weights from local disk, the tokenizer is still pulled from the gated NVIDIA HF repo at first launch.
- **Container is rootless on this cluster**. Drop `--user`/`--group-add 0` from the docker run block (they break bind-mount file ownership under rootless).
- **`--ulimit stack=...` removed** from CPT recipes because rootless can't raise `RLIMIT_STACK`. Keep only `--ulimit memlock=-1` here.
- **First-launch overhead is significant** (~3-5 min for HF tokenizer + Megatron helpers compile + dataset index build). Subsequent re-launches are < 1 min.

If you hit anything unexpected, `Automodel/` contains the source clone for grepping internals (e.g. `nemo_automodel/recipes/llm/train_ft.py` for the training-loop logic, `nemo_automodel/components/datasets/llm/chat_dataset.py` for the SFT data path).
