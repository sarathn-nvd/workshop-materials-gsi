# Stage 2 - Analyze token length distribution

Tokenizes every `messages` array in the stage-1 output (via `apply_chat_template`) and reports the token-count distribution. This stage does not modify the data; it informs the `--max_length` choice for stage 3 and the `packed_sequence_size` choice for stage 4.

## Why it matters

The SFT recipe uses sequence packing (`packed_sequence_size: 5350`). The packing size needs to be large enough to fit any single record (else the trainer drops or splits it) and small enough that short records pack two-per-pack to maximize GPU utilization. The percentile report tells you where to set the knob.

## Usage

```bash
cd /sadata/swaminathanb/gsi-training/4.run_sft/2.analyse_dataset

docker run --rm \
  -e HF_HOME=/workspace/4.run_sft/4.run_sft/hf_cache \
  -e HF_TOKEN=$HF_TOKEN \
  -v /sadata/swaminathanb/gsi-training:/workspace \
  --workdir /workspace/4.run_sft/2.analyse_dataset \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 analyse_dataset.py \
    --input_dir      /workspace/4.run_sft/1.shuffle_dataset/data \
    --tokenizer_path /workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated \
    --workers        32
```

## Output: `corpus_details.log`

Total tokens, record count, mean tokens/record, p10..p99.9 percentiles, plus per-bucket min/max/count.

## Reading the report

| Observation | Decision |
|---|---|
| p99 <= 5350 and p99.9 <= 8000 | Default recipe `packed_sequence_size: 5350` is fine. Set stage-3 `--max_length 5120`. |
| p99 > 5350 but p99.9 <= 8000 | Bump `packed_sequence_size` to ~p99 * 1.1 and `--max_length` to match. |
| p99 > 8000 | Long-context regime. Either filter to <=8000 or push packing to ~16384 (memory-expensive on A100; verify LBS=1 fits). |
| p50 < 1000 and p99 < 3000 | Drop `packed_sequence_size` to ~3072 to pack more records per pack, lift throughput. |

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--input_dir` | required | Dir of .jsonl files (point at `../1.shuffle_dataset/data/`) |
| `--tokenizer_path` | required | Same model the recipe uses; either local consolidated checkpoint or HF id |
| `--output_file` | `corpus_details.log` | Output filename next to the script |
| `--workers` | `cpu_count() - 4` | Parallel tokenizer workers |
