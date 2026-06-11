# Stage 3 - Filter SFT JSONL by token length + JSON-sanitize

Two scripts here, run in this order:

1. `filter_data.py` - drops records that exceed a token-length budget, using the **target SFT model's tokenizer** (so the filter matches the recipe's actual encoding path).
2. `rebuild_sft_jsonl.py` - re-serializes JSONL to escape Unicode line separators and other control characters that ChatDataset's `splitlines()` reader would otherwise miscount.

The output of stage 3 is what the recipe in stage 4 reads (`dataset.path_or_dataset_id` and `validation_dataset.path_or_dataset_id`).

## Step A - filter by token length

```bash
cd /sadata/swaminathanb/gsi-training/4.run_sft/3.filter_dataset

# In container (loads the same tokenizer SFT will use)
docker run --rm \
  -e HF_HOME=/workspace/4.run_sft/4.run_sft/hf_cache \
  -e HF_TOKEN=$HF_TOKEN \
  -v /sadata/swaminathanb/gsi-training:/workspace \
  --workdir /workspace/4.run_sft/3.filter_dataset \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 filter_data.py \
    --input_dir  /workspace/4.run_sft/1.shuffle_dataset/data \
    --output_dir /workspace/4.run_sft/3.filter_dataset/final_data \
    --model_path /workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated \
    --max_length 5120 \
    --log_file   /workspace/4.run_sft/3.filter_dataset/filtered_data.log
```

Output: `final_data/sft_mixed.chunk.00.jsonl`, `final_data/sft_mixed.val.jsonl`, `final_data/sft_mixed.test.jsonl` (or whichever subset of those exist in `--input_dir`).

What gets dropped: records where `tokenizer(text)` returns more than `--max_length - 100` tokens. The 100-token buffer compensates for the chat-template overhead (BOS / role prefixes / EOS) that gets added at training time but isnt counted by the raw `tokenizer(...)` call.

A summary log lands in `--log_file`:

```
Total Records: ...
Kept Records:  ...
Dropped Records: ...
Drop Rate: ...%
Global Max Length Found: ...
Top Longest Samples:
 - <length> tokens in <file>
 - ...
```

If drop rate is > 5%, your `--max_length` is too tight relative to the corpus distribution -- go back to stage 2's percentile report and re-pick.

## Step B - JSON sanitation rebuild

ChatDataset reads JSONL via `splitlines()`, which respects Unicode line separators (`U+2028`, `U+2029`) as line breaks. If a chat message body contains those characters, ChatDataset will silently split a single record in half and produce a malformed batch (or skip silently). This rebuild escapes them.

```bash
docker run --rm \
  -v /sadata/swaminathanb/gsi-training:/workspace \
  --workdir /workspace/4.run_sft/3.filter_dataset \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python3 rebuild_sft_jsonl.py \
    /workspace/4.run_sft/3.filter_dataset/final_data \
    /workspace/4.run_sft/3.filter_dataset/final_data_clean \
    --on-fail               skip \
    --workers               16 \
    --ensure-last-assistant
```

Output goes to `final_data_clean/`. Same filenames as input (the script knows to look for `sft_mixed.{chunk.00,val,test}.jsonl`). A `<file>.bad.tsv` is written alongside each output containing the line numbers and reasons for any dropped records.

After this completes, **point the recipe at `final_data_clean/`** (or symlink `final_data` -> `final_data_clean`).

### What `--ensure-last-assistant` does

Drops trailing non-assistant turns so each record's last message has `role: assistant`. Required because:
- The chat template appends the assistant turn as the **target** for next-token-prediction loss.
- A record ending in `user` has no target -- the trainer either crashes or computes loss against an empty string.

If you ran `data/check_format_fix.py` upstream, this flag is redundant (already enforced there). Keeping it on as a safety net costs nothing.

### `--on-fail` choices

| Value | Behavior |
|---|---|
| `skip` (default) | Drop unparseable records. Stats log them; final JSONL doesn't contain them. |
| `keep_raw` | Wrap each unparseable record as `{"_raw": "...", "_error": "..."}` in the output. ChatDataset will skip them at load time, but they're preserved for debugging. |

## Arguments - filter_data.py

| Arg | Default | Meaning |
|---|---|---|
| `--input_dir` | `../1.shuffle_dataset/data` | Where the stage-1 chunks live |
| `--output_dir` | `./final_data` | Where filtered chunks land |
| `--log_file` | `filtered_data.log` | Where the summary log writes |
| `--model_path` | `/workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated` | Tokenizer source. **Must match the recipe's `pretrained_model_name_or_path`** |
| `--max_length` | `5120` | Drop records whose `tokenizer(text)` length > `max_length - 100` (the 100 is chat-template padding). Should equal `packed_sequence_size` from the recipe minus a small buffer. |

## Arguments - rebuild_sft_jsonl.py

Positional: `<input>` `<output>`. Both can be files or directories. If directories, the script processes the canonical SFT names (`sft_mixed.chunk.00.jsonl`, `sft_mixed.test.jsonl`, `sft_mixed.val.jsonl`).

| Flag | Default | Meaning |
|---|---|---|
| `--on-fail` | `skip` | `skip` or `keep_raw` (see above). |
| `--workers` | `min(32, cpu_count())` | Parallel rebuild workers. |
| `--chunk-lines` | `20000` | Lines per task sent to a worker. |
| `--multiline` | off | Try to assemble JSON objects that span multiple lines (turns off multiprocessing -- only for severely broken inputs). |
| `--max-buffer-lines` | `5000` | Cap on how many lines an in-progress object can span (multiline mode only). |
| `--ensure-last-assistant` | off | Drop records whose last message isn't `assistant` (recommended). |
