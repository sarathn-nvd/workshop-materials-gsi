# SFT raw-data drop point + format check

Place your raw chat-format JSONL files under `data/raw/` (or wherever you prefer; pass via `--input_dir`).
Each record must be a single line of JSON with a `messages` array compatible with `nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset`:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The downstream stages (`1.shuffle_dataset/`, `3.filter_dataset/`) only look at the `messages` key — any other fields (`source`, `id`, `metadata`, etc.) are discarded by the shuffle stage's normalizer.

## When to run `check_format_fix.py` (optional, recommended for noisy raw data)

This is a **pre-cleanup pass** that catches the most common SFT data problems *before* you tokenize / filter / shuffle:

1. Lines that aren't valid JSON.
2. Records missing the `messages` key, or with `messages` not a list.
3. Per-message records missing `role` or `content`.
4. Records whose **last message isn't from `assistant`** (these confuse the next-token-prediction loss — the chat template would put the assistant's response as the prediction target, but here there is none).
5. Records that exceed a token-count ceiling (default 15,000) — these would either be dropped by `3.filter_dataset/` or, worse, silently truncated by the tokenizer mid-conversation.

Records that fail any of those are **dropped** in the output file (the input is read-only). A summary log is written to `check_and_clean_log.txt` next to the script.

Run from this directory:

```bash
cd /sadata/swaminathanb/gsi-training/4.run_sft/data

python3 check_format_fix.py \
  --input_dir   ./raw \
  --output_dir  ./fixed \
  --tokenizer_path /sadata/swaminathanb/gsi-training/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated \
  --max_tokens  15000 \
  --workers     32
```

After this completes, point `1.shuffle_dataset/shuffle.py --input_dir` at `./fixed/` instead of `./raw/`.

## Skipping `check_format_fix.py`

If you trust your raw data is already clean (e.g. it came from an SFT-aware preprocessing pipeline upstream, not a scrape), you can point `1.shuffle_dataset/shuffle.py` directly at `./raw/`. The shuffle stage's inline normalizer drops malformed lines silently, so the only risk of skipping is silently losing data without a log.

The full SFT pipeline order is:

```
data/raw/   →  [optional] check_format_fix.py  →  data/fixed/
                                                       │
            1.shuffle_dataset/shuffle.py  ◄────────────┘
                       │
                       ▼
            1.shuffle_dataset/data/sft_mixed.{chunk.NN,val,test}.jsonl
                       │
            ┌──────────┴──────────────────────────┐
            ▼                                     ▼
   2.analyse_dataset/analyse_dataset.py    3.filter_dataset/filter_data.py
   (token-stats report only;               (drops records > MAX_LENGTH;
    informs MAX_LENGTH choice)              writes 3.filter_dataset/final_data/)
                                                       │
                                                       ▼
                                            4.run_sft/recipe_a100sxm-8.yaml
                                            (training reads final_data/)
```
