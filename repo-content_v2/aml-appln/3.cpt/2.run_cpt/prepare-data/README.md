# CPT Data Preparation

Deterministic partition of the curated CPT corpus into the two folders the two CPT phases consume directly:

```
data/final/
├── level_1/<source>.jsonl     <-- Phase 1 trains on this (1 epoch)
└── level_2/<source>.jsonl     <-- Phase 2 trains on this (3-4 epochs)
```

This stage is **partition only** — no shuffling, no tokenization, no `.bin/.idx`. Those run downstream.

---

## 1. How to run

Pure stdlib Python (≥ 3.10). No GPU, no NeMo container, no extra dependencies.

```bash
cd /data/swami/gsi-training/3.cpt/2.run_cpt/prepare-data

python3 split_pools.py \
    --input_dir  ../data/raw \
    --output_dir ../data/final \
    --seed 42 \
    --workers 8
```

Re-running with the same `--seed` and `--edgar_l2_pct` is bit-for-bit reproducible; existing output files are overwritten.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--input_dir` | (required) | Curated raw root (the `level_1/` + `level_2/` parent directory). |
| `--output_dir` | (required) | Final partition root. Will create `level_1/`, `level_2/`. |
| `--seed` | `42` | Hash salt for deterministic per-doc routing. **Pin once and never change** unless you want to invalidate the EDGAR partition. |
| `--workers` | `min(8, cpu_count)` | Per-source parallel processes. EDGAR is the dominant cost; 8 lets it overlap with the small sources. |
| `--no_chunk_cap` | off | Disable the per-doc 5% cap. Keeps `uscode_house` etc. as single huge records — usually not what you want. |
| `--edgar_l2_pct` | `13` | Share of EDGAR documents (by count) routed to level_2 as Phase-2 replay. Drop to ~10% if Phase 2 over-trains; raise to ~18% if forgetting signal is weak. |

### Wall clock + outputs

End-to-end run takes ~100 s on a typical machine (EDGAR's 10 GB JSONL is the bottleneck). Outputs at `<output_dir>`:

```
data/final/
├── level_1/<source>.jsonl   8 files,  ~9.25 GB,  ~75 K docs   (Phase 1 input)
├── level_2/<source>.jsonl   9 files,  ~1.25 GB,  ~8 K docs    (Phase 2 input)
├── split_pools.log          full run log
└── split_summary.json       machine-readable manifest
```

Sanity-check after a run:

```bash
python3 -c "
import json
m = json.load(open('../data/final/split_summary.json'))
print(f'edgar_l2_pct={m[\"edgar_l2_pct\"]}%, seed={m[\"seed\"]}')
for p, v in m['pool_totals'].items():
    print(f'  {p:<10} -> {v[\"docs\"]:>8,} docs | {v[\"chars\"]:>14,} chars | {v[\"sources_with_records\"]} sources')
"
```

---

## 2. The split design

### Three flows, one partition pass

```
                  data/raw/level_1/                         data/raw/level_2/
                  +-----------------+                       +-----------------+
                  | edgar_corpus    |                       | ofac_guidance   |
                  | pile_of_law_*   |                       | fincen_*        |
                  | uscode_house    |                       | fatf_*          |
                  +-----------------+                       | courtlistener   |
                          |                                 +-----------------+
                          |                                          |
              +-----------+-----------+                              |
              |                       |                              |
              v                       v                              v
    +-------------------+   +-------------------+         +-------------------+
    | "rest" (=87% of   |   | "replay" (=13% of |         |   100% retained   |
    |  EDGAR + 100% of  |   |  EDGAR by doc id) |         |                   |
    |  small L1)        |   |                   |         |                   |
    +-------------------+   +-------------------+         +-------------------+
              |                       |                              |
              v                       +------------+-----------------+
              |                                    v
              v                          +-------------------+
   +--------------------+                |                   |
   | data/final/level_1 |                | data/final/level_2|
   |  ~9.25 GB / ~75 K  |                |  ~1.25 GB / ~8 K  |
   |       docs         |                |       docs        |
   +--------------------+                +-------------------+
   (Phase 1 input)                       (Phase 2 input;
                                          dataloader blends
                                          replay vs native L2)
```

So in plain English:

- **Level 1 raw data** is split into "**replay**" and "**rest**":
  - "rest" (87% of EDGAR + 100% of every small L1 source) **stays under `level_1/`**
  - "replay" (13% of EDGAR by document id) **goes under `level_2/`**
- **Level 2 raw data** stays 100% under `level_2/` (with the single exception of `enterprise_financial_crime`, which is dropped — 1 record, ~400 tokens; below useful threshold).

`enterprise_financial_crime` is the only source that's dropped entirely.

### Per-source routing table

| Source | level_1 | level_2 | Dropped |
|---|---:|---:|:---:|
| `edgar_corpus`                      | **87%** | **13%** (L1-replay) |   |
| `pile_of_law_oig`                   | 100% |  |   |
| `pile_of_law_federal_register`      | 100% |  |   |
| `pile_of_law_sec`                   | 100% |  |   |
| `pile_of_law_cfr`                   | 100% |  |   |
| `uscode_house`                      | 100% |  |   |
| `pile_of_law_doj_guidance`          | 100% |  |   |
| `pile_of_law_uscode`                | 100% |  |   |
| `ofac_guidance`                     |  | 100% |   |
| `fincen_federal_register`           |  | 100% |   |
| `fincen_sar_reviews`                |  | 100% |   |
| `fincen_advisories`                 |  | 100% |   |
| `fincen_enforcement`                |  | 100% |   |
| `courtlistener`                     |  | 100% |   |
| `fatf_publications`                 |  | 100% |   |
| `fincen_files`                      |  | 100% |   |
| `enterprise_financial_crime`        |  |  | ✓ |

**Why EDGAR is the only L1 source split.** EDGAR is 89% of L1's tokens — large enough to spare a meaningful share for Phase 2 replay. The 13% slice produces ~370 M EDGAR tokens of unseen-by-Phase-1 replay material; mixed with the 6× dataloader upsample of the 40 M native L2, Phase 2's per-epoch budget lands at ~540 M effective tokens.

**Why the small L1 sources stay 100% in level_1.** Their combined ~340 M tokens is too small to spare a replay share without hurting Phase 1's broad-register coverage.

### Two contracts the partition enforces

1. **Deterministic + disjoint.** Routing is `BLAKE2b(seed | doc_id) % 1000` mapped to per-source bucket ranges. Re-running with the same `--seed` is bit-identical. An EDGAR document never lands in both `level_1` and `level_2`.

2. **Per-doc 5% cap.** If the largest document in a source exceeds 5% of that source's total characters, every long document is paragraph-aware-chunked into pieces of at most 5%. Each chunk gets a fresh hash and is routed independently. Without this cap, `uscode_house` (2 records of ~3 M chars each), `pile_of_law_uscode` (2 records of ~2.5 M each), and several FinCEN/FATF mega-docs would each dominate their entire source. On by default; pass `--no_chunk_cap` to disable.

### Output JSONL schema

Each output line is a minimal training-shape record. All curator metadata (`lang_id`, `non_alpha_ratio`, `unique_line_ratio`, `_curator_dedup_id`, `element_type`, `page`, `metadata`, `phase`, `layer`, etc.) is dropped — none of it is consumed by the tokenizer or dataloader. `source` and `id` are preserved for downstream auditing and shard-level traceability.

```json
{"text": "...", "source": "edgar_corpus", "id": "cpt_25180bf6-...-_11355"}
```

Chunked sub-documents carry an `id` of the form `f"{parent_id}#chunk_{i}"`.

---

## 3. How NeMo blends `level_2` at training time

Once `level_2/` has been tokenized into `.bin/.idx` shards (a downstream step), Phase 2's dataloader uses NeMo's `MegatronPretraining` dataset to blend the EDGAR-replay shard with the eight native L2 shards. The blend mechanism is the load-bearing piece of the Phase 2 design, so it's worth understanding precisely.

### How weights are interpreted

> Weights are **sample-count ratios**, **not** per-step Bernoulli probabilities and **not** with-replacement sampling. They are normalized then realized as a **deterministic pre-computed index mapping** that picks exactly the right number of samples from each underlying dataset.
>
> So yes — with weights `[0.7, 0.3]` on two equally-sized shards, the model sees source-A samples ~7/3× as often per "epoch" (= per total-samples budget) as source-B. Because the underlying `GPTDataset` will silently **replay** its own documents to satisfy the requested count (it auto-computes `total number of epochs` per shard — see the `> total number of epochs: N` log lines), the small source is **upsampled by repetition** to hit its weight share.

The code that decides the sample-count split lives in [`BlendedDataset._build_indices` (megatron/core/datasets/blended_dataset.py, lines ~149–177)](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/blended_dataset.py):

```python
if self.size is not None:
    dataset_index = numpy.zeros(self.size, dtype=numpy.int16)
    dataset_sample_index = numpy.zeros(self.size, dtype=numpy.int64)
    helpers.build_blending_indices(
        dataset_index, dataset_sample_index,
        self.weights, len(self.datasets), self.size, _VERBOSE,
    )
else:
    size = sum(self.weights)
    helpers.build_exhaustive_blending_indices(
        dataset_index, dataset_sample_index, self.weights, len(self.datasets)
    )
```

The actual sampling routine is in [`megatron/core/datasets/helpers.cpp` lines 76–128](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/helpers.cpp) — a deterministic Bresenham-style **greedy max-error scheduler**: at every output slot it picks the dataset whose realized share is furthest below its target share. The result is essentially perfectly proportional sample counts (`current_samples[i] ≈ size * weights[i]`), with sequences from the two sources interleaved finely (not block-grouped), so every microbatch is approximately a weighted mixture. The two output arrays (`dataset_index`, `dataset_sample_index`) are persisted to `.npy` files under `index_mapping_dir` and reused across runs / ranks.

### What "epoch" means with blending

Two regimes:

- **Bounded mode** (`max_steps` is a positive int — our case): `size = max_steps × global_batch_size`. Megatron then allocates `≈ size × weights[i]` samples to shard *i*. Each underlying `GPTDataset` runs as many internal epochs over its `.bin/.idx` as needed to satisfy that allocation (you'll see `> total number of epochs: N` per shard in the log; small shards get N ≥ 2). This is the "epoch" — defined by `max_steps × global_batch_size`, **not** by any one shard. The `num_epochs: 1` field in the YAML is decorative under this regime.
- **Exhaustive mode** (`max_steps == -1`): one full pass through everything; weights become integer sample counts.

### Weights syntax in the YAML

There is **no separate `blend:` or `weights:` YAML key.** Weights are interleaved as strings inside the `paths` arg of `MegatronPretraining`. Strings are required (Megatron parses them as floats); `"44"`/`"56"` and `"0.44"`/`"0.56"` are equivalent.

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining
  paths:
    - "44"
    - /workspace/3.cpt/2.run_cpt/data/final/level_2/native_text_document
    - "56"
    - /workspace/3.cpt/2.run_cpt/data/final/level_2/edgar_replay_text_document
  index_mapping_dir: /workspace/3.cpt/2.run_cpt/data/final/level_2/_mapping
  ...
```

### Implication for our Phase 2

Native L2 is ~40 M tokens; EDGAR-replay is ~310 M tokens. Per [`../README.md`](../README.md) §3.3, Phase 2 wants ~44% native L2 / 56% EDGAR-replay per microstep so the AML domain gets its full share of compute despite being 10× smaller on disk.

With a typical Phase 2 budget of `max_steps × GBS` ≈ 540 M tokens / epoch × 4 epochs ≈ 2.16 B total samples, the `"44"` weight asks Megatron to fill ~44% of those slots from a 40 M-token shard — so the native-L2 `GPTDataset` reports `total number of epochs: ~24` for its 40 M-token shard while the EDGAR-replay `GPTDataset` reports `total number of epochs: ~4` for its 310 M-token shard. **You never write "6×" or "24×" anywhere — they fall out of the math.**

Per-source upsampling within native L2 (so OFAC's natural 45% share doesn't drown out FATF's 2.6%) means tokenizing each native L2 source into its own `.bin/.idx` shard and giving each its own weight (square-root-tempered per `../README.md` §3.2). Concretely, 8 native L2 paths + 1 EDGAR-replay path = 9-shard blend.

The fully-worked Phase 2 YAML lands with the downstream tokenizer step.

---

## 4. What is intentionally NOT in this script

| | Reason |
|---|---|
| Train / val holdout split | Done at training time via the NeMo dataloader's `split:` arg (the reference recipe used `"0.95, 0.05, 0.0"`). No separate holdout files. |
| Document shuffling (`terashuf`-style) | Done at the tokenizer / dataloader stage once the per-pool shard layout is final. |
| Tokenization + Megatron `.bin/.idx` packing | Handled downstream by `preprocess_megatron_dataset.py` with the FP8 base tokenizer + EOD insertion. |
| L2 per-source upsampling for Phase 2 | Done at training time via NeMo's blend-weight mechanism (§3 above). Native L2 records are written **once** on disk; the dataloader replays them as needed. |
| Length / quality / dedup / PII filters | The curator (`../../1.data_curation/`) already applied all of these. The partition is partition-only and trusts its input. |
