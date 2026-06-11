# SFT Corpus — Stratified Train/Val Distribution

Companion to `../raw/distribution.md`. Reports the realized distribution of the **stratified** split produced by `../stratified_split.py`. Every per-stratum table mirrors the categorical axes documented in the raw `distribution.md` so the two files can be diffed side by side.


Generated from:
- raw:   `/data/swami/gsi-training/7.run_sft/1.shuffle_dataset/data/raw`
- split: `/data/swami/gsi-training/7.run_sft/1.shuffle_dataset/data/final`


## Files

| File | Records | Size |
|---|---:|---:|
| `sft_mixed.chunk.00.jsonl` | 38,494 | 589.3 MB |
| `sft_mixed.val.jsonl` | 4,291 | 65.9 MB |
| **TOTAL** | **42,785** | **655.2 MB** |


## Top-level split (sar_judgment vs auxiliary_*)

| Group | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| sar_judgment | 28,551 | 3,182 | 31,733 | 74.17% | 74.16% | 74.17% | 10.03% |
| auxiliary_* | 9,943 | 1,109 | 11,052 | 25.83% | 25.84% | 25.83% | 10.03% |
| **Total** | **38,494** | **4,291** | **42,785** | 100.00% | 100.00% | 100.00% | 10.03% |


## Variant mix (`sar_judgment` only)

| Variant | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial_aux | 376 | 50 | 426 | 1.32% | 1.57% | 1.34% | 11.74% |
| augmented | 10,878 | 1,219 | 12,097 | 38.10% | 38.31% | 38.12% | 10.08% |
| bare | 17,297 | 1,913 | 19,210 | 60.58% | 60.12% | 60.54% | 9.96% |
| **Total** | **28,551** | **3,182** | **31,733** | 100.00% | 100.00% | 100.00% | 10.03% |


## Label mix (`sar_judgment` only)

| Label | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| POS | 3,813 | 427 | 4,240 | 13.36% | 13.42% | 13.36% | 10.07% |
| NEG | 24,738 | 2,755 | 27,493 | 86.64% | 86.58% | 86.64% | 10.02% |
| **Total** | **28,551** | **3,182** | **31,733** | 100.00% | 100.00% | 100.00% | 10.03% |


## Typology distribution (`sar_judgment` only)

| Typology | Train | Val | Total | Train POS | Train NEG | Val POS | Val NEG | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| elder_exploitation | 1,485 | 166 | 1,651 | 923 | 562 | 103 | 63 | 10.05% |
| human_trafficking | 159 | 18 | 177 | 98 | 61 | 11 | 7 | 10.17% |
| layering | 574 | 65 | 639 | 345 | 229 | 39 | 26 | 10.17% |
| none | 22,276 | 2,476 | 24,752 | 0 | 22,276 | 0 | 2,476 | 10.00% |
| shell_company | 499 | 57 | 556 | 301 | 198 | 34 | 23 | 10.25% |
| smurfing | 799 | 90 | 889 | 481 | 318 | 54 | 36 | 10.12% |
| structuring | 1,620 | 181 | 1,801 | 981 | 639 | 109 | 72 | 10.05% |
| terrorist_financing | 445 | 51 | 496 | 265 | 180 | 30 | 21 | 10.28% |
| trade_based_ml | 694 | 78 | 772 | 419 | 275 | 47 | 31 | 10.10% |


## Source distribution (`sar_judgment` only)

| Source | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| amlgentex | 67 | 12 | 79 | 0.23% | 0.38% | 0.25% | 15.19% |
| cfpb | 10,267 | 1,092 | 11,359 | 35.96% | 34.32% | 35.80% | 9.61% |
| efc | 1,323 | 149 | 1,472 | 4.63% | 4.68% | 4.64% | 10.12% |
| ibm | 2,165 | 274 | 2,439 | 7.58% | 8.61% | 7.69% | 11.23% |
| sarsum_neg | 1,814 | 207 | 2,021 | 6.35% | 6.51% | 6.37% | 10.24% |
| synth:elder_exploitation | 1,485 | 166 | 1,651 | 5.20% | 5.22% | 5.20% | 10.05% |
| synth:human_trafficking | 159 | 18 | 177 | 0.56% | 0.57% | 0.56% | 10.17% |
| synth:layering | 79 | 7 | 86 | 0.28% | 0.22% | 0.27% | 8.14% |
| synth:none_clean | 5,658 | 628 | 6,286 | 19.82% | 19.74% | 19.81% | 9.99% |
| synth:none_near_miss | 2,474 | 291 | 2,765 | 8.67% | 9.15% | 8.71% | 10.52% |
| synth:shell_company | 499 | 57 | 556 | 1.75% | 1.79% | 1.75% | 10.25% |
| synth:smurfing | 636 | 73 | 709 | 2.23% | 2.29% | 2.23% | 10.30% |
| synth:structuring | 1,166 | 121 | 1,287 | 4.08% | 3.80% | 4.06% | 9.40% |
| synth:terrorist_financing | 445 | 51 | 496 | 1.56% | 1.60% | 1.56% | 10.28% |
| synth:trade_based_ml | 314 | 36 | 350 | 1.10% | 1.13% | 1.10% | 10.29% |
| **Total** | **28,551** | **3,182** | **31,733** | 100.00% | 100.00% | 100.00% | 10.03% |


## Auxiliary task mix

| Task | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| auxiliary_behavioral | 4,382 | 488 | 4,870 | 44.07% | 44.00% | 44.06% | 10.02% |
| auxiliary_citation | 2,995 | 334 | 3,329 | 30.12% | 30.12% | 30.12% | 10.03% |
| auxiliary_numeric | 2,194 | 245 | 2,439 | 22.07% | 22.09% | 22.07% | 10.05% |
| auxiliary_statutory | 372 | 42 | 414 | 3.74% | 3.79% | 3.75% | 10.14% |
| **Total** | **9,943** | **1,109** | **11,052** | 100.00% | 100.00% | 100.00% | 10.03% |


## Auxiliary source distribution (per task)


### `auxiliary_behavioral` (4,870 records)

| Source | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| amlgentex | 53 | 6 | 59 | 1.21% | 1.23% | 1.21% | 10.17% |
| enterprise_fc | 408 | 46 | 454 | 9.31% | 9.43% | 9.32% | 10.13% |
| ibm_aml | 3,921 | 436 | 4,357 | 89.48% | 89.34% | 89.47% | 10.01% |
| **Total** | **4,382** | **488** | **4,870** | 100.00% | 100.00% | 100.00% | 10.02% |


### `auxiliary_citation` (3,329 records)

| Source | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| efc | 174 | 20 | 194 | 5.81% | 5.99% | 5.83% | 10.31% |
| ffiec | 2,421 | 269 | 2,690 | 80.83% | 80.54% | 80.81% | 10.00% |
| sarsum | 400 | 45 | 445 | 13.36% | 13.47% | 13.37% | 10.11% |
| **Total** | **2,995** | **334** | **3,329** | 100.00% | 100.00% | 100.00% | 10.03% |


### `auxiliary_numeric` (2,439 records)

| Source | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| financebench | 134 | 15 | 149 | 6.11% | 6.12% | 6.11% | 10.07% |
| finqa | 0 | 1 | 1 | 0.00% | 0.41% | 0.04% | 100.00% |
| tat_qa | 2,060 | 229 | 2,289 | 93.89% | 93.47% | 93.85% | 10.00% |
| **Total** | **2,194** | **245** | **2,439** | 100.00% | 100.00% | 100.00% | 10.05% |


### `auxiliary_statutory` (414 records)

| Source | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| legalbench | 211 | 24 | 235 | 56.72% | 57.14% | 56.76% | 10.21% |
| synthetic_aml_statute | 161 | 18 | 179 | 43.28% | 42.86% | 43.24% | 10.06% |
| **Total** | **372** | **42** | **414** | 100.00% | 100.00% | 100.00% | 10.14% |


## Stratification verification (val % per stratum)

Per-stratum val sampling rate target was 10% (per-stratum `ceil` so every non-empty stratum gets >=1 val record). Smallest cells over-sample slightly by design.


| Stratum | Train | Val | Total | Val % of stratum |
|---|---:|---:|---:|---:|
| auxiliary_behavioral | amlgentex | 53 | 6 | 59 | 10.17% |
| auxiliary_behavioral | enterprise_fc | 408 | 46 | 454 | 10.13% |
| auxiliary_behavioral | ibm_aml | 3,921 | 436 | 4,357 | 10.01% |
| auxiliary_citation | efc | 174 | 20 | 194 | 10.31% |
| auxiliary_citation | ffiec | 2,421 | 269 | 2,690 | 10.00% |
| auxiliary_citation | sarsum | 400 | 45 | 445 | 10.11% |
| auxiliary_numeric | financebench | 134 | 15 | 149 | 10.07% |
| auxiliary_numeric | finqa | 0 | 1 | 1 | 100.00% |
| auxiliary_numeric | tat_qa | 2,060 | 229 | 2,289 | 10.00% |
| auxiliary_statutory | legalbench | 211 | 24 | 235 | 10.21% |
| auxiliary_statutory | synthetic_aml_statute | 161 | 18 | 179 | 10.06% |
| sar_judgment | elder_exploitation | NEG | 562 | 63 | 625 | 10.08% |
| sar_judgment | elder_exploitation | POS | 923 | 103 | 1,026 | 10.04% |
| sar_judgment | human_trafficking | NEG | 61 | 7 | 68 | 10.29% |
| sar_judgment | human_trafficking | POS | 98 | 11 | 109 | 10.09% |
| sar_judgment | layering | NEG | 229 | 26 | 255 | 10.20% |
| sar_judgment | layering | POS | 345 | 39 | 384 | 10.16% |
| sar_judgment | none | NEG | 22,276 | 2,476 | 24,752 | 10.00% |
| sar_judgment | shell_company | NEG | 198 | 23 | 221 | 10.41% |
| sar_judgment | shell_company | POS | 301 | 34 | 335 | 10.15% |
| sar_judgment | smurfing | NEG | 318 | 36 | 354 | 10.17% |
| sar_judgment | smurfing | POS | 481 | 54 | 535 | 10.09% |
| sar_judgment | structuring | NEG | 639 | 72 | 711 | 10.13% |
| sar_judgment | structuring | POS | 981 | 109 | 1,090 | 10.00% |
| sar_judgment | terrorist_financing | NEG | 180 | 21 | 201 | 10.45% |
| sar_judgment | terrorist_financing | POS | 265 | 30 | 295 | 10.17% |
| sar_judgment | trade_based_ml | NEG | 275 | 31 | 306 | 10.13% |
| sar_judgment | trade_based_ml | POS | 419 | 47 | 466 | 10.09% |
| **TOTAL** | **38,494** | **4,291** | **42,785** | **10.03%** |


## Schema in the split files

After stratified_split.py, only the `messages` field is retained -- metadata was stripped to match the `ChatDataset` input format used by the SFT recipe (`4.run_sft/4.run_sft/recipe_a100sxm-8.yaml`):

```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
