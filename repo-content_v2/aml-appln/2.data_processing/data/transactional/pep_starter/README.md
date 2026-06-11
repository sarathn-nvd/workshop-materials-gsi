# Synthetic PEP Starter Pool

Synthetic placeholder for the PEP (Politically Exposed Person) name pool used
by Stage 4 of the SFT synthetic data generation pipeline (see
`sgd_understanding.md`). To be replaced once OpenSanctions consolidated PEP
data is downloaded into a separate pool.

## Provenance

- **Generator**: `2.data_processing/generate_pep_starter.py`
- **Seed**: `42` (deterministic)
- **Entries**: `200`
- **Schema**: matches `2.data_processing/data/transactional/ofac_enforcement/targets.simple.csv`
  so Stage 4 can union the two pools transparently.
- **Names**: synthetic permutations of common regional first names and
  surnames. Not real people.
- **Tagging**: `dataset` column = `"Synthetic PEP Starter v1"`; `id` prefix =
  `"PEP-SYN-"`.

## Distribution

- ~95% `Person`, ~5% `Company` (state-owned enterprises).
- Country mix weighted toward high-risk jurisdictions
  (Russia, China, Venezuela, Iran, Nigeria, Pakistan, Kazakhstan, etc.)
  with EU / US / Latin America baseline.
- Role mix (`program_ids`):
  HoS, Minister, Legislator, Family-PEP, Close-Assoc,
  Judge-Senior, Military-Senior, CB-Regulator, SOE-Exec, IO-Official.

## Regenerating

```bash
cd 2.data_processing
python generate_pep_starter.py
```
