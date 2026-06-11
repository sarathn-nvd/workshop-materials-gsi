"""One-time artifact builders. Run BEFORE the main pipeline.

These produce reusable artifacts that the per-record pipeline reads:
- `build_policy_chunks` → `data/tools/policy_chunks.parquet`  (Stage 5A)
- `synthesize_sops`     → `data/tools/sops/{typology}_v{1|2|3}.md`  (Stage 5B)
"""
