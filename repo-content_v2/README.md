# GenAI Workshop Materials (v2)

Portable workshop source for a fresh GPU machine. Runtime artifacts (`.venv/`, `work/`, Docker layers, checkpoints) are **not** included — notebooks recreate them on first run.

## Layout

- `workshop-Materials/` — notebooks, small seed data, recipes, shared helpers
- See `workshop-Materials/M7-model_training/README.md` for CPT → SFT → DPO chain

## New machine setup

1. Clone/copy this tree (e.g. under `/home/ubuntu/repo-content_v2`).
2. Ensure a spacious `/data` volume exists for Docker and checkpoints (`docker_storage.py`).
3. Open notebooks from their module folders and run cells in order.
4. Provide **NGC API key** and **HF token** when prompted (not stored in repo).

## Shared helpers

| File | Purpose |
|------|---------|
| `workshop-Materials/notebook_env.py` | Per-notebook `.venv` via `uv` |
| `workshop-Materials/docker_storage.py` | Docker data-root + `/data/workshop` work dirs |
