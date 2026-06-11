# M7 · Model Training

Workshop notebooks for the CPT → SFT → DPO training chain. All large artifacts
(checkpoints, HF caches, Docker layers) are directed to the spacious `/data`
volume so the root disk does not fill up.

## Notebooks

| Notebook | What it does | Runtime |
|----------|--------------|---------|
| `cpt.ipynb` | Continued pre-training (Megatron indexed dataset + NeMo AutoModel) | Docker (`nvcr.io/nvidia/nemo-automodel:26.04`) |
| `sft.ipynb` | Supervised fine-tuning on chat JSONL; chains on CPT checkpoint if present | Docker (same container) |
| `rl_dpo.ipynb` | DPO preference alignment (NeMo-RL `run_dpo.py`); `Qwen/Qwen2.5-0.5B` on 1× A100 | Docker (`nvcr.io/nvidia/nemo-rl:v0.6.0`) |
| `reference_rl_to_be_updated.ipynb` | Legacy LoRA DPO mini-run (host GPU, Qwen2.5-0.5B) | Host Python (uv `.venv`) |

Run notebooks from their own directory (e.g. `M7-model_training/`).

## Shared helpers

Both helpers live one level up in `workshop-Materials/`:

### `notebook_env.py` — per-notebook uv virtualenv

Used by host-side notebooks (DPO, M5, M6, M8, M9). The first code cell calls:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))
from notebook_env import bootstrap_notebook_env, ensure

bootstrap_notebook_env()
ensure("torch", ["torch>=2.5.0,<2.7.0"], quiet=True)
# ... more ensure() calls as needed
```

- Creates/reuses `.venv/` in the notebook folder.
- Installs packages with `uv pip` (no `pip` module required in the venv).
- Auto-installs `uv` if missing.

### `docker_storage.py` — Docker + workshop storage on `/data`

Used by Docker notebooks (CPT, SFT) and host notebooks that download models (DPO).
Call once in the first code cell:

```python
from docker_storage import (
    ensure_docker_storage,
    workshop_work_dir,
    docker_workspace_volumes,
)

ensure_docker_storage()
NB_DIR = Path.cwd().resolve()
WORK_DIR = workshop_work_dir("M7-model_training")
```

**`ensure_docker_storage()`** (idempotent):

- Migrates Docker `data-root` to `/data/docker` if needed.
- Sets cache env vars: `HF_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `TMPDIR`, …
- Logs to `/data/logs/docker_storage.log`.

**`workshop_work_dir(module_name)`** → `/data/workshop/{module}/work`

Writable training outputs: checkpoints, `hf_cache`, tokenized shards.

**`docker_workspace_volumes(nb_dir, work_dir)`** — split bind-mounts for containers:

| Host path | Container path | Mode |
|-----------|----------------|------|
| `{nb_dir}/data` | `/workspace/data` | read-only |
| `{nb_dir}/recipes` | `/workspace/recipes` | read-write |
| `{work_dir}` | `/workspace/work` | read-write |

Avoids mounting the entire notebook directory (which would write checkpoints to
the root disk).

**NeMo-RL note:** `rl_dpo.ipynb` runs the container as **root** (no `-u`) and uses
`/opt/nemo_rl_venv/bin/python` with `--entrypoint ""`. The NeMo-RL v0.6.0 venv is
not executable as the host user, and the default NVIDIA entrypoint fails with
`python: not found` (exit 127).

## `/data` layout

```
/data/
├── docker/              # Docker images, containers, overlay2
├── cache/
│   ├── hf/              # HuggingFace hub cache (host notebooks)
│   ├── uv/              # uv package cache
│   ├── pip/
│   └── tmp/
├── logs/
│   └── docker_storage.log
└── workshop/
    └── M7-model_training/
        └── work/
            ├── cpt_checkpoints/
            ├── cpt_data/
            ├── sft_checkpoints/
            ├── dpo_checkpoints/
            ├── rl_checkpoints/
            └── hf_cache/    # HF cache inside Docker runs
```

Legacy checkpoints may still exist under `M7-model_training/work/` on the root
disk. CPT/SFT notebooks fall back to those paths read-only when newer checkpoints
are not found on `/data`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DOCKER_DATA_ROOT` | `/data/docker` | Docker `data-root` migration target |
| `WORKSHOP_WORK_ROOT` | `/data/workshop` | Base for per-module `work/` dirs |
| `WORKSHOP_CACHE_ROOT` | `/data/cache` | Download and temp caches |
| `WORKSHOP_LOG_DIR` | `/data/logs` | Helper logs |
| `NGC_API_KEY` | (prompted) | Pull NeMo AutoModel container |
| `HF_TOKEN` | (prompted) | Gated Nemotron model access |
| `RAY_NUM_CPUS` | (optional) | Override Ray CPU count in M5 Curator |

## Training chain

1. **CPT** (`cpt.ipynb`) — domain-adapts the base model; writes `work/cpt_checkpoints/`.
2. **SFT** (`sft.ipynb`) — instruction-tunes on `data/sft_corpus.jsonl`; chains CPT if found; writes `work/sft_checkpoints/`.
3. **DPO** (`rl_dpo.ipynb`) — NeMo-RL preference alignment on `data/dpo_train.jsonl` + `data/dpo_val.jsonl` with `Qwen/Qwen2.5-0.5B` (production chains SFT `LOWEST_VAL` on 30B MoE); writes `work/dpo_checkpoints/`.

### DPO data files

| File | Format | Records |
|------|--------|--------:|
| `data/dpo_train.jsonl` | NeMo-RL PreferenceDataset (`context` + ranked `completions`) | 18 |
| `data/dpo_val.jsonl` | Same | 4 |
| `data/dpo_pairs.jsonl` | Legacy simple pairs (`system`/`user`/`chosen`/`rejected`) | 13 |

`dpo_train.jsonl` merges converted `dpo_pairs.jsonl` rows with condensed SDG-RL v3
task samples (`sar_judgment`, `auxiliary_*`). Production uses 38k train / 4.2k val.

> **Note:** `reference_rl_to_be_updated.ipynb` is a separate legacy host-GPU LoRA
> mini-run (Qwen2.5-0.5B) kept for comparison; the production-aligned path is
> `rl_dpo.ipynb`.
