# Backend deployment bundle (self-contained)

Copy **only** this `backend/` folder to the target machine. Nothing from the parent `gsi-training` tree is required.

The zip includes:

- `src/` — application code, `pyproject.toml`, NAT configs
- `data/` — all runtime data (tool parquet/CSV, demo manifest, traces, evals, benchmarks). Symlinks are **materialized as real files**; absolute paths in JSON/logs are rewritten to be relative to this folder.
- `api_documentation.md`, `backend.md`, `launch_nat_e2e.sh`

**Not included:** `env/` (Python venv). Recreate on the target host with `pip install -e ./src`.

## Quick start

```bash
unzip backend-deploy.zip
cd backend

python3.12 -m venv env
source env/bin/activate
pip install -U pip wheel
pip install -e ./src

export NAT_AML_DATA_DIR="$PWD/data"
export NAT_AML_BEHAVIORAL_MODE=python_only
export NAT_AML_ENABLE_JUDGE=false
export NAT_AML_NIM_BASE_URL=http://localhost:8088/v1   # your NIM endpoint
export NAT_AML_NIM_API_KEY=EMPTY
export NAT_AML_NIM_MODEL=aml-custom-task-nim-1
export NAT_AML_SAR_MAX_TOKENS=6000
export NAT_AML_AUX_MAX_TOKENS=3000

nat serve --config_file "$PWD/src/configs/workflow.yaml" --host 0.0.0.0 --port 9111
```

Or: `./launch_nat_e2e.sh` after editing NIM URL/model in that script.

## Requirements

- Python 3.11+
- NVIDIA NAT (`nvidia-nat`) and a reachable OpenAI-compatible NIM (or compatible) endpoint for the configured model
