#!/bin/bash
# Portable launcher — run from backend/ root after: pip install -e ./src
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x "${ROOT}/env/bin/nat" ]]; then
  echo "Missing env/bin/nat. Create venv and install: pip install -e ./src" >&2
  exit 1
fi

export NAT_AML_DATA_DIR="${NAT_AML_DATA_DIR:-$ROOT/data}"
export NAT_AML_BEHAVIORAL_MODE="${NAT_AML_BEHAVIORAL_MODE:-python_only}"
export NAT_AML_ENABLE_JUDGE="${NAT_AML_ENABLE_JUDGE:-false}"
export NAT_AML_NIM_BASE_URL="${NAT_AML_NIM_BASE_URL:-http://localhost:8088/v1}"
export NAT_AML_NIM_API_KEY="${NAT_AML_NIM_API_KEY:-EMPTY}"
export NAT_AML_NIM_MODEL="${NAT_AML_NIM_MODEL:-aml-custom-task-nim-1}"
export NAT_AML_SAR_MAX_TOKENS="${NAT_AML_SAR_MAX_TOKENS:-6000}"
export NAT_AML_AUX_MAX_TOKENS="${NAT_AML_AUX_MAX_TOKENS:-3000}"

exec "${ROOT}/env/bin/nat" serve \
  --config_file "${ROOT}/src/configs/workflow.yaml" \
  --host "${NAT_HOST:-127.0.0.1}" \
  --port "${NAT_PORT:-9111}"
