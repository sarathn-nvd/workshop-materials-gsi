#!/usr/bin/env bash
# Launch the AML investigation backend (FastAPI via nat serve).
set -euo pipefail

# Find the project root (two dirs above this script).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
BACKEND="$(cd "$SRC/.." && pwd)"

# Activate the venv if it exists; otherwise assume PATH is set.
if [[ -f "$BACKEND/env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$BACKEND/env/bin/activate"
fi

# Default data dir to the local data plane.
export NAT_AML_DATA_DIR="${NAT_AML_DATA_DIR:-$BACKEND/data}"

# Pretty-print what's wired.
echo "============================================================"
echo "  AML Investigation Backend"
echo "  data_dir: $NAT_AML_DATA_DIR"
echo "  config:   $SRC/configs/workflow.yaml"
echo "============================================================"

cd "$SRC"
exec nat serve --config_file configs/workflow.yaml "$@"
