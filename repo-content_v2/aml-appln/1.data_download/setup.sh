#!/usr/bin/env bash
# setup.sh — Install Step 1 dependencies into the shared jupyter-env.
# Idempotent: safe to re-run.
#
# Step 1 (data download), Step 2 (data processing), and the Jupyter dev
# environment all share a single virtualenv at ../jupyter-env so we don't
# duplicate ~3 GB of HF/torch/etc. on disk and don't drift between steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/jupyter-env}"
REQ_FILE="${REQ_FILE:-$REPO_ROOT/requirements.txt}"

echo "[setup] Using Python: $($PYTHON_BIN --version)"
echo "[setup] Target venv:  $VENV_DIR"
echo "[setup] Requirements: $REQ_FILE"

if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating shared virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[setup] Reusing existing virtualenv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[setup] Upgrading pip"
pip install --upgrade pip wheel setuptools >/dev/null

echo "[setup] Installing requirements"
pip install -r "$REQ_FILE"

echo "[setup] Installing Playwright Chromium browser"
python -m playwright install chromium

echo ""
echo "[setup] Done."
echo ""
echo "Next steps:"
echo "  1. Export credentials:"
echo "       export HF_TOKEN=<your_huggingface_token>"
echo "       (Kaggle: place kaggle.json at ~/.kaggle/kaggle.json, chmod 600)"
echo "  2. Activate the shared venv:"
echo "       source $VENV_DIR/bin/activate"
echo "  3. Run the downloader:"
echo "       python download.py --tasks all --parallel-workers 8"
