#!/usr/bin/env bash
# Validate that the configured register-token run can start cleanly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-dinov3}"
CONFIG="${CONFIG:-$REPO/register_project/configs/vits_im1k_reg7_slot.yaml}"

if [[ -n "${ENV_PY:-}" ]]; then
    PYTHON="$ENV_PY/python"
else
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    else
        CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
    fi
    ENV_PREFIX="${ENV_PREFIX:-$CONDA_BASE/envs/$ENV_NAME}"
    PYTHON="$ENV_PREFIX/bin/python"
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: python not found or not executable at $PYTHON" >&2
    echo "Run register_project/scripts/setup_env.sh first, or set ENV_PY=/path/to/env/bin." >&2
    exit 1
fi

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
"$PYTHON" "$REPO/register_project/tools/check_train_ready.py" --config "$CONFIG" "$@"
