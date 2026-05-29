#!/usr/bin/env bash
# Create the `dinov3` conda env for register-tokens training.
#
# Defaults target Blackwell GPUs (CUDA 12.8, torch 2.11 / torchvision 0.26).
# Override for a different CUDA build, e.g.:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
#   TORCH_VERSION=2.5.1 TORCHVISION_VERSION=0.20.1 bash setup_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-dinov3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[env] '$ENV_NAME' already exists — reusing it."
else
    echo "[env] creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

echo "[torch] installing torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION from $TORCH_INDEX_URL"
conda run -n "$ENV_NAME" pip install --no-input \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "$TORCH_INDEX_URL"

echo "[deps] installing training requirements"
conda run -n "$ENV_NAME" pip install --no-input -r "$REPO/register_project/requirements-train.txt"

echo "[verify] checking torch / CUDA"
conda run -n "$ENV_NAME" python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
PY

echo
echo "[done] env '$ENV_NAME' ready. Next:"
echo "  bash register_project/scripts/prepare_data.sh        # check/build datasets"
echo "  conda activate $ENV_NAME && bash register_project/scripts/train_baseline.sh"
