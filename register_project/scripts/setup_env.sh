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
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
else
    CONDA_BASE="$(conda info --base)"
fi
ENV_PREFIX="${ENV_PREFIX:-$CONDA_BASE/envs/$ENV_NAME}"
ENV_PYTHON="$ENV_PREFIX/bin/python"

if [[ -x "$ENV_PYTHON" ]]; then
    echo "[env] '$ENV_NAME' already exists — reusing it."
else
    echo "[env] creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

# Run pip without inheriting machine-level hash/constraint enforcement
# (PIP_REQUIRE_HASHES / PIP_CONSTRAINT), which otherwise rejects our unpinned
# requirements with "PACKAGES DO NOT MATCH THE HASHES". Set PIP_IGNORE_CONFIG=1
# to additionally bypass pip.conf (also disables any configured mirror).
PIP_IGNORE_CONFIG="${PIP_IGNORE_CONFIG:-0}"
pip_install() {
    local outer_env=(env -u PIP_REQUIRE_HASHES -u PIP_CONSTRAINT)
    if [[ "$PIP_IGNORE_CONFIG" == "1" ]]; then
        outer_env+=(PIP_CONFIG_FILE=/dev/null)
    fi
    "${outer_env[@]}" "$ENV_PYTHON" -m pip install --no-input --no-cache-dir "$@"
}

if "$ENV_PYTHON" -c 'import sys, torch, torchvision; sys.exit(0 if torch.__version__.split("+")[0] == sys.argv[1] and torchvision.__version__.split("+")[0] == sys.argv[2] else 1)' "$TORCH_VERSION" "$TORCHVISION_VERSION" >/dev/null 2>&1; then
    echo "[torch] torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION already installed — ok"
else
    echo "[torch] installing torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION from $TORCH_INDEX_URL"
    echo "[torch] installing CUDA/runtime dependencies from $PYPI_INDEX_URL"
    pip_install \
        "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1" \
        "cuda-bindings<13,>=12.9.4" \
        "nvidia-cudnn-cu12==9.19.0.56" \
        "nvidia-cusparselt-cu12==0.7.1" \
        "nvidia-nccl-cu12==2.28.9" \
        "nvidia-nvshmem-cu12==3.4.5" \
        "triton==3.6.0" \
        "filelock" \
        "typing-extensions>=4.10.0" \
        "setuptools<82" \
        "sympy>=1.13.3" \
        "networkx>=2.5.1" \
        "jinja2" \
        "fsspec>=0.8.5" \
        "pillow!=8.3.*,>=5.3.0" \
        --index-url "$PYPI_INDEX_URL"
    pip_install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "$TORCH_INDEX_URL" --no-deps
fi

echo "[deps] installing training requirements"
pip_install -r "$REPO/register_project/requirements-train.txt"

echo "[verify] checking torch / CUDA"
"$ENV_PYTHON" -c '
import importlib
import torch
import torchvision

required = [
    "numpy",
    "PIL",
    "omegaconf",
    "wandb",
    "huggingface_hub",
    "pyarrow",
    "pycocotools",
    "matplotlib",
    "scipy",
    "sklearn",
    "termcolor",
    "submitit",
    "iopath",
    "pandas",
    "torchmetrics",
    "fvcore",
    "ftfy",
    "regex",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
if missing:
    raise SystemExit("missing/unimportable training dependencies:\n  " + "\n  ".join(missing))
'

echo
echo "[done] env '$ENV_NAME' ready. Next:"
echo "  bash register_project/scripts/prepare_data.sh        # check/build datasets"
echo "  conda activate $ENV_NAME && bash register_project/scripts/train_baseline.sh"
