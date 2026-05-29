#!/usr/bin/env bash
# Launch ViT-S/16 + 7 register tokens (SLOT-REGISTER competition attention) pretraining.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-dinov3}"
CONFIG="$REPO/register_project/configs/vits_im1k_reg7_slot.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/runs/vits_reg7_slot}"
NGPUS="${NGPUS:-1}"

if [[ -n "${ENV_PY:-}" ]]; then
    ENV_PREFIX="$(cd "$ENV_PY/.." && pwd)"
    TORCHRUN="$ENV_PY/torchrun"
else
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    else
        CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
    fi
    ENV_PREFIX="${ENV_PREFIX:-$CONDA_BASE/envs/$ENV_NAME}"
    TORCHRUN="$ENV_PREFIX/bin/torchrun"
fi
if [[ ! -x "$TORCHRUN" ]]; then
    echo "ERROR: torchrun not found or not executable at $TORCHRUN" >&2
    echo "Run register_project/scripts/setup_env.sh first, or set ENV_PY=/path/to/env/bin." >&2
    exit 1
fi

export CONDA_PREFIX="$ENV_PREFIX"
export CONDA_DEFAULT_ENV="$ENV_NAME"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# DataLoader host-RAM vs throughput knobs (all override-able). Defaults favor
# TRAINING SPEED; on a RAM-constrained box use the low-RAM values in the comments.
# glibc still returns freed memory to the OS lazily, just not on every free.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-8}"                       # low-RAM: 2
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-268435456}"   # low-RAM: 0 (trim on every free)
export DINOV3_DATASET_MALLOC_TRIM_EVERY="${DINOV3_DATASET_MALLOC_TRIM_EVERY:-0}"  # low-RAM: 128 (per-item heap walk)
export DINOV3_PACKED_DROP_CACHE="${DINOV3_PACKED_DROP_CACHE:-0}"       # low-RAM: 1 (drop page cache after each read)

mkdir -p "$OUTPUT_DIR"

cd "$REPO"
"$TORCHRUN" --nproc_per_node="$NGPUS" --master_port="${MASTER_PORT:-29502}" \
    dinov3/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
