#!/usr/bin/env bash
# Shared launcher helpers for register_project training scripts.
# Source this file, then call:
#   dinov3_launch <config> <output_dir> <master_port> [config overrides...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-dinov3}"
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

# DataLoader host-RAM vs throughput knobs (all override-able). These machines
# have limited host RAM, so defaults avoid long-run page-cache / allocator creep.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-134217728}"
export DINOV3_DATASET_MALLOC_TRIM_EVERY="${DINOV3_DATASET_MALLOC_TRIM_EVERY:-512}"
export DINOV3_COLLATE_MALLOC_TRIM_EVERY="${DINOV3_COLLATE_MALLOC_TRIM_EVERY:-16}"
export DINOV3_PACKED_MMAP_INDEX="${DINOV3_PACKED_MMAP_INDEX:-1}"
export DINOV3_PACKED_DROP_CACHE="${DINOV3_PACKED_DROP_CACHE:-1}"
export DINOV3_WANDB_MAX_IMAGE_PIXELS="${DINOV3_WANDB_MAX_IMAGE_PIXELS:-4000000}"

dinov3_launch() {
    local config="$1" output_dir="$2" master_port="$3"
    shift 3
    mkdir -p "$output_dir"
    cd "$REPO"
    "$TORCHRUN" --nproc_per_node="$NGPUS" --master_port="${MASTER_PORT:-$master_port}" \
        dinov3/train/train.py \
        --config-file "$config" \
        --output-dir "$output_dir" \
        "$@"
}
