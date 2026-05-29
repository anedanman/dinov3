#!/usr/bin/env bash
# Launch ViT-S/16 + 7 register tokens (STANDARD attention) pretraining on ImageNet-1k.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PY="${ENV_PY:-$HOME/miniconda3/envs/dinov3/bin}"
CONFIG="$REPO/register_project/configs/vits_im1k_reg7_baseline.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/runs/vits_reg7_baseline}"
NGPUS="${NGPUS:-1}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Tame glibc host-RAM creep in the DataLoader workers (inherited by all forks).
# Variable-size JPEG decode/augmentation fragments per-thread malloc arenas that
# are never returned to the OS; on a low-RAM box this OOMs after hours even though
# GPU memory stays flat and gc.collect() (disabled in train.py) can't reclaim it.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"        # fewer arenas -> less fragmentation
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-0}"  # return freed memory to the OS

mkdir -p "$OUTPUT_DIR"

cd "$REPO"
"$ENV_PY/torchrun" --nproc_per_node="$NGPUS" --master_port="${MASTER_PORT:-29501}" \
    dinov3/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
