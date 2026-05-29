#!/usr/bin/env bash
# Launch ViT-S/16 + 7 register tokens (SLOT-REGISTER competition attention) pretraining.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PY="${ENV_PY:-$HOME/miniconda3/envs/dinov3/bin}"
CONFIG="$REPO/register_project/configs/vits_im1k_reg7_slot.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/runs/vits_reg7_slot}"
NGPUS="${NGPUS:-1}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

cd "$REPO"
"$ENV_PY/torchrun" --nproc_per_node="$NGPUS" --master_port="${MASTER_PORT:-29502}" \
    dinov3/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
