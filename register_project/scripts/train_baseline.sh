#!/usr/bin/env bash
# Launch ViT-S/16 + 7 register tokens (STANDARD attention) pretraining on ImageNet-1k.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

dinov3_launch \
    "$REPO/register_project/configs/vits_im1k_reg7_baseline.yaml" \
    "${OUTPUT_DIR:-$REPO/runs/vits_reg7_baseline}" \
    29501 \
    train.batch_size_per_gpu=256 \
    train.grad_accum_steps=2 \
    train.num_workers=12 \
    train.prefetch_factor=2 \
    train.wandb.name=vits-reg7-baseline \
    "$@"
