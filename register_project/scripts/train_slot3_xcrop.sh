#!/usr/bin/env bash
# slot3-xcrop: slot2 (slot competition + separate register budget + gaussian
# register init) PLUS the cross-crop register consistency loss
# (Hungarian-matched student/teacher registers across global crops).
# v2: subtract_mean=true — registers collapse onto one shared direction
# (reg_pairwise_cos ~0.999), so raw-cosine matching saturates trivially;
# matching mean-subtracted residuals keeps the loss meaningful.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

dinov3_launch \
    "$REPO/register_project/configs/vits_im1k_reg7_slot.yaml" \
    "${OUTPUT_DIR:-$REPO/runs/vits_reg7_slot3_xcrop2}" \
    29504 \
    student.patch_cls_attn_type=separate_register_budget \
    student.register_init=gaussian \
    register_consistency.enabled=true \
    register_consistency.loss_weight=0.1 \
    register_consistency.matching=hungarian \
    register_consistency.warmup_steps=10000 \
    register_consistency.subtract_mean=true \
    train.batch_size_per_gpu=256 \
    train.grad_accum_steps=2 \
    train.num_workers=12 \
    train.prefetch_factor=2 \
    train.wandb.name=vits-reg7-slot3-xcrop2-resid \
    "$@"
