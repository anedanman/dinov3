#!/usr/bin/env bash
# slot5-gate: slot2 (slot competition + separate register budget + gaussian
# register init) PLUS a learnable per-head, per-layer gate on the register
# budget (init 1 = ungated). Gate values are logged to wandb under
# register_gate/* by the register diagnostics.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

dinov3_launch \
    "$REPO/register_project/configs/vits_im1k_reg7_slot.yaml" \
    "${OUTPUT_DIR:-$REPO/runs/vits_reg7_slot5_gate}" \
    29506 \
    student.patch_cls_attn_type=separate_register_budget \
    student.register_init=gaussian \
    student.register_budget_gate=true \
    train.batch_size_per_gpu=256 \
    train.grad_accum_steps=2 \
    train.num_workers=12 \
    train.prefetch_factor=2 \
    train.wandb.name=vits-reg7-slot5-gate \
    "$@"
