#!/usr/bin/env bash
# slot4-late6: slot2 (slot competition + separate register budget + gaussian
# register init) but slot competition only in the SECOND HALF of the blocks
# (layers 6-11 for ViT-S); earlier layers keep standard register rows with the
# separate register budget.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

dinov3_launch \
    "$REPO/register_project/configs/vits_im1k_reg7_slot.yaml" \
    "${OUTPUT_DIR:-$REPO/runs/vits_reg7_slot4_late6}" \
    29505 \
    student.patch_cls_attn_type=separate_register_budget \
    student.register_init=gaussian \
    student.slot_start_layer=6 \
    train.batch_size_per_gpu=256 \
    train.grad_accum_steps=2 \
    train.num_workers=12 \
    train.prefetch_factor=2 \
    train.wandb.name=vits-reg7-slot4-late6 \
    "$@"
