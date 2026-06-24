#!/usr/bin/env bash
# Sequentially train three slot2-based ablations (slot competition, 256 micro
# x 2 grad-accum, NO cross-crop consistency loss):
#   1. slot5-gate:        slot2 + learnable per-head gate on the separate register budget
#   2. slot6-nobudget:    slot2 with the separate CLS/patch register budget DISABLED
#   3. slot7-learnedinit: slot2 with plain learned register init (no gaussian sampling)
#
# Each stage resumes from its latest checkpoint if the script is relaunched and
# is skipped entirely once its DONE marker exists, so the tmux session can be
# restarted at any point and continues where it left off.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

COMMON_OVERRIDES=(
    train.batch_size_per_gpu=256
    train.grad_accum_steps=2
    train.num_workers=12
    train.prefetch_factor=2
)

run_stage() {
    local name="$1" wandb_name="$2"
    shift 2
    local output_dir="$REPO/runs/$name"
    local marker="$output_dir/SEQ_STAGE_DONE"
    if [[ -f "$marker" ]]; then
        echo "=== stage $name already done, skipping ==="
        return 0
    fi
    mkdir -p "$output_dir"
    echo "=== stage $name ($wandb_name) starting at $(date) ==="
    dinov3_launch \
        "$REPO/register_project/configs/vits_im1k_reg7_slot.yaml" \
        "$output_dir" \
        29507 \
        "${COMMON_OVERRIDES[@]}" \
        train.wandb.name="$wandb_name" \
        "$@" \
        2>&1 | tee -a "$output_dir/train.log"
    touch "$marker"
    echo "=== stage $name finished at $(date) ==="
}

run_stage vits_reg7_slot5_gate vits-reg7-slot5-gate \
    student.patch_cls_attn_type=separate_register_budget \
    student.register_init=gaussian \
    student.register_budget_gate=true

run_stage vits_reg7_slot6_nobudget vits-reg7-slot6-nobudget \
    student.patch_cls_attn_type=standard \
    student.register_init=gaussian

run_stage vits_reg7_slot7_learnedinit vits-reg7-slot7-learnedinit \
    student.patch_cls_attn_type=separate_register_budget \
    student.register_init=learned

echo "=== all ablation stages complete at $(date) ==="
