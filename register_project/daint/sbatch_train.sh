#!/bin/bash
#SBATCH --account=u0
#SBATCH --partition=normal
#SBATCH --time=23:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --environment=dinov3
#
# DINOv3 register-tokens training segment on one Daint GH200 node (4 GPUs).
# Auto-resumes from the latest checkpoint in the run's output dir, so segments
# can be chained with --dependency=afterany (see submit_chain.sh).
#
# Usage: sbatch --job-name=<run> --output=<log> sbatch_train.sh <config-stem> <run-name>
#   e.g. sbatch sbatch_train.sh vits_im1k_reg4_slot vits-reg4-slot
set -euo pipefail

CONFIG_STEM="$1"
RUN_NAME="$2"

SCRATCH_BASE=/capstor/scratch/cscs/fbombass
REPO="$SCRATCH_BASE/dinov3"
VENV="$SCRATCH_BASE/venvs/dinov3"
OUT="$SCRATCH_BASE/dinov3-runs/$RUN_NAME"
TOTAL_STEPS=200000
LAST_CKPT=$((TOTAL_STEPS - 1))

# Skip cleanly if a previous segment already finished training.
if [ -d "$OUT/ckpt/$LAST_CKPT" ] || [ -d "$OUT/ckpt/${LAST_CKPT}_keep" ]; then
    echo "[skip] $RUN_NAME already trained to step $LAST_CKPT"
    exit 0
fi

export WANDB_API_KEY="$(cat "$SCRATCH_BASE/.wandb_key")"
export WANDB_RUN_ID="$RUN_NAME"     # same wandb run across chained segments
export PYTHONPATH="$REPO"
export DINOV3_PACKED_MMAP_INDEX=1
export DINOV3_PACKED_DROP_CACHE=0   # GH200 nodes have plenty of host RAM
export DINOV3_WANDB_MAX_IMAGE_PIXELS=90000000
export OMP_NUM_THREADS=8

source "$VENV/bin/activate"
mkdir -p "$OUT"
cd "$REPO"

# NOTE: not `torchrun` — the container's /usr/local/bin/torchrun runs under the
# system python, which cannot see the venv overlay's packages.
python -m torch.distributed.run --nproc_per_node=4 --master_port=29501 \
    dinov3/train/train.py \
    --config-file "$REPO/register_project/configs/daint/${CONFIG_STEM}.yaml" \
    --output-dir "$OUT"
