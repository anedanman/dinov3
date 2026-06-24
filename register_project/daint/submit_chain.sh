#!/bin/bash
# Submit a chain of dependent 24h training segments for one variant.
# Usage: bash submit_chain.sh <config-stem> <run-name> [num-segments=3] [first-dep-jobid]
# If first-dep-jobid is given, segment 1 waits for that job to succeed (afterok).
set -euo pipefail

CONFIG_STEM="$1"
RUN_NAME="$2"
NSEG="${3:-3}"
FIRST_DEP="${4:-}"

SCRATCH_BASE=/capstor/scratch/cscs/fbombass
SBATCH_SCRIPT="$SCRATCH_BASE/dinov3/register_project/daint/sbatch_train.sh"
LOG_DIR="$SCRATCH_BASE/dinov3-runs/$RUN_NAME/slurm"
mkdir -p "$LOG_DIR"

dep=()
if [ -n "$FIRST_DEP" ]; then
    dep=(--dependency=afterok:"$FIRST_DEP")
fi
for i in $(seq 1 "$NSEG"); do
    jid=$(sbatch --parsable "${dep[@]}" \
        --job-name="$RUN_NAME-$i" \
        --output="$LOG_DIR/seg$i-%j.out" \
        "$SBATCH_SCRIPT" "$CONFIG_STEM" "$RUN_NAME")
    echo "submitted $RUN_NAME segment $i: job $jid"
    dep=(--dependency=afterany:"$jid")
done
