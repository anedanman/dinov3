#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 HOST CONFIG_STEM [key=value ...]" >&2
    exit 2
fi

HOST="$1"
CONFIG_STEM="$2"
shift 2

REPO="${REMOTE_REPO:-~/dinov3}"
ENV_PYTHON="${REMOTE_PYTHON:-~/miniconda3/envs/dinov3/bin/python}"
SESSION="classifier_${CONFIG_STEM}"
CONFIG="register_project/classification/configs/${CONFIG_STEM}.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-~/dinov3-runs/${CONFIG_STEM}}"
OVERRIDES=("train.output_dir=$OUTPUT_DIR" "$@")
printf -v OVERRIDES_Q ' %q' "${OVERRIDES[@]}"

ssh "$HOST" "cd $REPO && PYTHONPATH=. $ENV_PYTHON -m register_project.classification.train --config $CONFIG --probe-only$OVERRIDES_Q"
ssh "$HOST" "mkdir -p $OUTPUT_DIR; tmux has-session -t '$SESSION' 2>/dev/null && tmux kill-session -t '$SESSION' || true; \
    tmux new-session -d -s '$SESSION' \
    'cd $REPO && PYTHONPATH=. $ENV_PYTHON -u -m register_project.classification.train --config $CONFIG$OVERRIDES_Q 2>&1 | tee -a $OUTPUT_DIR/launcher.log'"

echo "launched $CONFIG_STEM on $HOST in tmux session $SESSION"
