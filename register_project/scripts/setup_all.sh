#!/usr/bin/env bash
# One-shot bootstrap on a fresh machine: create the env, then prepare datasets.
#   git clone https://github.com/anedanman/dinov3 && cd dinov3
#   git checkout register-tokens
#   bash register_project/scripts/setup_all.sh
#
# Pass ALLOW_IMAGENET_DOWNLOAD=1 (with an HF token) to also fetch ImageNet parquet.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/setup_env.sh"
bash "$HERE/prepare_data.sh"

echo
echo "[all done] To train:"
echo "  conda activate ${ENV_NAME:-dinov3}"
echo "  bash register_project/scripts/train_baseline.sh   # or train_slot.sh"
