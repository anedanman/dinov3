#!/bin/bash
#SBATCH --account=u0
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --environment=dinov3
#SBATCH --job-name=dinov3-prep
#
# One-time prep on a Daint compute node (inside the NGC container):
#   1. build the python venv overlay (training requirements)
#   2. download ImageNet-1k parquet from HuggingFace (train + validation)
#   3. build the packed ImageNet bins used by the training dataloader
#   4. sanity-check CUDA, wandb connectivity
set -euo pipefail

B=/capstor/scratch/cscs/fbombass
REPO="$B/dinov3"

echo "=== [1/4] venv setup ==="
bash "$REPO/register_project/daint/setup_daint_env.sh"
source "$B/venvs/dinov3/bin/activate"

echo "=== [2/4] ImageNet parquet download ==="
if [ -f "$B/datasets/imagenet1k_packed/train.bin" ]; then
    echo "packed train.bin already present — skipping download"
else
    # snapshot_download resumes completed files; retry through transient
    # connection drops (HF closes long-lived connections occasionally).
    ok=0
    for attempt in $(seq 1 12); do
        if python "$REPO/register_project/tools/download_imagenet1k.py" \
            --output-dir "$B/datasets/imagenet-1k" \
            --allow 'data/train-*' --allow 'data/validation-*' \
            --token-file "$B/.hf_token"; then
            ok=1; break
        fi
        echo "[download] attempt $attempt failed; retrying in 30s"
        sleep 30
    done
    if [ "$ok" != "1" ]; then
        echo "[download] giving up after 12 attempts" >&2
        exit 1
    fi
fi

echo "=== [3/4] build packed ImageNet ==="
python "$REPO/register_project/tools/build_packed_imagenet.py" \
    --parquet-dir "$B/datasets/imagenet-1k/data" \
    --out-dir "$B/datasets/imagenet1k_packed" \
    --splits train val
ls -la "$B/datasets/imagenet1k_packed/"

echo "=== [4/4] sanity checks ==="
python -c "import torch; print('cuda devices:', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
WANDB_API_KEY="$(cat "$B/.wandb_key")" python -c "
import wandb
wandb.login(key='$(cat "$B/.wandb_key")'.strip())
print('wandb login ok')
"
echo "PREP_DONE"
