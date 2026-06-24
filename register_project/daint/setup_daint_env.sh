#!/bin/bash
# One-time environment setup on Daint: build a venv overlay on top of the NGC
# PyTorch container. Run INSIDE the container, e.g.:
#   srun -p debug -t 30 --environment=dinov3 bash register_project/daint/setup_daint_env.sh
set -euo pipefail

SCRATCH_BASE=/capstor/scratch/cscs/fbombass
REPO="$SCRATCH_BASE/dinov3"
VENV="$SCRATCH_BASE/venvs/dinov3"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "[venv] creating $VENV (system-site-packages on top of container torch)"
    python3 -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"

echo "[deps] installing training requirements"
pip install --no-cache-dir -r "$REPO/register_project/requirements-train.txt"

echo "[verify] torch / CUDA / deps"
python - <<'EOF'
import importlib
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
required = ["numpy", "PIL", "omegaconf", "wandb", "huggingface_hub", "pyarrow",
            "pycocotools", "matplotlib", "scipy", "sklearn", "termcolor", "submitit",
            "iopath", "pandas", "torchmetrics", "fvcore", "ftfy", "regex"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("missing deps:\n  " + "\n  ".join(missing))
print("all training deps importable")
EOF
echo "[done]"
