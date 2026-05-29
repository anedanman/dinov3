#!/usr/bin/env bash
# Check for / prepare all datasets needed for register-tokens training:
#   * packed ImageNet-1k (train + val)  -> $PACKED_DIR
#   * COCO val2017 + annotations        -> $COCO_DIR
#
# Idempotent: skips anything already present. ImageNet is gated on HF, so if the
# raw parquet is missing this script downloads it only when ALLOW_IMAGENET_DOWNLOAD=1
# and an HF token is available (HF_TOKEN env or hf_token.txt).
#
# Override locations via env vars (defaults match the configs' ~/datasets paths):
#   PACKED_DIR, PARQUET_DIR, COCO_DIR, ENV_NAME
set -euo pipefail

ENV_NAME="${ENV_NAME:-dinov3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKED_DIR="${PACKED_DIR:-$HOME/datasets/imagenet1k_packed}"
PARQUET_DIR="${PARQUET_DIR:-$HOME/datasets/imagenet-1k/data}"
COCO_DIR="${COCO_DIR:-$HOME/datasets/coco}"
ALLOW_IMAGENET_DOWNLOAD="${ALLOW_IMAGENET_DOWNLOAD:-0}"

RUN="conda run -n ${ENV_NAME} python"
TOOLS="$REPO/register_project/tools"

echo "=== Dataset preparation ==="
echo "  packed ImageNet : $PACKED_DIR"
echo "  parquet source  : $PARQUET_DIR"
echo "  COCO val        : $COCO_DIR"
echo

# --- ImageNet (packed) -----------------------------------------------------
ensure_parquet() {
    if compgen -G "$PARQUET_DIR/*.parquet" >/dev/null; then
        return 0
    fi
    echo "[imagenet] raw parquet not found at $PARQUET_DIR"
    if [[ "$ALLOW_IMAGENET_DOWNLOAD" == "1" ]]; then
        echo "[imagenet] downloading parquet from HuggingFace (gated; needs HF token)..."
        $RUN "$TOOLS/download_imagenet1k.py" --output-dir "$(dirname "$PARQUET_DIR")" --allow 'data/*'
    else
        cat <<EOF
  -> To download it (ImageNet is gated, accept terms + provide an HF token via
     HF_TOKEN or hf_token.txt), re-run with:
       ALLOW_IMAGENET_DOWNLOAD=1 bash register_project/scripts/prepare_data.sh
     or manually:
       $RUN $TOOLS/download_imagenet1k.py --output-dir $(dirname "$PARQUET_DIR") --allow 'data/*'
EOF
        return 1
    fi
}

for split in train val; do
    bin="$PACKED_DIR/$split.bin"; idx="$PACKED_DIR/${split}_index.npy"
    if [[ -f "$bin" && -f "$idx" ]]; then
        echo "[imagenet] packed '$split' present — ok"
        continue
    fi
    echo "[imagenet] packed '$split' missing — building..."
    if ensure_parquet; then
        $RUN "$TOOLS/build_packed_imagenet.py" \
            --parquet-dir "$PARQUET_DIR" --out-dir "$PACKED_DIR" --splits "$split"
    else
        echo "[imagenet] skipping '$split' build (parquet unavailable)."
    fi
done

# --- COCO val (for MBO) ----------------------------------------------------
if [[ -f "$COCO_DIR/annotations/instances_val2017.json" && -d "$COCO_DIR/val2017" ]]; then
    echo "[coco] val2017 + annotations present — ok"
else
    echo "[coco] missing — downloading val2017 + annotations..."
    $RUN "$TOOLS/download_coco_val.py" --out-dir "$COCO_DIR"
fi

echo
echo "[done] dataset preparation finished."
