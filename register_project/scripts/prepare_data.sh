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
if [[ -n "${ENV_PY:-}" ]]; then
    ENV_PYTHON="$ENV_PY/python"
else
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    else
        CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
    fi
    ENV_PREFIX="${ENV_PREFIX:-$CONDA_BASE/envs/$ENV_NAME}"
    ENV_PYTHON="$ENV_PREFIX/bin/python"
fi
PACKED_DIR="${PACKED_DIR:-$HOME/datasets/imagenet1k_packed}"
PARQUET_DIR="${PARQUET_DIR:-$HOME/datasets/imagenet-1k/data}"
COCO_DIR="${COCO_DIR:-$HOME/datasets/coco}"
ALLOW_IMAGENET_DOWNLOAD="${ALLOW_IMAGENET_DOWNLOAD:-0}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/datasets/hf_token.txt}"

TOOLS="$REPO/register_project/tools"
missing_imagenet=0

if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "ERROR: python not found or not executable at $ENV_PYTHON" >&2
    echo "Run register_project/scripts/setup_env.sh first, or set ENV_PY=/path/to/env/bin." >&2
    exit 1
fi

echo "=== Dataset preparation ==="
echo "  packed ImageNet : $PACKED_DIR"
echo "  parquet source  : $PARQUET_DIR"
echo "  COCO val        : $COCO_DIR"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    echo "  HF token file   : $HF_TOKEN_FILE"
fi
echo

# --- ImageNet (packed) -----------------------------------------------------
ensure_parquet() {
    local split="$1"
    local pattern
    if [[ "$split" == "val" ]]; then
        pattern="validation-*.parquet"
    else
        pattern="${split}-*.parquet"
    fi
    shopt -s nullglob
    local shards=("$PARQUET_DIR"/$pattern)
    shopt -u nullglob
    if [[ "${#shards[@]}" -gt 0 ]]; then
        local first total
        first="$(basename "${shards[0]}")"
        total="$(sed -nE 's/.*-of-([0-9]+)\.parquet$/\1/p' <<<"$first")"
        if [[ -z "$total" || "${#shards[@]}" -eq "$((10#$total))" ]]; then
            return 0
        fi
        echo "[imagenet] raw '$split' parquet incomplete at $PARQUET_DIR (${#shards[@]}/$((10#$total)) shards)"
    else
        echo "[imagenet] raw '$split' parquet not found at $PARQUET_DIR/$pattern"
    fi

    if [[ "$ALLOW_IMAGENET_DOWNLOAD" == "1" ]]; then
        echo "[imagenet] downloading parquet from HuggingFace (gated; needs HF token)..."
        local token_args=()
        if [[ -f "$HF_TOKEN_FILE" ]]; then
            token_args=(--token-file "$HF_TOKEN_FILE")
        fi
        "$ENV_PYTHON" "$TOOLS/download_imagenet1k.py" \
            --output-dir "$(dirname "$PARQUET_DIR")" \
            --allow 'data/*' \
            "${token_args[@]}"
        shopt -s nullglob
        shards=("$PARQUET_DIR"/$pattern)
        shopt -u nullglob
        if [[ "${#shards[@]}" -gt 0 ]]; then
            first="$(basename "${shards[0]}")"
            total="$(sed -nE 's/.*-of-([0-9]+)\.parquet$/\1/p' <<<"$first")"
            if [[ -z "$total" || "${#shards[@]}" -eq "$((10#$total))" ]]; then
                return 0
            fi
        fi
        echo "[imagenet] '$split' parquet is still incomplete after download." >&2
        return 1
    fi

    if [[ "${#shards[@]}" -gt 0 ]]; then
        cat <<EOF
  -> Existing '$split' parquet shards are incomplete. To resume the gated HF
     download, accept terms + provide an HF token via HF_TOKEN or hf_token.txt,
     then re-run with:
       ALLOW_IMAGENET_DOWNLOAD=1 bash register_project/scripts/prepare_data.sh
EOF
        return 1
    fi

    cat <<EOF
  -> To download it (ImageNet is gated, accept terms + provide an HF token via
     HF_TOKEN or hf_token.txt), re-run with:
       ALLOW_IMAGENET_DOWNLOAD=1 bash register_project/scripts/prepare_data.sh
     or manually:
       $ENV_PYTHON $TOOLS/download_imagenet1k.py --output-dir $(dirname "$PARQUET_DIR") --allow 'data/*'
EOF
    return 1
}

for split in train val; do
    bin="$PACKED_DIR/$split.bin"; idx="$PACKED_DIR/${split}_index.npy"
    if [[ -f "$bin" && -f "$idx" ]]; then
        echo "[imagenet] packed '$split' present — ok"
        continue
    fi
    echo "[imagenet] packed '$split' missing — building..."
    if ensure_parquet "$split"; then
        "$ENV_PYTHON" "$TOOLS/build_packed_imagenet.py" \
            --parquet-dir "$PARQUET_DIR" --out-dir "$PACKED_DIR" --splits "$split"
    else
        echo "[imagenet] '$split' is not ready (parquet unavailable)."
        missing_imagenet=1
    fi
done

# --- COCO val (for MBO) ----------------------------------------------------
if [[ -f "$COCO_DIR/annotations/instances_val2017.json" && -d "$COCO_DIR/val2017" ]]; then
    echo "[coco] val2017 + annotations present — ok"
else
    echo "[coco] missing — downloading val2017 + annotations..."
    "$ENV_PYTHON" "$TOOLS/download_coco_val.py" --out-dir "$COCO_DIR"
fi

echo
if [[ "$missing_imagenet" == "1" ]]; then
    cat <<EOF
[not ready] Packed ImageNet is still missing.
Training will fail until both of these files exist:
  $PACKED_DIR/train.bin
  $PACKED_DIR/train_index.npy
  $PACKED_DIR/val.bin
  $PACKED_DIR/val_index.npy
EOF
    exit 1
fi

echo "[done] dataset preparation finished."
