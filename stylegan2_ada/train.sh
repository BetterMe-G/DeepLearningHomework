#!/usr/bin/env bash
# Launch StyleGAN2-ADA training on A800(s).
#
# Env-var overrides:
#   GPUS=N     — number of GPUs to use (power of 2: 1/2/4/8). Default 1.
#   BATCH=N    — per-GPU batch size. Default 32.
#   KIMG=N     — total training images in thousands. Default 25000.
#   RUN_NAME=s — subdirectory under runs/. Default celeba64.

set -euo pipefail
cd "$(dirname "$0")"

GPUS=${GPUS:-1}
BATCH=${BATCH:-32}
KIMG=${KIMG:-25000}
RUN_NAME=${RUN_NAME:-celeba64}

DATA="data/celeba64.zip"
if [[ ! -f "$DATA" ]]; then
    echo "[ERROR] $DATA not found. Run prepare_data.sh first." >&2
    exit 1
fi

# --gpus must be a power of 2 (upstream requirement)
if ! (( GPUS >= 1 && (GPUS & (GPUS - 1)) == 0 )); then
    echo "[ERROR] GPUS=$GPUS is not a power of 2 (1/2/4/8)." >&2
    exit 1
fi

mkdir -p runs samples

echo "[INFO] Launching training: GPUS=$GPUS BATCH=$BATCH KIMG=$KIMG RUN_NAME=$RUN_NAME"
echo "[INFO] Data:   $DATA"
echo "[INFO] Output: runs/"

exec python3 -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
    ../stylegan2-ada-pytorch/train.py \
    --outdir=runs \
    --cfg=stylegan2 \
    --data="$DATA" \
    --gpus="$GPUS" --batch="$BATCH" \
    --gamma=1 \
    --kimg="$KIMG" \
    --snap=10 \
    --metrics=fid50k_full,is50k \
    --aug=noaug \
    --mirror=1
