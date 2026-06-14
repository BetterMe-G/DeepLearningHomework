#!/usr/bin/env bash
# Convert CelebA raw images to a 64x64 ZIP for StyleGAN2-ADA.
#
# Output:
#   data/celeba64.zip   — 202,599 images, 178-px center crop, resized to 64x64.
#
# One-time cost: ~5-10 min, output ~2.4 GB on disk.
#
# Override the CelebA source via CELEBA_SRC env var (needed on the A800 server,
# where the path is /hpc_stor03/sjtu_home/siru.ge/DeepLearningHomework/data/celeba/img_align_celeba
# with PNG files instead of JPG).

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data

SRC="${CELEBA_SRC:-../data/celeba/img_align_celeba/img_align_celeba}"
DEST="data/celeba64.zip"

if [[ -f "$DEST" ]]; then
    echo "[INFO] $DEST already exists; skipping. Delete it to re-create."
    ls -lh "$DEST"
    exit 0
fi

if [[ ! -d "$SRC" ]]; then
    echo "[ERROR] CelebA source directory not found: $SRC" >&2
    echo "        Did you place img_align_celeba/ under data/celeba/?" >&2
    exit 1
fi

echo "[INFO] Building $DEST from $SRC ..."
python3 ../stylegan2-ada-pytorch/dataset_tool.py \
    --source="$SRC" \
    --dest="$DEST" \
    --width=64 --height=64 \
    --transform=center-crop

echo "[INFO] Dataset ready."
ls -lh data/
