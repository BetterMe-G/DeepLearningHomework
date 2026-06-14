#!/usr/bin/env bash
# Download CelebA (jessicali9530/celeba-dataset) from Kaggle and extract under ./data/celeba
# Prereq: pip install kaggle, and place ~/.kaggle/kaggle.json (your API token, chmod 600)
#
# After this script finishes you should have ~202599 jpgs at:
#   ./data/celeba/img_align_celeba/*.jpg
# which is what dataset.py expects when --dataset celeba.
set -e
mkdir -p ./data/celeba
cd ./data/celeba
echo "[INFO] Downloading CelebA (jessicali9530/celeba-dataset) from Kaggle ..."
kaggle datasets download -d jessicali9530/celeba-dataset
echo "[INFO] Unzipping ..."
unzip -o celeba-dataset.zip
# The Kaggle archive sometimes wraps img_align_celeba in another zip.
if [ -f "img_align_celeba.zip" ]; then
    echo "[INFO] Extracting inner img_align_celeba.zip ..."
    unzip -o img_align_celeba.zip
fi
# Some versions of the archive nest one extra folder
# (img_align_celeba/img_align_celeba/*.jpg). Flatten if so.
# NOTE: do NOT use `mv img_align_celeba/img_align_celeba/*.jpg ...`
# -- 202k filenames blow past ARG_MAX. Rename the inner dir instead.
if [ -d "img_align_celeba/img_align_celeba" ]; then
    echo "[INFO] Flattening nested img_align_celeba/img_align_celeba ..."
    mv img_align_celeba img_align_celeba_outer
    mv img_align_celeba_outer/img_align_celeba .
    rm -rf img_align_celeba_outer
fi
echo "[INFO] Done. Image count:"
find img_align_celeba -type f -iname '*.jpg' | wc -l
echo "[INFO] Train with: python train.py --dataset celeba --data_root ./data/celeba/img_align_celeba"
