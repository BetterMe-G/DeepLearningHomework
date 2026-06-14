#!/bin/bash

LOCKDIR=/tmp/dlhw_ablation.lock

# 防止脚本被同时执行
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "Another ablation run is already executing. Exit."
  exit 1
fi

trap 'rm -rf "$LOCKDIR"' EXIT

# none (GPU 1)
CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --data_root ./data/celeba/img_align_celeba \
  --epochs 80 \
  --out_dir ablation/ckpts/none \
  --sample_dir ablation/samples/none \
  --log_dir ablation/logs/none \
  --no_sn > ablation/logs/none.out 2>&1 &

# ema_only (GPU 0)
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data_root ./data/celeba/img_align_celeba \
  --epochs 80 \
  --out_dir ablation/ckpts/ema_only \
  --sample_dir ablation/samples/ema_only \
  --log_dir ablation/logs/ema_only \
  --no_sn \
  --ema_decay 0.999 > ablation/logs/ema_only.out 2>&1 &

# hinge_only (GPU 2)
CUDA_VISIBLE_DEVICES=2 python3 train.py \
  --data_root ./data/celeba/img_align_celeba \
  --epochs 80 \
  --out_dir ablation/ckpts/hinge_only \
  --sample_dir ablation/samples/hinge_only \
  --log_dir ablation/logs/hinge_only \
  --no_sn \
  --loss hinge > ablation/logs/hinge_only.out 2>&1 &

# hinge_sn (GPU 4)
CUDA_VISIBLE_DEVICES=4 python3 train.py \
  --data_root ./data/celeba/img_align_celeba \
  --epochs 80 \
  --out_dir ablation/ckpts/hinge_sn \
  --sample_dir ablation/samples/hinge_sn \
  --log_dir ablation/logs/hinge_sn \
  --loss hinge > ablation/logs/hinge_sn.out 2>&1 &

# full (GPU 5)
CUDA_VISIBLE_DEVICES=5 python3 train.py \
  --data_root ./data/celeba/img_align_celeba \
  --epochs 80 \
  --out_dir ablation/ckpts/full \
  --sample_dir ablation/samples/full \
  --log_dir ablation/logs/full \
  --label_smooth 0.9 \
  --d_noise 0.1 \
  --ema_decay 0.999 \
  --loss hinge > ablation/logs/full.out 2>&1 &
