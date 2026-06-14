#!/bin/bash
set -euo pipefail

cd "$HOME/DeepLearningHomework"

# sn_ls
CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --data_root ./data/celeba/img_align_celeba \
  --epochs 80 \
  --out_dir ablation/ckpts/sn_ls \
  --sample_dir ablation/samples/sn_ls \
  --log_dir ablation/logs/sn_ls \
  --label_smooth 0.9 \
  > ablation/logs/sn_ls.out 2>&1 &

# sn_dnoise
CUDA_VISIBLE_DEVICES=2 python3 train.py \
  --data_root ./data/celeba/img_align_celeba \
  --epochs 80 \
  --out_dir ablation/ckpts/sn_dnoise \
  --sample_dir ablation/samples/sn_dnoise \
  --log_dir ablation/logs/sn_dnoise \
  --d_noise 0.1 \
  > ablation/logs/sn_dnoise.out 2>&1 &

# sn_ema
CUDA_VISIBLE_DEVICES=4 python3 train.py \
  --data_root ./data/celeba/img_align_celeba \
  --epochs 80 \
  --out_dir ablation/ckpts/sn_ema \
  --sample_dir ablation/samples/sn_ema \
  --log_dir ablation/logs/sn_ema \
  --ema_decay 0.999 \
  > ablation/logs/sn_ema.out 2>&1 &
