#!/bin/bash

# ==============================
# Parallel FID Evaluation (No GPU 1)
# ==============================

LOCKDIR=/tmp/dlhw_fid_eval.lock

# 防止脚本被重复执行
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "FID evaluation is already running. Exit."
  exit 1
fi
trap 'rm -rf "$LOCKDIR"' EXIT

echo "Starting FID evaluation on GPUs 0,2,4,5 ..."

# ema_only -> GPU 0
CUDA_VISIBLE_DEVICES=0 \
python3 ablation/scripts/compute_fid.py --name ema_only \
> ablation/eval/ema_only/fid.log 2>&1 &

# hinge_only -> GPU 2
CUDA_VISIBLE_DEVICES=2 \
python3 ablation/scripts/compute_fid.py --name hinge_only \
> ablation/eval/hinge_only/fid.log 2>&1 &

# hinge_sn -> GPU 4
CUDA_VISIBLE_DEVICES=4 \
python3 ablation/scripts/compute_fid.py --name hinge_sn \
> ablation/eval/hinge_sn/fid.log 2>&1 &

# full -> GPU 5
CUDA_VISIBLE_DEVICES=5 \
python3 ablation/scripts/compute_fid.py --name full \
> ablation/eval/full/fid.log 2>&1 &

echo "All FID jobs submitted."
