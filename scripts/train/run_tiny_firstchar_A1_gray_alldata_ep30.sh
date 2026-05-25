#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training:${PYTHONPATH:-}

mkdir -p /home/wzzz/LPRNet/experiments/firstchar_tiny_A1_gray_alldata_ep30

python /home/wzzz/LPRNet/src/training/train_tiny_province_net.py \
  --train_manifest /home/wzzz/LPRNet/manifests/firstchar_tiny_gray_alldata_v1/train.csv \
  --test_manifest /home/wzzz/LPRNet/manifests/firstchar_tiny_gray_alldata_v1/test.csv \
  --save_dir /home/wzzz/LPRNet/experiments/firstchar_tiny_A1_gray_alldata_ep30 \
  --ocr_preproc gray \
  --epochs 30 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3
