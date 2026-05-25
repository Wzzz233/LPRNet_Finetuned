#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training:${PYTHONPATH:-}

TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/firstchar_tiny_gray_alldata_v1/train.csv
TEST_MANIFEST=/home/wzzz/LPRNet/manifests/firstchar_tiny_gray_alldata_v1/test.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/firstchar_tiny_gray_alldata_v2_ep30

python /home/wzzz/LPRNet/src/training/train_tiny_province_net.py \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$TEST_MANIFEST" \
  --save_dir "$SAVE_DIR" \
  --ocr_preproc gray \
  --epochs 30 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3
