#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training:${PYTHONPATH:-}

SAVE_DIR=/home/wzzz/LPRNet/experiments/firstchar_tiny_A4C_gray3_fullcrop_bal31_bs4_nw0_ep30_fix1
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/firstchar_tiny_gray3_fullcrop_bal31_v1/train.csv
TEST_MANIFEST=/home/wzzz/LPRNet/manifests/firstchar_tiny_gray_alldata_v1/test.csv
LOG_PATH="$SAVE_DIR/train.log"

mkdir -p "$SAVE_DIR"

python /home/wzzz/LPRNet/src/training/train_tiny_province_net.py \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$TEST_MANIFEST" \
  --save_dir "$SAVE_DIR" \
  --ocr_preproc gray3 \
  --input_mode full_crop \
  --full_crop_height 64 \
  --epochs 30 \
  --batch_size 4 \
  --num_workers 0 \
  --log_interval 200 \
  --lr 1e-3 | tee "$LOG_PATH"
