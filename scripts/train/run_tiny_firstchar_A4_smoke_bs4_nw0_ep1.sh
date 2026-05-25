#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training:${PYTHONPATH:-}
SAVE_DIR=/home/wzzz/LPRNet/experiments/firstchar_tiny_A4_smoke_bs4_nw0_ep1
mkdir -p "$SAVE_DIR"
python /home/wzzz/LPRNet/src/training/train_tiny_province_net.py \
  --train_manifest /home/wzzz/LPRNet/manifests/firstchar_tiny_gray3_fullcrop_bal31_v1/train.csv \
  --test_manifest /home/wzzz/LPRNet/manifests/firstchar_tiny_gray_alldata_v1/test.csv \
  --save_dir "$SAVE_DIR" \
  --ocr_preproc gray3 \
  --input_mode full_crop \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 0 \
  --lr 1e-3
