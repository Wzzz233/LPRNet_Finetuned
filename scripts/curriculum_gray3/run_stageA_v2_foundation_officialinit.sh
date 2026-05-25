#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

INIT="experiments/curriculum_gray3_base/init_multihead_from_official.pth"
TRAIN_MANIFEST="manifests/curriculum_gray3_stagea_v2_foundation/train_stageA_v2.csv"
VAL_MANIFEST="manifests/curriculum_gray3_stagea_v2_foundation/val_stageA_v2.csv"
SAVE_DIR="experiments/curriculum_gray3_stageA_v2_foundation_officialinit"
LOG_PATH="$SAVE_DIR/train.log"

mkdir -p "$SAVE_DIR"

python3 src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$VAL_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --trainable_families normal7,green8 \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc gray3 \
  --gray3_prob 1.0 \
  --main_group_by family \
  --train_batch_size 128 \
  --test_batch_size 120 \
  --max_epoch 30 \
  --learning_rate 0.005 \
  --lr_schedule 12 20 26 \
  --freeze_backbone False \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --save_folder "$SAVE_DIR" \
  --num_workers 4 \
  --seed 42 \
  --cuda True \
  > "$LOG_PATH" 2>&1
