#!/bin/bash
# Green CCPD2019 CV-replace smoke training script.
# Usage: bash scripts/train/run_green_ccpd2019_cvreplace_smoke.sh

DATE_TAG="20260508"
EXP_DIR="experiments/green_ccpd2019_tilt_db_challenge_cvreplace_${DATE_TAG}_smoke"
mkdir -p "$EXP_DIR"

cd /home/wzzz/LPRNet

export PYTHONPATH=/home/wzzz/LPRNet/src

python3 -u src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest "manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_${DATE_TAG}/train_combined.csv" \
  --test_manifest "manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_${DATE_TAG}/val_cvreplace.csv" \
  --dataset_root /home/wzzz/LPRNet \
  --pretrained_model experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth \
  --save_folder "$EXP_DIR" \
  --head_mode multihead \
  --enhanced_green_head expD \
  --trainable_families green8 \
  --freeze_backbone True \
  --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --max_steps 50 --max_epoch 1 --train_batch_size 8 --num_workers 2 \
  --learning_rate 0.00005 --lr_schedule 100 \
  --lpr_max_len 8 \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_channel_order bgr \
  --ocr_quad_pad_ratio 0.0 \
  --province_balance_mode inv_sqrt \
  --train_brightness_aug_max 120.0 \
  --gray3_prob 0.0 \
  --first_char_aux_weight 0.3 \
  --cuda True --phase_train True \
  --save_interval 25 --test_interval 25 \
  --save_folder "$EXP_DIR" 2>&1 | tee "${EXP_DIR}/train.log"
