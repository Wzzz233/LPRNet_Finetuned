#!/bin/bash
# v4 Training A: clean start from old_green
DATE_TAG="20260508"
EXP_DIR="experiments/green_ccpd2019_cvr_v4_from_oldgreen_${DATE_TAG}"
mkdir -p "$EXP_DIR"
cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

python3 -u src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest "manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4_${DATE_TAG}/train_v4_balanced.csv" \
  --test_manifest "manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_${DATE_TAG}/val_cvreplace_v2.csv" \
  --dataset_root /home/wzzz/LPRNet \
  --pretrained_model experiments/green_e12_province_degrade_unfreeze/best_LPRNet_model.pth \
  --save_folder "$EXP_DIR" \
  --head_mode multihead --enhanced_green_head expD \
  --trainable_families green8 \
  --freeze_backbone True --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --max_epoch 10 --train_batch_size 64 --num_workers 4 \
  --learning_rate 0.00005 --lr_schedule 4 6 8 \
  --lpr_max_len 8 \
  --ocr_crop_mode obb_warp --ocr_resize_mode letterbox --ocr_resize_kernel nn \
  --ocr_preproc none --ocr_channel_order bgr --ocr_quad_pad_ratio 0.0 \
  --province_balance_mode inv_sqrt \
  --train_brightness_aug_max 120.0 --gray3_prob 0.0 \
  --first_char_aux_weight 0.3 \
  --save_interval 2000 --test_interval 2000 \
  --cuda True --phase_train True 2>&1 | tee "${EXP_DIR}/train.log"
