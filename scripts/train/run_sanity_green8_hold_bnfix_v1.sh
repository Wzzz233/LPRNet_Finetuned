#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet
OUT_DIR="/home/wzzz/LPRNet/experiments/debug_checks/sanity_green8_hold_bnfix_v1"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
./.conda/bin/python train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --manifest /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --pretrained_model /home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth \
  --head_mode multihead \
  --freeze_backbone true \
  --trainable_families green8 \
  --max_epoch 1 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 3 5 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --first_char_aux_weight 0.4 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --selection_proxy_eval_samples 2000 \
  --selection_decode_mode family_aware_beam \
  --selection_beam_size 20 \
  --selection_beam_topk 12 \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --early_stop_patience 0 \
  --early_stop_regression_patience 0 \
  --save_folder "$OUT_DIR/" \
  2>&1 | tee "$OUT_DIR/train.log"
