#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_h35/H35A_h32b_firstaux018

mkdir -p /home/wzzz/LPRNet/experiments/green_h35
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

echo "[RUN] H35A = H32B + first_char_aux_weight 0.18"

$PY train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$EVAL_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --freeze_backbone true \
  --trainable_families green8 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --enhanced_green_head expD \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 7 10 15 20 \
  --selection_proxy_eval_samples 2000 \
  --selection_strategy proxy_exact \
  --selection_proxy_mode stratified \
  --ctc_loss_type focal \
  --focal_ctc_alpha 0.5 \
  --focal_ctc_gamma 2.0 \
  --first_char_loss_type class_balanced_ce \
  --first_char_cb_beta 0.999 \
  --first_char_aux_weight 0.18 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 15 \
  2>&1 | tee "$SAVE_DIR/train.log"

$PY eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

$PY eval_green8_by_dataset_province.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_by_dataset_province.json" \
  2>&1 | tee "$SAVE_DIR/eval_green8_by_dataset_province.log"

echo "[DONE] H35A complete"
