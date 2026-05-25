#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

INIT=/home/wzzz/LPRNet/experiments/green_h32/H32B_classbalanced_plus_focalctc/Final_LPRNet_model.pth
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/firstchar_batch1/D2_firstchar_manifest_green8_normal7_v1_train.csv
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/firstchar_batch1/G0_h36c_no_pos0_baseline

mkdir -p /home/wzzz/LPRNet/experiments/firstchar_batch1
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

$PY src/training/train_LPRNet.py \
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
  --trainable_families all \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --main_group_by none \
  --enhanced_green_head expD \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 7 10 15 20 \
  --selection_proxy_eval_samples 0 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_tuple \
  --ctc_loss_type focal \
  --focal_ctc_alpha 0.5 \
  --focal_ctc_gamma 2.0 \
  --first_char_loss_type class_balanced_ce \
  --first_char_cb_beta 0.999 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --pos0_head_cols 0 \
  --pos0_head_weight 0.0 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 10 \
  2>&1 | tee "$SAVE_DIR/train.log"

$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"
