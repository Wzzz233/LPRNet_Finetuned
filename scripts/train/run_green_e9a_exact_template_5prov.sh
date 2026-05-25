#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

INIT=/home/wzzz/LPRNet/experiments/green_e8c_brightness_replace/Final_LPRNet_model.pth
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e9a_exact_template_5prov_1800.csv
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e9a_exact_template_5prov_1800

mkdir -p /home/wzzz/LPRNet/experiments
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

echo "[RUN] E9A 5prov exact-template"
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
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 15 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --train_brightness_aug_max 120.0 \
  --gray3_prob 0.0 \
  2>&1 | tee "$SAVE_DIR/train.log"

$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

$PY src/evaluation/eval_board_native_track_stability.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest /home/wzzz/LPRNet/tmp/green_board_native_cluster1_benchmark_manifest_20260413.csv \
  --out_json "$SAVE_DIR/eval_cluster1_track.json" \
  2>&1 | tee "$SAVE_DIR/eval_cluster1_track.log"

$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest /home/wzzz/LPRNet/tmp/green_board_native_e7_v2/manifests/eval_manifest.csv \
  --out_json "$SAVE_DIR/eval_e7_board_native.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  2>&1 | tee "$SAVE_DIR/eval_e7_board_native.log"

echo "[DONE] E9A complete"
