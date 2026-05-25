#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

# E8B: full gray3 pipeline with brightness augmentation
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e6c_boardnative_append_v1_20260413.csv
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e8b_gray3_brightness

mkdir -p /home/wzzz/LPRNet/experiments
if [ -d "$SAVE_DIR" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  mv "$SAVE_DIR" "${SAVE_DIR}__bak_${TS}"
fi
mkdir -p "$SAVE_DIR"

echo "[RUN] E8B: full gray3 with brightness augmentation (gray3_prob=1.0, max_delta=120, 50% chance)"
echo "[INIT] $INIT"
echo "[NOTE] Starting from official init weights (not E6C-A)"
echo "[SAVE_DIR] $SAVE_DIR"

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
  --gray3_prob 1.0 \
  2>&1 | tee "$SAVE_DIR/train.log"

echo "[EVAL] E8B: green8 metrics (BGR original)..."
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only_bgr.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only_bgr.log"

echo "[EVAL] E8B: green8 metrics (gray3)..."
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only_gray3.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc gray3 \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only_gray3.log"

echo "[EVAL] E8B: family-aware (gray3)..."
$PY src/evaluation/eval_lpr_detailed.py \
  --cuda true \
  --data_mode manifest \
  --test_img_dirs "$EVAL_MANIFEST" \
  --txt_file "$EVAL_MANIFEST" \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc gray3 \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --head_mode multihead \
  --enhanced_green_head expD \
  --pretrained_model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --test_batch_size 300 \
  --decode_mode family_aware_beam \
  --beam_size 20 \
  --beam_topk 12 \
  --out_json "$SAVE_DIR/eval_family_aware_gray3.json" \
  2>&1 | tee "$SAVE_DIR/eval_family_aware_gray3.log"

echo "[EVAL] E8B: cluster1 track (gray3)..."
$PY src/evaluation/eval_board_native_track_stability.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest /home/wzzz/LPRNet/tmp/green_board_native_cluster1_benchmark_manifest_20260413.csv \
  --out_json "$SAVE_DIR/eval_cluster1_track_gray3.json" \
  --ocr_preproc gray3 \
  2>&1 | tee "$SAVE_DIR/eval_cluster1_track_gray3.log"

echo "[EVAL] E8B: cluster1 track (BGR original)..."
$PY src/evaluation/eval_board_native_track_stability.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest /home/wzzz/LPRNet/tmp/green_board_native_cluster1_benchmark_manifest_20260413.csv \
  --out_json "$SAVE_DIR/eval_cluster1_track_bgr.json" \
  2>&1 | tee "$SAVE_DIR/eval_cluster1_track_bgr.log"

echo "[EVAL] E8B: E7 board-native (gray3)..."
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest /home/wzzz/LPRNet/tmp/green_board_native_e7_v2/manifests/eval_manifest.csv \
  --out_json "$SAVE_DIR/eval_e7_board_native_gray3.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc gray3 \
  2>&1 | tee "$SAVE_DIR/eval_e7_board_native_gray3.log"

echo "[DONE] E8B complete"