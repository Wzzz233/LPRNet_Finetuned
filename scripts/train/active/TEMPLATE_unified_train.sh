#!/usr/bin/env bash
# =============================================================================
# TEMPLATE: Unified LPRNet Training (Active Round)
# =============================================================================
# RULES:
#   1. Manifest filename MUST match --ocr_preproc value
#   2. Normal7 head MUST be validated before freeze_backbone=true
#   3. All on-disk extreme assets must be merged into manifest
#   4. Anhui real down-weighted (x0.60), synthetic lightly (x0.90)
# =============================================================================
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

# --- CONFIGURE THESE ---
EXPERIMENT_NAME="CHANGE_ME"
INIT="/home/wzzz/LPRNet/models/weights/weights_official/Final_LPRNet_model.pth"
TRAIN_MANIFEST="/home/wzzz/LPRNet/manifests/CHANGE_ME.csv"
EVAL_MANIFEST="/home/wzzz/LPRNet/manifests/CHANGE_ME.csv"
PREPROC="none"   # MUST match manifest name: none OR gray3
SAVE_DIR="/home/wzzz/LPRNet/experiments/${EXPERIMENT_NAME}"

mkdir -p "$SAVE_DIR"

echo "[RUN] $EXPERIMENT_NAME"
echo "[INIT] $INIT"
echo "[PREPROC] $PREPROC"
echo "[MANIFEST] $TRAIN_MANIFEST"

# --- VALIDATE MANIFEST FIRST ---
$PY scripts/utils/validate_pure_manifest.py "$TRAIN_MANIFEST" \
  --expected_preproc "$PREPROC" \
  --expected_families normal7,green8

# --- TRAIN ---
$PY src/training/train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc "$PREPROC" \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$EVAL_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --enhanced_green_head expD \
  --freeze_backbone false \
  --trainable_families normal7,green8 \
  --province_balance_mode none \
  --strata_balance_mode none \
  --main_group_by family \
  --main_group_ratios normal7=0.60,green8=0.40 \
  --main_group_clip 2.0 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 10 16 22 28 \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_beam_size 20 \
  --selection_beam_topk 12 \
  --selection_strategy balanced_tuple \
  --first_char_aux_weight 0.15 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 30 \
  2>&1 | tee "$SAVE_DIR/train.log"

# --- EVAL: family-aware (primary) ---
$PY src/evaluation/eval_lpr_detailed.py \
  --data_mode manifest \
  --test_img_dirs "$EVAL_MANIFEST" \
  --txt_file dummy.txt \
  --pretrained_model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --cuda True \
  --decode_mode family_aware_beam \
  --beam_size 20 \
  --beam_topk 12 \
  --head_mode multihead \
  --enhanced_green_head expD \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc "$PREPROC" \
  --ocr_channel_order bgr \
  --out_json "$SAVE_DIR/eval_family_aware.json" \
  --test_batch_size 128 \
  --num_workers 4 \
  2>&1 | tee "$SAVE_DIR/eval_family_aware.log"

# --- EVAL: green8 metrics ---
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

# --- EVAL: province breakdown ---
$PY src/evaluation/eval_family_aware_blue_green_by_province.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_family_aware_blue_green_by_province.json" \
  --batch_size 300 \
  --num_workers 4 \
  2>&1 | tee "$SAVE_DIR/eval_family_aware_blue_green_by_province.log"

echo "[DONE] $EXPERIMENT_NAME"
