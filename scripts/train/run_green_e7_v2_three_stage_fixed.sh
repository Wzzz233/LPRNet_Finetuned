#!/bin/bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

BASELINE_INIT=/home/wzzz/LPRNet/models/weights/weights_official/Final_LPRNet_model.pth
MANIFEST_DIR=/home/wzzz/LPRNet/manifests/e7_three_stage_v2
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
CLUSTER1_MANIFEST=/home/wzzz/LPRNet/tmp/green_board_native_cluster1_benchmark_manifest_20260413.csv

SAVE_DIR_BASE=/home/wzzz/LPRNet/experiments/green_e7_v2_three_stage_fixed_v2
mkdir -p "$SAVE_DIR_BASE"

echo "======================================"
echo "E7 三阶段训练 (修正版)"
echo "======================================"

# 阶段1
STAGE1_DIR="${SAVE_DIR_BASE}/stage1"
echo "[阶段1/3] 替据90%E2数据"
mkdir -p "$STAGE1_DIR"

$PY src/training/train_LPRNet.py \
  --cuda true --data_mode manifest \
  --ocr_channel_order bgr --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox --ocr_resize_kernel nn \
  --ocr_preproc none --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --train_manifest "${MANIFEST_DIR}/stage1_v2.csv" \
  --test_manifest "$EVAL_MANIFEST" \
  --pretrained_model "$BASELINE_INIT" \
  --head_mode multihead --freeze_backbone true \
  --trainable_families green8 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --enhanced_green_head expD \
  --train_batch_size 64 --test_batch_size 120 \
  --learning_rate 0.0003 --lr_schedule 3 5 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --save_folder "$STAGE1_DIR/" \
  --max_epoch 5 \
  --first_char_aux_weight 0.15 --rear_seq_aux_weight 0.30 \
  2>&1 | tee "$STAGE1_DIR/train.log"

echo "[阶段1完成]"

# 阶段2
STAGE2_DIR="${SAVE_DIR_BASE}/stage2"
echo "[阶段2/3] 替据70%E2数据"
mkdir -p "$STAGE2_DIR"

$PY src/training/train_LPRNet.py \
  --cuda true --data_mode manifest \
  --ocr_channel_order bgr --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox --ocr_resize_kernel nn \
  --ocr_preproc none --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --train_manifest "${MANIFEST_DIR}/stage2_v2.csv" \
  --test_manifest "$EVAL_MANIFEST" \
  --pretrained_model "$STAGE1_DIR/Final_LPRNet_model.pth" \
  --head_mode multihead --freeze_backbone true \
  --trainable_families green8 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --enhanced_green_head expD \
  --train_batch_size 64 --test_batch_size 120 \
  --learning_rate 0.0002 --lr_schedule 3 5 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --save_folder "$STAGE2_DIR/" \
  --max_epoch 5 \
  --first_char_aux_weight 0.15 --rear_seq_aux_weight 0.30 \
  2>&1 | tee "$STAGE2_DIR/train.log"

echo "[阶段2完成]"

# 阶段3
STAGE3_DIR="${SAVE_DIR_BASE}/stage3"
echo "[阶段3/3] 全量微调"
mkdir -p "$STAGE3_DIR"

$PY src/training/train_LPRNet.py \
  --cuda true --data_mode manifest \
  --ocr_channel_order bgr --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox --ocr_resize_kernel nn \
  --ocr_preproc none --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --train_manifest "${MANIFEST_DIR}/stage3_v2.csv" \
  --test_manifest "$EVAL_MANIFEST" \
  --pretrained_model "$STAGE2_DIR/Final_LPRNet_model.pth" \
  --head_mode multihead --freeze_backbone true \
  --trainable_families green8 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --enhanced_green_head expD \
  --train_batch_size 64 --test_batch_size 120 \
  --learning_rate 0.0001 --lr_schedule 3 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --save_folder "$STAGE3_DIR/" \
  --max_epoch 5 \
  --first_char_aux_weight 0.15 --rear_seq_aux_weight 0.30 \
  2>&1 | tee "$STAGE3_DIR/train.log"

echo "[阶段3完成]"

# 评估
echo "开始评估..."
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$STAGE3_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$STAGE3_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 --num_workers 4

$PY src/evaluation/eval_board_native_track_stability.py \
  --model "$STAGE3_DIR/Final_LPRNet_model.pth" \
  --manifest "$CLUSTER1_MANIFEST" \
  --out_json "$STAGE3_DIR/eval_board_native_cluster1_track.json" \
  --batch_size 64 --num_workers 2

echo "完成!"
