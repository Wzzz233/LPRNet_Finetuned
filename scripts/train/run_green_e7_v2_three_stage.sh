#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# E7 三阶段训练脚本
# 策略: 分阶段 + 关键省份替据
# 基线: 与E2-V4相同 (official权重)
# ============================================================

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

# 基线权重 - 与E2-V4保持一膴
BASELINE_INIT=/home/wzzz/LPRNet/models/weights/weights_official/Final_LPRNet_model.pth

# E7数据路径
E7_SIMPLE_ROOT=/home/wzzz/LPRNet/tmp/green_board_native_e7_v2_simple
E7_MEDIUM_ROOT=/home/wzzz/LPRNet/tmp/green_board_native_e7_v2_medium
E7_FULL_ROOT=/home/wzzz/LPRNet/tmp/green_board_native_e7_v2

# 评估数据
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
CLUSTER1_MANIFEST=/home/wzzz/LPRNet/tmp/green_board_native_cluster1_benchmark_manifest_20260413.csv

# 实验目录
SAVE_DIR_BASE=/home/wzzz/LPRNet/experiments/green_e7_v2_three_stage

mkdir -p "$SAVE_DIR_BASE"

echo "======================================"
echo "E7 三阶段训练 - 与E2同基线"
echo "======================================"
echo "基线权重: $BASELINE_INIT"
echo ""

# ============================================================
# 阶段一: 基础学习 (geometry_clean)
# ============================================================
STAGE1_DIR="${SAVE_DIR_BASE}/stage1_simple"
echo "[阶段1/3] 基础学习 - geometry_clean (亮面入门)"
echo "--------------------------------------------"

mkdir -p "$STAGE1_DIR"

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
  --train_manifest /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv \
  --test_manifest "$EVAL_MANIFEST" \
  --secondary_train_img_dirs "$E7_SIMPLE_ROOT" \
  --secondary_train_txt_file "$E7_SIMPLE_ROOT/train_labels.txt" \
  --secondary_train_sample_weight 8.0 \
  --pretrained_model "$BASELINE_INIT" \
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
  --lr_schedule 5 8 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --save_folder "$STAGE1_DIR/" \
  --max_epoch 8 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  2>&1 | tee "$STAGE1_DIR/train.log"

echo "[阶段1完成] 模型保存在: $STAGE1_DIR/Final_LPRNet_model.pth"
echo ""

# ============================================================
# 阶段二: 难度进阶 (board_mid_occ)
# ============================================================
STAGE2_DIR="${SAVE_DIR_BASE}/stage2_medium"
echo "[阶段2/3] 难度进阶 - board_mid_occ (中等难度)"
echo "--------------------------------------------"

mkdir -p "$STAGE2_DIR"

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
  --train_manifest /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv \
  --test_manifest "$EVAL_MANIFEST" \
  --secondary_train_img_dirs "$E7_FULL_ROOT" \
  --secondary_train_txt_file "$E7_FULL_ROOT/train_labels.txt" \
  --secondary_train_sample_weight 6.0 \
  --pretrained_model "$STAGE1_DIR/Final_LPRNet_model.pth" \
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
  --learning_rate 0.0002 \
  --lr_schedule 5 10 15 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --save_folder "$STAGE2_DIR/" \
  --max_epoch 12 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  2>&1 | tee "$STAGE2_DIR/train.log"

echo "[阶段2完成] 模型保存在: $STAGE2_DIR/Final_LPRNet_model.pth"
echo ""

# ============================================================
# 阶段三: 知识融合 (全量微调)
# ============================================================
STAGE3_DIR="${SAVE_DIR_BASE}/stage3_finetune"
echo "[阶段3/3] 知识融合 - 全量数据微调"
echo "--------------------------------------------"

mkdir -p "$STAGE3_DIR"

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
  --train_manifest /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv \
  --test_manifest "$EVAL_MANIFEST" \
  --secondary_train_img_dirs "$E7_FULL_ROOT" \
  --secondary_train_txt_file "$E7_FULL_ROOT/train_labels.txt" \
  --secondary_train_sample_weight 2.0 \
  --pretrained_model "$STAGE2_DIR/Final_LPRNet_model.pth" \
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
  --learning_rate 0.0001 \
  --lr_schedule 3 5 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --save_folder "$STAGE3_DIR/" \
  --max_epoch 6 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  2>&1 | tee "$STAGE3_DIR/train.log"

echo "[阶段3完成] 最终模型: $STAGE3_DIR/Final_LPRNet_model.pth"
echo ""

# ============================================================
# 最终评估
# ============================================================
echo "======================================"
echo "开始最终评估..."
echo "======================================"

# Green8评估
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$STAGE3_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$STAGE3_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  2>&1 | tee "$STAGE3_DIR/eval_green8.log"

# Cluster1 Track评估
$PY src/evaluation/eval_board_native_track_stability.py \
  --model "$STAGE3_DIR/Final_LPRNet_model.pth" \
  --manifest "$CLUSTER1_MANIFEST" \
  --out_json "$STAGE3_DIR/eval_board_native_cluster1_track.json" \
  --batch_size 64 \
  --num_workers 2 \
  2>&1 | tee "$STAGE3_DIR/eval_cluster1.log"

echo "======================================"
echo "三阶段训练完成!"
echo "======================================"
echo "最终模型: $STAGE3_DIR/Final_LPRNet_model.pth"
echo "评估结果: $STAGE3_DIR/eval_green8_metrics_only.json"
