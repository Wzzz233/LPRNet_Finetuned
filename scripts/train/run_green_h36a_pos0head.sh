#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python

# H36A: H32B 配置 + pos0_head（仅架构变量，不换数据）
# 对比对象：H32B（overall 0.6927）
# 变量：+pos0_head_cols=4 +pos0_head_weight=0.5

INIT=/home/wzzz/LPRNet/experiments/green_h32/H32B_classbalanced_plus_focalctc/Final_LPRNet_model.pth
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_h36/H36A_pos0head_arch_only

mkdir -p /home/wzzz/LPRNet/experiments/green_h36
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

echo "[RUN] H36-A: H32B + pos0_head（架构变量，数据不变）"
echo "[CONFIG] pos0_head_cols=4, pos0_head_weight=0.5, pos0_num_classes=31"
echo "[SUCCESS CRITERIA] 苏 exact/first >= H32B, 沪 exact >= H32B, overall exact >= H32B"

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
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --pos0_head_cols 4 \
  --pos0_head_weight 0.5 \
  --pos0_num_classes 31 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 15 \
  2>&1 | tee "$SAVE_DIR/train.log"

echo "[EVAL] Running detailed metrics..."
$PY eval_lpr_detailed.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --head_mode multihead \
  --enhanced_green_head expD \
  --decode_mode family_aware_beam \
  --pos0_head_cols 4 \
  --pos0_head_mode hybrid \
  2>&1 | tee "$SAVE_DIR/eval_detailed.log" || true

echo "[DONE] H36-A complete. Check $SAVE_DIR/train.log"
