#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

# E12 Final 起跑 + Pose-quad 替换数据 append 微调
INIT=/home/wzzz/LPRNet/experiments/green_e12_e9c_append_boarddump_anticollapse_5prov_1200_stage2/Final_LPRNet_model.pth
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_replace_pose_v3_append.csv
TEST_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_pose_replace_test.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e12_pose_replace_append_stage2

for f in "$INIT" "$TRAIN_MANIFEST" "$TEST_MANIFEST"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

mkdir -p /home/wzzz/LPRNet/experiments
if [[ -d "$SAVE_DIR" ]]; then
  mv "$SAVE_DIR" "${SAVE_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$SAVE_DIR"

echo "[RUN] E12 → Pose-replace append stage2"
echo "[INIT] $INIT"
echo "[TRAIN] $TRAIN_MANIFEST"
echo "[SAVE] $SAVE_DIR"

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
  --test_manifest "$TEST_MANIFEST" \
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
  --learning_rate 0.00005 \
  --lr_schedule 2 4 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --first_char_aux_weight 0.15 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 4 \
  --train_brightness_aug_max 120.0 \
  --gray3_prob 0.0 \
  2>&1 | tee "$SAVE_DIR/train.log"

echo "[DONE] E12 Pose-replace stage2 complete"
