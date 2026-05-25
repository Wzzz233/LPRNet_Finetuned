#!/usr/bin/env bash
# =============================================================================
# GATE CHECK: Verify normal7 head viability BEFORE unified training
# =============================================================================
# Must achieve exact_plate_acc > 0.90 on CCPD2019 board test before proceeding
# to any unified experiment.  This prevents the U1 "normal7=0.0" disaster.
# =============================================================================
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

INIT="/home/wzzz/LPRNet/models/weights/weights_official/Final_LPRNet_model.pth"
NORMAL7_MANIFEST="/home/wzzz/LPRNet/manifests/normal7_test_only_v1.csv"
SAVE_DIR="/home/wzzz/LPRNet/experiments/gatecheck_normal7_head_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAVE_DIR"

echo "[GATE CHECK] normal7 head viability"
echo "[INIT] $INIT"

# Quick 5-epoch smoke test with unfrozen backbone
$PY src/training/train_LPRNet.py \
  --cuda true \
  --data_mode manifest \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --train_manifest "$NORMAL7_MANIFEST" \
  --test_manifest "$NORMAL7_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --enhanced_green_head expD \
  --freeze_backbone false \
  --trainable_families normal7 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 5 \
  --test_interval 500 \
  2>&1 | tee "$SAVE_DIR/train.log"

# Evaluate
$PY src/evaluation/eval_lpr_detailed.py \
  --data_mode manifest \
  --test_img_dirs "$NORMAL7_MANIFEST" \
  --txt_file dummy.txt \
  --pretrained_model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --cuda True \
  --decode_mode greedy \
  --head_mode multihead \
  --enhanced_green_head expD \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_channel_order bgr \
  --out_json "$SAVE_DIR/eval.json" \
  --test_batch_size 128 \
  2>&1 | tee "$SAVE_DIR/eval.log"

echo "[GATE CHECK DONE] Results in $SAVE_DIR/eval.json"
echo "If normal7 exact < 0.90: DO NOT proceed to unified training."
