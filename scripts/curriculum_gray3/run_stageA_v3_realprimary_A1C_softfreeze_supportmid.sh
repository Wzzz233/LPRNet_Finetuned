#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

A0_CKPT_FILE="experiments/curriculum_gray3_stageA_v3_realprimary_A0/proxy_eval_v3/recommended_checkpoint.txt"
if [[ ! -f "$A0_CKPT_FILE" ]]; then
  echo "[FATAL] missing A0 recommended checkpoint file: $A0_CKPT_FILE" >&2
  exit 2
fi
INIT=$(cat "$A0_CKPT_FILE" | tr -d '\r\n')
TRAIN_MANIFEST="manifests/curriculum_gray3_stagea_v3_realprimary/train_A1B.csv"
VAL_MANIFEST="manifests/curriculum_gray3_stagea_v3_realprimary/val_A1B.csv"
SAVE_DIR="experiments/curriculum_gray3_stageA_v3_realprimary_A1C_softfreeze_supportmid"
LOG_PATH="$SAVE_DIR/train.log"

if [[ ! -f "$INIT" ]]; then
  echo "[FATAL] A0 recommended checkpoint does not exist: $INIT" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VAL_MANIFEST" ]]; then
  echo "[FATAL] missing manifest(s): $TRAIN_MANIFEST $VAL_MANIFEST" >&2
  exit 2
fi
if [[ -e "$SAVE_DIR" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  mv "$SAVE_DIR" "${SAVE_DIR}.bak_${ts}"
  echo "[INFO] backed up old SAVE_DIR to ${SAVE_DIR}.bak_${ts}"
fi
mkdir -p "$SAVE_DIR"

python3 src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$VAL_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --trainable_families normal7,green8 \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_channel_order bgr \
  --ocr_preproc gray3 \
  --gray3_prob 1.0 \
  --main_group_by family \
  --train_batch_size 128 \
  --test_batch_size 120 \
  --max_epoch 6 \
  --learning_rate 0.0005 \
  --lr_schedule 3 5 \
  --freeze_backbone True \
  --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --save_folder "$SAVE_DIR" \
  --num_workers 4 \
  --seed 45 \
  --cuda True \
  > "$LOG_PATH" 2>&1
