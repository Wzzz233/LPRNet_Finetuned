#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

INIT="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth"
EXPECTED_SHA="d6c011e6572b025c5fae618bc253ec4f5d30e6acd60b46012ee6fdef79dcad2a"
MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_softfreeze"
TRAIN_MANIFEST="$MANIFEST_DIR/train_B2C_paradigm3_softfreeze.csv"
VAL_MANIFEST="$MANIFEST_DIR/val_B2C_paradigm3_softfreeze.csv"
SAVE_DIR="experiments/curriculum_gray3_stageB_B2C_paradigm3_softfreeze"
LOG_PATH="$SAVE_DIR/train.log"

if [[ ! -f "$INIT" ]]; then
  echo "[FATAL] missing A1D iter_002000 mother: $INIT" >&2
  exit 2
fi
sha=$(sha256sum "$INIT" | awk '{print $1}')
if [[ "$sha" != "$EXPECTED_SHA" ]]; then
  echo "[FATAL] A1D mother sha mismatch: got=$sha expected=$EXPECTED_SHA" >&2
  exit 2
fi
for f in "$TRAIN_MANIFEST" "$VAL_MANIFEST"; do
  if [[ ! -f "$f" ]]; then
    echo "[FATAL] missing manifest $f" >&2
    exit 2
  fi
done

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
  --adapter_target_families green8 \
  --adapter_hidden_channels 128 \
  --province_head_weight 0.20 \
  --province_num_classes 31 \
  --province_target_families green8 \
  --pos0_head_cols 4 \
  --pos0_head_weight 0.10 \
  --pos0_num_classes 31 \
  --pos0_target_families green8 \
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
  --learning_rate 0.0002 \
  --lr_schedule 3 5 \
  --freeze_backbone False \
  --freeze_bn_stats True \
  --backbone_lr_mult 0.01 \
  --head_lr_mult 1.0 \
  --adapter_lr_mult 1.0 \
  --aux_lr_mult 1.0 \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --save_folder "$SAVE_DIR" \
  --num_workers 4 \
  --seed 47 \
  --cuda True \
  > "$LOG_PATH" 2>&1

echo "[DONE] B2-C paradigm-3 soft-freeze training completed."

# Run evaluation
python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --exp_dir "$SAVE_DIR" \
  --manifest_dir manifests/curriculum_gray3_stageb_v1_difficulty \
  --out_dir "$SAVE_DIR/proxy_eval_stageB_v1"

echo "[DONE] B2-C evaluation completed."
