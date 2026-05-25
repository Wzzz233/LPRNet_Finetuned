#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

# Mother: StageB1A best checkpoint (already soft-freeze tuned on B1A difficulty)
INIT="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservative/best_LPRNet_model.pth"
EXPECTED_SHA="989b3a7ca6438034a73e0f8a64929a3ea535b65d67dacc077b95e275436a7897"
TRAIN_MANIFEST="manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/train_B1B_E6AB_preblur_v3_combined.csv"
VAL_MANIFEST="manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_combined/val_B1B_E6AB_combined.csv"
SAVE_DIR="experiments/curriculum_gray3_stageB_v1_B1B_E6AB_preblur_v3_combined"
LOG_PATH="$SAVE_DIR/train.log"

if [[ ! -f "$INIT" ]]; then
  echo "[FATAL] missing StageB1A best mother: $INIT" >&2
  exit 2
fi
sha=$(sha256sum "$INIT" | awk '{print $1}')
if [[ "$sha" != "$EXPECTED_SHA" ]]; then
  echo "[FATAL] StageB1A mother sha mismatch: got=$sha expected=$EXPECTED_SHA" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VAL_MANIFEST" ]]; then
  echo "[FATAL] missing B1B E6AB manifest(s)" >&2
  exit 2
fi

if [[ -e "$SAVE_DIR" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  mv "$SAVE_DIR" "${SAVE_DIR}.bak_${ts}"
  echo "[INFO] backed up old SAVE_DIR to ${SAVE_DIR}.bak_${ts}"
fi
mkdir -p "$SAVE_DIR"

# B1B: same soft-freeze as B1A, lower LR (0.0001 vs 0.0002), shorter (4 epochs vs 6)
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
  --max_epoch 4 \
  --learning_rate 0.0001 \
  --lr_schedule 2 3 \
  --freeze_backbone True \
  --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --save_folder "$SAVE_DIR" \
  --num_workers 4 \
  --seed 47 \
  --cuda True \
  > "$LOG_PATH" 2>&1

echo "[DONE] Training completed. Starting evaluation..."

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --exp_dir "$SAVE_DIR" \
  --manifest_dir manifests/curriculum_gray3_stageb_v1_B1B_E6AB_preblur_v3_new_proxy \
  --out_dir "$SAVE_DIR/proxy_eval_stageB_v1"
