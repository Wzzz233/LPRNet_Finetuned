#!/usr/bin/env bash
# Step3: Province-region degradation training
# From unfreeze best, unfreeze backbone.18/19/20, LR=1e-4, first_char_aux=0.30
set -euo pipefail
cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

# ── Paths ─────────────────────────────────────────────────────────
INIT=/home/wzzz/LPRNet/experiments/green_e12_pose_replace_unfreeze_bk18_20/best_LPRNet_model.pth
BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_replace_pose_v3_append.csv
DEGRADE_TRAIN=/home/wzzz/LPRNet/manifests/province_degrade_train_v1/train_province_degrade_v1.csv
STRESS_TEST=/home/wzzz/LPRNet/manifests/province_stress_pose_val_v1/province_stress_pose_val_v1.csv
TEST_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_pose_replace_test.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e12_province_degrade_unfreeze

# Build combined train manifest (append degrade data to base)
COMBINED=/tmp/manifest_province_degrade_combined.csv
$PY -c "
import csv
# Read base manifest
rows = []
with open('$BASE_MANIFEST', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        r = {k: row.get(k, '') for k in fields}
        rows.append(r)
# Read degrade train
deg = []
with open('$DEGRADE_TRAIN', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        r = {k: row.get(k, '') for k in fields}
        deg.append(r)
# Combine
all_rows = rows + deg
with open('$COMBINED', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(all_rows)
print(f'Base: {len(rows)}, Degrade: {len(deg)}, Total: {len(all_rows)}')
"

for f in "$INIT" "$COMBINED" "$TEST_MANIFEST" "$STRESS_TEST"; do
  [[ -f "$f" ]] || { echo "[ERR] missing: $f" >&2; exit 1; }
done

mkdir -p /home/wzzz/LPRNet/experiments
if [[ -d "$SAVE_DIR" ]]; then
  mv "$SAVE_DIR" "${SAVE_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$SAVE_DIR"

echo "[RUN] Province-degrade training from unfreeze best"
echo "[INIT] $INIT"
echo "[COMBINED_TRAIN] $COMBINED"

$PY src/training/train_LPRNet.py \
  --cuda true --data_mode manifest \
  --ocr_channel_order bgr --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 --ocr_quad_pad_ratio 0.0 \
  --train_manifest "$COMBINED" \
  --test_manifest "$TEST_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --freeze_backbone true \
  --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --trainable_families green8 \
  --province_balance_mode inv_sqrt --strata_balance_mode none \
  --main_group_by family --main_group_ratios green8=1.0 --main_group_clip 1.0 \
  --enhanced_green_head expD \
  --train_batch_size 64 --test_batch_size 120 \
  --learning_rate 0.0001 --lr_schedule 2 4 5 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --first_char_aux_weight 0.30 \
  --rear_seq_aux_weight 0.30 --rear_seq_drop_chars 1 --rear_seq_start_step 4 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 5 \
  --train_brightness_aug_max 120.0 \
  --gray3_prob 0.0 \
  2>&1 | tee "$SAVE_DIR/train.log"

echo "[DONE] Province-degrade training complete"
