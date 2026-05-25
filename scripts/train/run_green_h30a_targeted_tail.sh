#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_h30a_targeted_tail.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_h30/H30A_targeted_tail
CHECK_JSON=$SAVE_DIR/preflight_manifest_check.json

mkdir -p /home/wzzz/LPRNet/experiments/green_h30
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

# Preflight 1: parameter sanity
SELECTION_STRATEGY=proxy_exact
SELECTION_PROXY_MODE=stratified
case "$SELECTION_STRATEGY" in
  proxy_exact|balanced_tuple|balanced_recovery) ;;
  *) echo "[FATAL] invalid selection strategy: $SELECTION_STRATEGY"; exit 2 ;;
esac
case "$SELECTION_PROXY_MODE" in
  sequential|stratified) ;;
  *) echo "[FATAL] invalid selection proxy mode: $SELECTION_PROXY_MODE"; exit 2 ;;
esac

# Preflight 2: manifest audit must pass
$PY check_manifest_h30a_targeted_tail.py | tee "$CHECK_JSON"
$PY - <<'PY'
import json, sys
p='/home/wzzz/LPRNet/experiments/green_h30/H30A_targeted_tail/preflight_manifest_check.json'
with open(p,'r',encoding='utf-8') as f:
    rep=json.load(f)
a=rep['audit']
if a['train_intersects_raw_synthetic_val'] != 0:
    print('[FATAL] train intersects raw synthetic val')
    sys.exit(3)
if a['train_intersects_raw_synthetic_test'] != 0:
    print('[FATAL] train intersects raw synthetic test')
    sys.exit(4)
if a['non_target_repeat_groups'] != 0:
    print('[FATAL] found non-target repeated groups')
    sys.exit(5)
print('[OK] preflight manifest audit passed')
PY

echo "[RUN] H30-A: targeted tail provinces on top of H29B no-leak 40% baseline"
echo "[CONFIG] Only targeted provinces boosted: 苏/沪/闽/浙"

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
  --selection_strategy "$SELECTION_STRATEGY" \
  --selection_proxy_mode "$SELECTION_PROXY_MODE" \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 15 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  2>&1 | tee "$SAVE_DIR/train.log"

echo "[EVAL] Running green8 family-aware metrics only..."
$PY eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

echo "[DONE] H30-A complete"
