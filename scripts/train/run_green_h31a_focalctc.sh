#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_h31/H31A_focalctc
DIAG_SUMMARY=/home/wzzz/LPRNet/reports/H30A_DIAG_SUMMARY_20260402.md

mkdir -p /home/wzzz/LPRNet/experiments/green_h31
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

# Preflight 1: parameter sanity
SELECTION_STRATEGY=proxy_exact
SELECTION_PROXY_MODE=stratified
CTC_LOSS_TYPE=focal
FOCAL_ALPHA=0.5
FOCAL_GAMMA=2.0
case "$SELECTION_STRATEGY" in
  proxy_exact|balanced_tuple|balanced_recovery) ;;
  *) echo "[FATAL] invalid selection strategy: $SELECTION_STRATEGY"; exit 2 ;;
esac
case "$SELECTION_PROXY_MODE" in
  sequential|stratified) ;;
  *) echo "[FATAL] invalid selection proxy mode: $SELECTION_PROXY_MODE"; exit 2 ;;
esac
case "$CTC_LOSS_TYPE" in
  standard|focal) ;;
  *) echo "[FATAL] invalid ctc loss type: $CTC_LOSS_TYPE"; exit 2 ;;
esac

# Preflight 2: diagnostics summary must exist
if [ ! -f "$DIAG_SUMMARY" ]; then
  echo "[FATAL] missing diag summary: $DIAG_SUMMARY"
  exit 3
fi

echo "[RUN] H31-A: H29B baseline + Focal CTC (main CTC + rear_seq_aux)"
echo "[CONFIG] Keep data/model/board params fixed; only switch CTC loss type"
echo "[DIAG] Using diagnostic summary: $DIAG_SUMMARY"

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
  --ctc_loss_type "$CTC_LOSS_TYPE" \
  --focal_ctc_alpha "$FOCAL_ALPHA" \
  --focal_ctc_gamma "$FOCAL_GAMMA" \
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

echo "[DONE] H31-A complete"
