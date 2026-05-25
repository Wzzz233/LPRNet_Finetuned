#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v1.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_h34/H34A_edgefit_allprov_v1
BASELINE_JSON=/home/wzzz/LPRNet/experiments/green_h29/H29B_anhui40_noleak/eval_green8_metrics_only.json

mkdir -p /home/wzzz/LPRNet/experiments/green_h34
if [ -d "$SAVE_DIR" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  mv "$SAVE_DIR" "${SAVE_DIR}_bak_${TS}"
fi
mkdir -p "$SAVE_DIR"

echo "[RUN] H34-A: H29B no-leak + edgefit allprov v1 (方案2：皖train=80, 非皖train=240)"
echo "[BASE] train_manifest=$TRAIN_MANIFEST"
echo "[EVAL] eval_manifest=$EVAL_MANIFEST"

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
  --selection_decode_mode family_aware_beam \
  --selection_beam_size 20 \
  --selection_beam_topk 12 \
  --ctc_loss_type standard \
  --first_char_loss_type ce \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 15 \
  2>&1 | tee "$SAVE_DIR/train.log"

echo "[EVAL] Running green8 family-aware metrics only..."
$PY eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

echo "[STATUS] Refresh shared status files"
$PY update_training_status.py \
  --log "$SAVE_DIR/train.log" \
  --process-keyword H34A_edgefit_allprov_v1 \
  --final-weight "$SAVE_DIR/Final_LPRNet_model.pth" \
  --eval-json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --baseline-json "$BASELINE_JSON" \
  --out-json /home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.json \
  --out-md /home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.md

echo "[DONE] H34-A complete"
