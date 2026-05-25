#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/green_h34/H34C_edgefit_allprov_v3_zhe_guard_yuehu_restore/LPRNet__iteration_2000.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
GREEN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v3_zhe_guard_yuehu_restore.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_h34/H34C_edgefit_allprov_v3_zhe_guard_yuehu_restore

echo "[RESUME] H34C: resuming from epoch 6 checkpoint (iteration 2000)"
echo "[CONFIG] same as before, continuing to epoch 15"

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
  --train_manifest "$GREEN_MANIFEST" \
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
  --learning_rate 0.00003 \
  --lr_schedule 7 10 15 20 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_strategy balanced_recovery \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 15 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  2>&1 | tee -a "$SAVE_DIR/train.log"

echo "[EVAL] Running green8 metrics only..."
$PY eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json"

echo "[DONE] H34C complete"
echo "[RESULT] Check $SAVE_DIR/eval_green8_metrics_only.json"
