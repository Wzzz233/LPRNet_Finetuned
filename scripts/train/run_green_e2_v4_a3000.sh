#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

INIT=/home/wzzz/LPRNet/models/weights/weights_official/Final_LPRNet_model.pth
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e2_v4_a3000

mkdir -p /home/wzzz/LPRNet/experiments
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

if [[ ! -f "$INIT" ]]; then
  echo "[FATAL] missing init weight: $INIT" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "[FATAL] missing train manifest: $TRAIN_MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$EVAL_MANIFEST" ]]; then
  echo "[FATAL] missing eval manifest: $EVAL_MANIFEST" >&2
  exit 2
fi

echo "[RUN] green_e2_v4_a3000"
echo "[INIT] $INIT"
echo "[TRAIN_MANIFEST] $TRAIN_MANIFEST"
echo "[EVAL_MANIFEST] $EVAL_MANIFEST"

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
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 15 \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  2>&1 | tee "$SAVE_DIR/train.log"

$PY src/evaluation/eval_lpr_detailed.py \
  --cuda true \
  --data_mode manifest \
  --test_img_dirs "$EVAL_MANIFEST" \
  --txt_file "$EVAL_MANIFEST" \
  --ocr_channel_order bgr \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --ocr_quad_pad_ratio 0.0 \
  --head_mode multihead \
  --enhanced_green_head expD \
  --pretrained_model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --test_batch_size 300 \
  --decode_mode family_aware_beam \
  --beam_size 20 \
  --beam_topk 12 \
  --out_json "$SAVE_DIR/eval_family_aware.json" \
  2>&1 | tee "$SAVE_DIR/eval_family_aware.log"

$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

echo "[DONE] green_e2_v4_a3000"
