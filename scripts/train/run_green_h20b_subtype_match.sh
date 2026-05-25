#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
GREEN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_h20b_subtype_match.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_h20b/H20B_subtype_match_quick
mkdir -p /home/wzzz/LPRNet/experiments/green_h20b
rm -rf "$SAVE_DIR"
mkdir -p "$SAVE_DIR"

COMMON_ARGS=(
  --cuda true
  --data_mode manifest
  --ocr_channel_order bgr
  --ocr_crop_mode obb_warp
  --ocr_resize_mode letterbox
  --ocr_resize_kernel nn
  --ocr_preproc none
  --ocr_min_occ_ratio 0.90
  --ocr_quad_pad_ratio 0.0
  --pretrained_model "$INIT"
  --head_mode multihead
  --freeze_backbone true
  --trainable_families green8
  --province_balance_mode inv_sqrt
  --strata_balance_mode none
  --first_char_aux_weight 0.4
  --second_char_aux_weight 0.0
  --ne_type_aux_weight 0.0
  --selection_decode_mode family_aware_beam
  --selection_beam_size 20
  --selection_beam_topk 12
  --main_group_by family
  --main_group_ratios green8=1.0
  --main_group_clip 1.0
  --enhanced_green_head expD
)

echo "[RUN] H20B subtype-match quick control"
$PY train_LPRNet.py \
  --manifest "$GREEN_MANIFEST" \
  --max_epoch 10 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0003 \
  --lr_schedule 7 10 \
  --selection_proxy_eval_samples 2000 \
  --save_folder "$SAVE_DIR/" \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "$SAVE_DIR/train.log"

$PY eval_lpr_detailed.py \
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

echo "[DONE] H20B subtype-match quick control complete"
