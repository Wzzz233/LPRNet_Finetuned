#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
INIT=/home/wzzz/LPRNet/experiments/debug_checks/epoch0_allfamilies/init_only_multihead_allfamilies.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
GREEN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv
ROOT_EXP=/home/wzzz/LPRNet/experiments/green_h21
mkdir -p "$ROOT_EXP"

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
  --train_manifest "$GREEN_MANIFEST"
  --test_manifest "$EVAL_MANIFEST"
  --pretrained_model "$INIT"
  --head_mode multihead
  --freeze_backbone true
  --trainable_families green8
  --province_balance_mode inv_sqrt
  --strata_balance_mode none
  --main_group_by family
  --main_group_ratios green8=1.0
  --main_group_clip 1.0
  --enhanced_green_head expD
  --train_batch_size 64
  --test_batch_size 120
  --learning_rate 0.0003
  --lr_schedule 7 10 15 20
  --selection_proxy_eval_samples 2000
)

run_eval() {
  local save_dir="$1"
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
    --pretrained_model "$save_dir/Final_LPRNet_model.pth" \
    --test_batch_size 300 \
    --decode_mode family_aware_beam \
    --beam_size 20 \
    --beam_topk 12 \
    --out_json "$save_dir/eval_family_aware.json" \
    2>&1 | tee "$save_dir/eval_family_aware.log"
}

run_one() {
  local name="$1"
  shift
  local save_dir="$ROOT_EXP/$name"
  rm -rf "$save_dir"
  mkdir -p "$save_dir"
  echo "[RUN] $name"
  $PY train_LPRNet.py \
    --save_folder "$save_dir/" \
    --max_epoch 15 \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$save_dir/train.log"
  run_eval "$save_dir"
}

# H2-1a: segmented per-position/sequence auxiliary
run_one H21A_segmented_rear_seq \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 2 \
  --rear_seq_start_step 4

# H2-1b: province auxiliary control
run_one H21B_province_aux \
  --first_char_aux_weight 0.30 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.0

echo "[DONE] H2-1a / H2-1b complete"
