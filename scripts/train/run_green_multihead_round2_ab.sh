#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
BLUE_WEIGHT=/home/wzzz/LPRNet/experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth
ROOT_EXP=/home/wzzz/LPRNet/experiments/green_multihead_round2_ab
mkdir -p "$ROOT_EXP"

run_eval_pair() {
  local save_dir="$1"
  local manifest="$2"
  local head_mode="$3"
  local batch_size="$4"

  if [[ ! -f "$save_dir/eval_family_aware.json" ]]; then
    $PY eval_lpr_detailed.py \
      --cuda true \
      --data_mode manifest \
      --test_img_dirs "$manifest" \
      --txt_file "$manifest" \
      --ocr_channel_order bgr \
      --ocr_crop_mode obb_warp \
      --ocr_resize_mode letterbox \
      --ocr_resize_kernel nn \
      --ocr_preproc none \
      --ocr_min_occ_ratio 0.90 \
      --ocr_quad_pad_ratio 0.0 \
      --head_mode "$head_mode" \
      --pretrained_model "$save_dir/Final_LPRNet_model.pth" \
      --test_batch_size "$batch_size" \
      --decode_mode family_aware_beam \
      --beam_size 20 \
      --beam_topk 12 \
      --out_json "$save_dir/eval_family_aware.json" \
      2>&1 | tee "$save_dir/eval_family_aware.log"
  fi

  if [[ ! -f "$save_dir/eval_greedy.json" ]]; then
    $PY eval_lpr_detailed.py \
      --cuda true \
      --data_mode manifest \
      --test_img_dirs "$manifest" \
      --txt_file "$manifest" \
      --ocr_channel_order bgr \
      --ocr_crop_mode obb_warp \
      --ocr_resize_mode letterbox \
      --ocr_resize_kernel nn \
      --ocr_preproc none \
      --ocr_min_occ_ratio 0.90 \
      --ocr_quad_pad_ratio 0.0 \
      --head_mode "$head_mode" \
      --pretrained_model "$save_dir/Final_LPRNet_model.pth" \
      --test_batch_size "$batch_size" \
      --decode_mode greedy \
      --out_json "$save_dir/eval_greedy.json" \
      2>&1 | tee "$save_dir/eval_greedy.log"
  fi
}

run_one() {
  local name="$1"
  local manifest="$2"
  shift 2
  local save_dir="$ROOT_EXP/$name"
  mkdir -p "$save_dir"
  echo "[RUN] $name manifest=$manifest"

  if [[ ! -f "$save_dir/Final_LPRNet_model.pth" ]]; then
    $PY train_LPRNet.py \
      --cuda true \
      --data_mode manifest \
      --manifest "$manifest" \
      --ocr_channel_order bgr \
      --ocr_crop_mode obb_warp \
      --ocr_resize_mode letterbox \
      --ocr_resize_kernel nn \
      --ocr_preproc none \
      --ocr_min_occ_ratio 0.90 \
      --ocr_quad_pad_ratio 0.0 \
      --pretrained_model "$BLUE_WEIGHT" \
      --head_mode multihead \
      --freeze_backbone true \
      --max_epoch 6 \
      --train_batch_size 64 \
      --test_batch_size 120 \
      --learning_rate 0.0003 \
      --lr_schedule 3 5 \
      --province_balance_mode inv_sqrt \
      --strata_balance_mode none \
      --first_char_aux_weight 0.4 \
      --second_char_aux_weight 0.0 \
      --ne_type_aux_weight 0.0 \
      --selection_proxy_eval_samples 5000 \
      --selection_decode_mode family_aware_beam \
      --selection_beam_size 20 \
      --selection_beam_topk 12 \
      --early_stop_patience 2 \
      --early_stop_regression_patience 2 \
      --early_stop_regression_pp 0.2 \
      --early_stop_start_epoch 3 \
      --save_folder "$save_dir/" \
      "$@" \
      2>&1 | tee "$save_dir/train.log"
  fi

  run_eval_pair "$save_dir" "$manifest" multihead 120
}

run_one A_conservative_manifest_mix \
  /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_conservative_a.csv \
  --main_group_by family \
  --main_group_ratios normal7=0.94,green8=0.06 \
  --main_group_clip 2.0

run_one B_green_head_only \
  /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --trainable_families green8 \
  --selection_proxy_eval_samples 2000

echo "[DONE] round2 A/B complete"
