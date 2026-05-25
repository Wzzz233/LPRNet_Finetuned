#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
BASE_WEIGHT=/home/wzzz/LPRNet/experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth
ROOT_EXP=/home/wzzz/LPRNet/experiments/green_balance_round1
REPORT_ROOT=/home/wzzz/LPRNet/reports/green_balance_round1
mkdir -p "$ROOT_EXP" "$REPORT_ROOT"

run_one() {
  local name="$1"
  local manifest="$2"
  local save_dir="$ROOT_EXP/$name"
  mkdir -p "$save_dir"
  echo "[RUN] $name manifest=$manifest save_dir=$save_dir"

  local final_model="$save_dir/Final_LPRNet_model.pth"
  if [[ ! -f "$final_model" ]]; then
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
      --pretrained_model "$BASE_WEIGHT" \
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
      --main_group_by preprocess_group \
      --main_group_ratios ccpd_board=0.90,plain_plate=0.10 \
      --main_group_clip 2.0 \
      --early_stop_patience 2 \
      --early_stop_regression_patience 2 \
      --early_stop_regression_pp 0.2 \
      --early_stop_start_epoch 3 \
      --save_folder "$save_dir/" \
      2>&1 | tee "$save_dir/train.log"
  else
    echo "[SKIP] training already finished for $name: $final_model"
  fi

  if [[ ! -f "$final_model" ]]; then
    echo "[WARN] final model not found at expected path: $final_model"
    alt=$(find "$ROOT_EXP" -maxdepth 2 -type f -name 'Final_LPRNet_model.pth' | grep "/$name/" | head -n 1 || true)
    if [[ -n "$alt" ]]; then
      final_model="$alt"
      echo "[INFO] using alternate final model path: $final_model"
    else
      echo "[ERROR] final model missing for $name" >&2
      exit 2
    fi
  fi

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
      --pretrained_model "$final_model" \
      --test_batch_size 120 \
      --decode_mode family_aware_beam \
      --beam_size 20 \
      --beam_topk 12 \
      --out_json "$save_dir/eval_family_aware.json" \
      2>&1 | tee "$save_dir/eval_family_aware.log"
  else
    echo "[SKIP] family-aware eval already finished for $name"
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
      --pretrained_model "$final_model" \
      --test_batch_size 120 \
      --decode_mode greedy \
      --out_json "$save_dir/eval_greedy.json" \
      2>&1 | tee "$save_dir/eval_greedy.log"
  else
    echo "[SKIP] greedy eval already finished for $name"
  fi
}

run_one baseline /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_baseline_v1.csv
run_one lite /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_lite_v1.csv
run_one mid /home/wzzz/LPRNet/manifests/unified_manifest_green_balance_mid_v1.csv

echo "[DONE] green balance round1 complete"
