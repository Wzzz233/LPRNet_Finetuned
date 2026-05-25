#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
BLUE_WEIGHT=/home/wzzz/LPRNet/experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth
OFFICIAL_WEIGHT=/home/wzzz/LPRNet/weights_official/Final_LPRNet_model.pth
MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1.csv
ROOT_EXP=/home/wzzz/LPRNet/experiments/green_multihead_round1
REPORT_ROOT=/home/wzzz/LPRNet/reports/green_balance_round1
mkdir -p "$ROOT_EXP" "$REPORT_ROOT"

run_one() {
  local name="$1"
  local init_weight="$2"
  local head_mode="$3"
  local save_dir="$ROOT_EXP/$name"
  mkdir -p "$save_dir"
  echo "[RUN] $name manifest=$MANIFEST head_mode=$head_mode init=$init_weight"

  local final_model="$save_dir/Final_LPRNet_model.pth"
  if [[ ! -f "$final_model" ]]; then
    $PY train_LPRNet.py \
      --cuda true \
      --data_mode manifest \
      --manifest "$MANIFEST" \
      --ocr_channel_order bgr \
      --ocr_crop_mode obb_warp \
      --ocr_resize_mode letterbox \
      --ocr_resize_kernel nn \
      --ocr_preproc none \
      --ocr_min_occ_ratio 0.90 \
      --ocr_quad_pad_ratio 0.0 \
      --pretrained_model "$init_weight" \
      --head_mode "$head_mode" \
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

  if [[ ! -f "$save_dir/eval_family_aware.json" ]]; then
    $PY eval_lpr_detailed.py \
      --cuda true \
      --data_mode manifest \
      --test_img_dirs "$MANIFEST" \
      --txt_file "$MANIFEST" \
      --ocr_channel_order bgr \
      --ocr_crop_mode obb_warp \
      --ocr_resize_mode letterbox \
      --ocr_resize_kernel nn \
      --ocr_preproc none \
      --ocr_min_occ_ratio 0.90 \
      --ocr_quad_pad_ratio 0.0 \
      --head_mode "$head_mode" \
      --pretrained_model "$final_model" \
      --test_batch_size 120 \
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
      --test_img_dirs "$MANIFEST" \
      --txt_file "$MANIFEST" \
      --ocr_channel_order bgr \
      --ocr_crop_mode obb_warp \
      --ocr_resize_mode letterbox \
      --ocr_resize_kernel nn \
      --ocr_preproc none \
      --ocr_min_occ_ratio 0.90 \
      --ocr_quad_pad_ratio 0.0 \
      --head_mode "$head_mode" \
      --pretrained_model "$final_model" \
      --test_batch_size 120 \
      --decode_mode greedy \
      --out_json "$save_dir/eval_greedy.json" \
      2>&1 | tee "$save_dir/eval_greedy.log"
  fi
}

run_one blue_expert_multihead_aggr "$BLUE_WEIGHT" multihead
run_one official_singlehead_aggr "$OFFICIAL_WEIGHT" single
run_one official_multihead_aggr "$OFFICIAL_WEIGHT" multihead

echo "[DONE] green multihead round1 complete"
