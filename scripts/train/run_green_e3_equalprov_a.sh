#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

REPO_ROOT=/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/repo_license_plate_generator
BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed.csv
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
INIT=/home/wzzz/LPRNet/experiments/green_specialist_official_v3_tier3_mix/Final_LPRNet_model.pth
DATA_DIR=/home/wzzz/LPRNet/tmp/green_edgefit_v4_e3_equalprov_a_20260412
V4_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_edgefit_v4_e3_equalprov_a_20260412_train.csv
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e3_equalprov_a_20260412.csv
SUMMARY_JSON=/home/wzzz/LPRNet/reports/GREEN_E3_EQUALPROV_A_MANIFEST_SUMMARY_20260412.json
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e3_equalprov_a_20260412
BASELINE_JSON=/home/wzzz/LPRNet/experiments/green_e2_v4_from_e1_baseline/eval_green8_metrics_only.json

mkdir -p /home/wzzz/LPRNet/experiments
if [ -d "$SAVE_DIR" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  mv "$SAVE_DIR" "${SAVE_DIR}__bak_${TS}"
fi
mkdir -p "$SAVE_DIR"

rm -rf "$DATA_DIR"

echo "[E3-A][GEN] generating equal-prov full dataset"
$PY src/utils/generate_green_edgefit_v4_boardlike_equalprov.py \
  --repo_root "$REPO_ROOT" \
  --out_dir "$DATA_DIR" \
  --dataset_name green_edgefit_v4_boardlike_e3_equalprov_a_20260412 \
  --source_name v4_boardlike_edgefit \
  --seed 20260412 \
  --preview_per_bucket 2 \
  --train_per_province_total 460 \
  --train_geometry_clean 69 \
  --train_board_mid_occ 184 \
  --train_board_low_occ 161 \
  --train_board_extreme_tail 46 \
  --val_geometry_clean 0 \
  --val_board_mid_occ 0 \
  --val_board_low_occ 0 \
  --val_board_extreme_tail 0 \
  --test_geometry_clean 0 \
  --test_board_mid_occ 0 \
  --test_board_low_occ 0 \
  --test_board_extreme_tail 0 \
  2>&1 | tee "$SAVE_DIR/generate.log"

cp "$DATA_DIR/manifests/train_manifest_v4.csv" "$V4_MANIFEST"

echo "[E3-A][MANIFEST] building final train manifest"
$PY src/manifest/build_green_v4_boardlike_equalprov_manifest.py \
  --base-manifest "$BASE_MANIFEST" \
  --v4-manifest "$V4_MANIFEST" \
  --out-manifest "$TRAIN_MANIFEST" \
  --out-summary "$SUMMARY_JSON" \
  2>&1 | tee "$SAVE_DIR/build_manifest.log"

echo "[E3-A][RUN] training"
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

if [[ ! -f "$SAVE_DIR/Final_LPRNet_model.pth" ]]; then
  echo "[E3-A][ERROR] Final model not found" >&2
  exit 1
fi

echo "[E3-A][EVAL] family aware"
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

echo "[E3-A][EVAL] green8 metrics"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

$PY src/utils/update_training_status.py \
  --log "$SAVE_DIR/train.log" \
  --process-keyword green_e3_equalprov_a_20260412 \
  --final-weight "$SAVE_DIR/Final_LPRNet_model.pth" \
  --eval-json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --baseline-json "$BASELINE_JSON" \
  --out-json /home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.json \
  --out-md /home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.md

echo "[DONE] green_e3_equalprov_a_20260412"
