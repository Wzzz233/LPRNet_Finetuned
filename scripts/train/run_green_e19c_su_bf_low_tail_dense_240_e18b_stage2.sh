#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

INIT=/home/wzzz/LPRNet/experiments/green_e18b_su_bf_transition_dense_600_stage2/Final_LPRNet_model.pth
TRAIN_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e19c_su_bf_low_tail_dense_240.csv
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
CLUSTER1_MANIFEST=/home/wzzz/LPRNet/tmp/green_board_native_cluster1_benchmark_manifest_20260413.csv
E7_MANIFEST=/home/wzzz/LPRNet/tmp/green_board_native_e7_v2/manifests/eval_manifest.csv
CLUSTER2_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv
CLUSTER3_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv
CLUSTER3_TAIL_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_tail_collapse_wsl.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e19c_su_bf_low_tail_dense_240_e18b_stage2
STATUS_JSON=/home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.json
STATUS_MD=/home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.md

for f in "$INIT" "$TRAIN_MANIFEST" "$EVAL_MANIFEST" "$CLUSTER1_MANIFEST" "$E7_MANIFEST" "$CLUSTER2_CSV" "$CLUSTER3_CSV" "$CLUSTER3_TAIL_CSV"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

mkdir -p /home/wzzz/LPRNet/experiments
if [[ -d "$SAVE_DIR" ]]; then
  mv "$SAVE_DIR" "${SAVE_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$SAVE_DIR"

echo "[RUN] E19C E18B->SuBF low+tail dense append stage2"
echo "[INIT] $INIT"
echo "[TRAIN_MANIFEST] $TRAIN_MANIFEST"
echo "[SAVE_DIR] $SAVE_DIR"

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
  --learning_rate 0.00005 \
  --lr_schedule 2 3 \
  --selection_proxy_eval_samples 2000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --save_folder "$SAVE_DIR/" \
  --max_epoch 3 \
  --train_brightness_aug_max 120.0 \
  --gray3_prob 0.0 \
  2>&1 | tee "$SAVE_DIR/train.log"

echo "[EVAL] family-aware detailed"
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

echo "[EVAL] green8 metrics"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

echo "[EVAL] cluster1 track"
$PY src/evaluation/eval_board_native_track_stability.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$CLUSTER1_MANIFEST" \
  --out_json "$SAVE_DIR/eval_cluster1_track.json" \
  2>&1 | tee "$SAVE_DIR/eval_cluster1_track.log"

echo "[EVAL] E7 board-native"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$E7_MANIFEST" \
  --out_json "$SAVE_DIR/eval_e7_board_native.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  2>&1 | tee "$SAVE_DIR/eval_e7_board_native.log"

echo "[EVAL] cluster2 dump replay"
$PY src/utils/replay_dump_compare.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --input-csv "$CLUSTER2_CSV" \
  --out-json "$SAVE_DIR/eval_cluster2_dump.json" \
  --out-csv "$SAVE_DIR/eval_cluster2_dump.csv" \
  2>&1 | tee "$SAVE_DIR/eval_cluster2_dump.log"

echo "[EVAL] cluster3 dump replay"
$PY src/utils/replay_dump_compare.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --input-csv "$CLUSTER3_CSV" \
  --out-json "$SAVE_DIR/eval_cluster3_dump.json" \
  --out-csv "$SAVE_DIR/eval_cluster3_dump.csv" \
  2>&1 | tee "$SAVE_DIR/eval_cluster3_dump.log"

echo "[EVAL] cluster3 tail dump replay"
$PY src/utils/replay_dump_compare.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --input-csv "$CLUSTER3_TAIL_CSV" \
  --out-json "$SAVE_DIR/eval_cluster3_tail_dump.json" \
  --out-csv "$SAVE_DIR/eval_cluster3_tail_dump.csv" \
  2>&1 | tee "$SAVE_DIR/eval_cluster3_tail_dump.log"

echo "[STATUS] refresh shared status"
$PY src/utils/update_training_status.py \
  --log "$SAVE_DIR/train.log" \
  --process-keyword green_e19c_su_bf_low_tail_dense_240_e18b_stage2 \
  --final-weight "$SAVE_DIR/Final_LPRNet_model.pth" \
  --eval-json "$SAVE_DIR/eval_family_aware.json" \
  --out-json "$STATUS_JSON" \
  --out-md "$STATUS_MD"

echo "[DONE] E19C complete"
