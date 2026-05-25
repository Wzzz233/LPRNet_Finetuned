#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:/home/wzzz/LPRNet/src/training:/home/wzzz/LPRNet/src/evaluation:${PYTHONPATH:-}

INIT=/home/wzzz/LPRNet/experiments/green_e12_e9c_append_boarddump_anticollapse_5prov_1200_stage2/Final_LPRNet_model.pth
BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e20a_cluster2_beijing_prefix_contrast_1200.csv
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
E7_MANIFEST=/home/wzzz/LPRNet/tmp/green_board_native_e7_v2/manifests/eval_manifest.csv
CLUSTER2_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv
CLUSTER3_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv
CLUSTER3_TAIL_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_tail_collapse_wsl.csv
BANK_JSON=/home/wzzz/LPRNet/tmp/green_e25a_cluster2_repr_boarddump_bank.json
APPEND_ROOT=/home/wzzz/LPRNet/tmp/green_e25a_cluster2_repr_boarddump_6000
APPEND_LOCAL_MANIFEST=$APPEND_ROOT/manifests/train_manifest_green_e25a_cluster2_repr_boarddump_6000.csv
APPEND_MERGED_MANIFEST=$APPEND_ROOT/manifests/unified_manifest_green_e25a_cluster2_repr_boarddump_6000_merged.csv
STAGEA_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e25a_stageA_targeted_repr_rebuild.csv
STAGEB_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e25a_stageB_full_reintegrate.csv
MANIFEST_SUMMARY=/home/wzzz/LPRNet/tmp/green_e25a_manifest_build_summary.json
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e25a_cluster2_repr_rebuild_twostage
STAGEA_DIR=$SAVE_DIR/stageA_preadapt
STAGEB_DIR=$SAVE_DIR/stageB_reintegrate
STATUS_JSON=/home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.json
STATUS_MD=/home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.md

for f in "$INIT" "$BASE_MANIFEST" "$EVAL_MANIFEST" "$E7_MANIFEST" "$CLUSTER2_CSV" "$CLUSTER3_CSV" "$CLUSTER3_TAIL_CSV" "$BANK_JSON"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

mkdir -p /home/wzzz/LPRNet/experiments /home/wzzz/LPRNet/manifests /home/wzzz/LPRNet/tmp
if [[ -d "$SAVE_DIR" ]]; then
  mv "$SAVE_DIR" "${SAVE_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$SAVE_DIR"

if [[ -d "$APPEND_ROOT" ]]; then
  mv "$APPEND_ROOT" "${APPEND_ROOT}.bak.$(date +%Y%m%d_%H%M%S)"
fi

echo "[PREP] generate E25A target board_dump data"
$PY src/utils/generate_green_e25a_cluster2_repr_boarddump.py \
  --base-manifest "$BASE_MANIFEST" \
  --bank-json "$BANK_JSON" \
  --out-dir "$APPEND_ROOT" \
  --out-manifest "$APPEND_MERGED_MANIFEST" \
  --dataset-name green_e25a_cluster2_repr_boarddump_6000 \
  --source-name e25a_cluster2_repr_boarddump_6000 \
  --appearance-mode board_native

[[ -f "$APPEND_LOCAL_MANIFEST" ]] || { echo "[ERR] append local manifest missing: $APPEND_LOCAL_MANIFEST" >&2; exit 1; }

echo "[PREP] build StageA/StageB manifests"
$PY src/utils/build_green_e25_manifests.py \
  --base-manifest "$BASE_MANIFEST" \
  --append-manifest "$APPEND_LOCAL_MANIFEST" \
  --out-stagea "$STAGEA_MANIFEST" \
  --out-stageb "$STAGEB_MANIFEST" \
  --out-summary "$MANIFEST_SUMMARY"

for f in "$STAGEA_MANIFEST" "$STAGEB_MANIFEST" "$MANIFEST_SUMMARY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing prepared file: $f" >&2; exit 1; }
done

mkdir -p "$STAGEA_DIR" "$STAGEB_DIR"

echo "[RUN] E25A StageA target-domain preadapt (partial-unfreeze backbone.18-20)"
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
  --train_manifest "$STAGEA_MANIFEST" \
  --test_manifest "$EVAL_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --freeze_backbone true \
  --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --trainable_families green8 \
  --province_balance_mode inv_sqrt \
  --strata_balance_mode none \
  --main_group_by family \
  --main_group_ratios green8=1.0 \
  --main_group_clip 1.0 \
  --enhanced_green_head expD \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --learning_rate 0.0001 \
  --lr_schedule 2 3 \
  --selection_proxy_eval_samples 1000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --selection_strategy balanced_recovery \
  --first_char_aux_weight 0.15 \
  --second_char_aux_weight 0.0 \
  --ne_type_aux_weight 0.0 \
  --rear_seq_aux_weight 0.30 \
  --rear_seq_drop_chars 1 \
  --rear_seq_start_step 4 \
  --save_folder "$STAGEA_DIR/" \
  --max_epoch 3 \
  --train_brightness_aug_max 120.0 \
  --gray3_prob 0.0 \
  2>&1 | tee "$STAGEA_DIR/train.log"

[[ -f "$STAGEA_DIR/Final_LPRNet_model.pth" ]] || { echo "[ERR] stageA final weight missing" >&2; exit 1; }

echo "[RUN] E25A StageB full reintegration"
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
  --train_manifest "$STAGEB_MANIFEST" \
  --test_manifest "$EVAL_MANIFEST" \
  --pretrained_model "$STAGEA_DIR/Final_LPRNet_model.pth" \
  --head_mode multihead \
  --freeze_backbone true \
  --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
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
  --save_folder "$STAGEB_DIR/" \
  --max_epoch 3 \
  --train_brightness_aug_max 120.0 \
  --gray3_prob 0.0 \
  2>&1 | tee "$STAGEB_DIR/train.log"

[[ -f "$STAGEB_DIR/Final_LPRNet_model.pth" ]] || { echo "[ERR] stageB final weight missing" >&2; exit 1; }
cp "$STAGEB_DIR/Final_LPRNet_model.pth" "$SAVE_DIR/Final_LPRNet_model.pth"

echo "[EVAL] family-aware base"
$PY src/evaluation/eval_family_aware_blue_green_by_province.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_family_aware.json" \
  --batch_size 300 \
  --num_workers 4 \
  2>&1 | tee "$SAVE_DIR/eval_family_aware.log"

echo "[EVAL] green8 metrics base"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$SAVE_DIR/Final_LPRNet_model.pth" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

echo "[EVAL] E7 board-native base"
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
  --log "$STAGEB_DIR/train.log" \
  --process-keyword green_e25a_cluster2_repr_rebuild_twostage \
  --final-weight "$SAVE_DIR/Final_LPRNet_model.pth" \
  --eval-json "$SAVE_DIR/eval_family_aware.json" \
  --out-json "$STATUS_JSON" \
  --out-md "$STATUS_MD"

echo "[DONE] E25A complete"
