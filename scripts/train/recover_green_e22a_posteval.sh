#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:/home/wzzz/LPRNet/src/training:/home/wzzz/LPRNet/src/evaluation:${PYTHONPATH:-}

SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e22a_cluster2_province_head_e12c_stage2
MODEL=$SAVE_DIR/Final_LPRNet_model.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
E7_MANIFEST=/home/wzzz/LPRNet/tmp/green_board_native_e7_v2/manifests/eval_manifest.csv
CLUSTER2_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv
CLUSTER3_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv
CLUSTER3_TAIL_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_tail_collapse_wsl.csv
STATUS_JSON=/home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.json
STATUS_MD=/home/wzzz/LPRNet/reports/ACTIVE_TRAINING_STATUS.md

for f in "$MODEL" "$EVAL_MANIFEST" "$E7_MANIFEST" "$CLUSTER2_CSV" "$CLUSTER3_CSV" "$CLUSTER3_TAIL_CSV"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

rm -f \
  "$SAVE_DIR/eval_family_aware_province_fused.json" \
  "$SAVE_DIR/eval_green8_metrics_only.json" \
  "$SAVE_DIR/eval_green8_metrics_only_province_fused.json" \
  "$SAVE_DIR/eval_e7_board_native.json" \
  "$SAVE_DIR/eval_e7_board_native_province_fused.json" \
  "$SAVE_DIR/eval_cluster2_dump.json" \
  "$SAVE_DIR/eval_cluster2_dump.csv" \
  "$SAVE_DIR/eval_cluster2_dump_province_fused.json" \
  "$SAVE_DIR/eval_cluster2_dump_province_fused.csv" \
  "$SAVE_DIR/eval_cluster3_dump.json" \
  "$SAVE_DIR/eval_cluster3_dump.csv" \
  "$SAVE_DIR/eval_cluster3_dump_province_fused.json" \
  "$SAVE_DIR/eval_cluster3_dump_province_fused.csv" \
  "$SAVE_DIR/eval_cluster3_tail_dump.json" \
  "$SAVE_DIR/eval_cluster3_tail_dump.csv" \
  "$SAVE_DIR/eval_cluster3_tail_dump_province_fused.json" \
  "$SAVE_DIR/eval_cluster3_tail_dump_province_fused.csv"

echo "[RECOVER] E22A post-train eval resume"

echo "[EVAL] family-aware province fused"
$PY src/evaluation/eval_family_aware_blue_green_by_province.py \
  --model "$MODEL" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_family_aware_province_fused.json" \
  --batch_size 300 \
  --num_workers 4 \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_family_aware_province_fused.log"

echo "[EVAL] green8 metrics base"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$MODEL" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only.log"

echo "[EVAL] green8 metrics province fused"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$MODEL" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_only_province_fused.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_only_province_fused.log"

echo "[EVAL] E7 board-native base"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$MODEL" \
  --manifest "$E7_MANIFEST" \
  --out_json "$SAVE_DIR/eval_e7_board_native.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  2>&1 | tee "$SAVE_DIR/eval_e7_board_native.log"

echo "[EVAL] E7 board-native province fused"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$MODEL" \
  --manifest "$E7_MANIFEST" \
  --out_json "$SAVE_DIR/eval_e7_board_native_province_fused.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_e7_board_native_province_fused.log"

echo "[EVAL] cluster2 dump replay base"
$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv "$CLUSTER2_CSV" \
  --out-json "$SAVE_DIR/eval_cluster2_dump.json" \
  --out-csv "$SAVE_DIR/eval_cluster2_dump.csv" \
  2>&1 | tee "$SAVE_DIR/eval_cluster2_dump.log"

echo "[EVAL] cluster2 dump replay province fused"
$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv "$CLUSTER2_CSV" \
  --out-json "$SAVE_DIR/eval_cluster2_dump_province_fused.json" \
  --out-csv "$SAVE_DIR/eval_cluster2_dump_province_fused.csv" \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_cluster2_dump_province_fused.log"

echo "[EVAL] cluster3 dump replay base"
$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv "$CLUSTER3_CSV" \
  --out-json "$SAVE_DIR/eval_cluster3_dump.json" \
  --out-csv "$SAVE_DIR/eval_cluster3_dump.csv" \
  2>&1 | tee "$SAVE_DIR/eval_cluster3_dump.log"

echo "[EVAL] cluster3 dump replay province fused"
$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv "$CLUSTER3_CSV" \
  --out-json "$SAVE_DIR/eval_cluster3_dump_province_fused.json" \
  --out-csv "$SAVE_DIR/eval_cluster3_dump_province_fused.csv" \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_cluster3_dump_province_fused.log"

echo "[EVAL] cluster3 tail dump replay base"
$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv "$CLUSTER3_TAIL_CSV" \
  --out-json "$SAVE_DIR/eval_cluster3_tail_dump.json" \
  --out-csv "$SAVE_DIR/eval_cluster3_tail_dump.csv" \
  2>&1 | tee "$SAVE_DIR/eval_cluster3_tail_dump.log"

echo "[EVAL] cluster3 tail dump replay province fused"
$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv "$CLUSTER3_TAIL_CSV" \
  --out-json "$SAVE_DIR/eval_cluster3_tail_dump_province_fused.json" \
  --out-csv "$SAVE_DIR/eval_cluster3_tail_dump_province_fused.csv" \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_cluster3_tail_dump_province_fused.log"

echo "[STATUS] refresh shared status"
$PY src/utils/update_training_status.py \
  --log "$SAVE_DIR/train.log" \
  --process-keyword green_e22a_cluster2_province_head_e12c_stage2 \
  --final-weight "$MODEL" \
  --eval-json "$SAVE_DIR/eval_family_aware_province_fused.json" \
  --out-json "$STATUS_JSON" \
  --out-md "$STATUS_MD"

echo "[DONE] E22A recovery complete"
