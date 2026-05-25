#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:/home/wzzz/LPRNet/src/training:/home/wzzz/LPRNet/src/evaluation:${PYTHONPATH:-}

BASE_MODEL=/home/wzzz/LPRNet/experiments/green_e12_e9c_append_boarddump_anticollapse_5prov_1200_stage2/Final_LPRNet_model.pth
PROVINCE_MODEL=/home/wzzz/LPRNet/experiments/green_e28_cluster2_province_specialist_stage2/Final_LPRNet_model.pth
EVAL_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_aggr_v1_existing_only.csv
E7_MANIFEST=/home/wzzz/LPRNet/tmp/green_board_native_e7_v2/manifests/eval_manifest.csv
CLUSTER2_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv
SAVE_DIR=/home/wzzz/LPRNet/experiments/green_e29_cluster2_specialist_overlay_eval

for f in "$BASE_MODEL" "$PROVINCE_MODEL" "$EVAL_MANIFEST" "$E7_MANIFEST" "$CLUSTER2_CSV"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

mkdir -p /home/wzzz/LPRNet/experiments
if [[ -d "$SAVE_DIR" ]]; then
  mv "$SAVE_DIR" "${SAVE_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$SAVE_DIR"

echo "[EVAL] cluster2 specialist overlay on green8 holdout"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$BASE_MODEL" \
  --province-model "$PROVINCE_MODEL" \
  --manifest "$EVAL_MANIFEST" \
  --out_json "$SAVE_DIR/eval_green8_metrics_specialist_overlay.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_green8_metrics_specialist_overlay.log"

echo "[EVAL] cluster2 specialist overlay on E7 board-native"
$PY src/evaluation/eval_green8_metrics_only.py \
  --model "$BASE_MODEL" \
  --province-model "$PROVINCE_MODEL" \
  --manifest "$E7_MANIFEST" \
  --out_json "$SAVE_DIR/eval_e7_board_native_specialist_overlay.json" \
  --batch_size 300 \
  --num_workers 4 \
  --ocr_preproc none \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_e7_board_native_specialist_overlay.log"

echo "[EVAL] cluster2 specialist overlay on cluster2 dump"
$PY src/utils/replay_dump_compare.py \
  --model "$BASE_MODEL" \
  --province-model "$PROVINCE_MODEL" \
  --input-csv "$CLUSTER2_CSV" \
  --out-json "$SAVE_DIR/eval_cluster2_dump_specialist_overlay.json" \
  --out-csv "$SAVE_DIR/eval_cluster2_dump_specialist_overlay.csv" \
  --province-fusion-mode replace_if_confident \
  --province-conf-threshold 0.55 \
  2>&1 | tee "$SAVE_DIR/eval_cluster2_dump_specialist_overlay.log"

echo "[DONE] E29 overlay eval complete"
