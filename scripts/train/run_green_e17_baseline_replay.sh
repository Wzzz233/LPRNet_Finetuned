#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
MODEL=/home/wzzz/LPRNet/experiments/green_e12_e9c_append_boarddump_anticollapse_5prov_1200_stage2/Final_LPRNet_model.pth
OUT_DIR=/home/wzzz/LPRNet/experiments/green_e12_e9c_append_boarddump_anticollapse_5prov_1200_stage2

[[ -f "$MODEL" ]] || { echo "[ERR] missing model: $MODEL" >&2; exit 1; }

$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv /home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv \
  --out-json "$OUT_DIR/eval_cluster2_dump.json" \
  --out-csv "$OUT_DIR/eval_cluster2_dump.csv"

$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv /home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv \
  --out-json "$OUT_DIR/eval_cluster3_dump.json" \
  --out-csv "$OUT_DIR/eval_cluster3_dump.csv"

$PY src/utils/replay_dump_compare.py \
  --model "$MODEL" \
  --input-csv /home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_tail_collapse_wsl.csv \
  --out-json "$OUT_DIR/eval_cluster3_tail_dump.json" \
  --out-csv "$OUT_DIR/eval_cluster3_tail_dump.csv"

echo "[DONE] baseline replay finished"
