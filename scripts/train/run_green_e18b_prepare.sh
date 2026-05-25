#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python

BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.csv
OUT_DIR=/home/wzzz/LPRNet/generated/green_e18b_su_bf_transition_dense_600
OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e18b_su_bf_transition_dense_600.csv
CLUSTER3_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv
CLUSTER3_TAIL_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_tail_collapse_wsl.csv

for f in "$BASE_MANIFEST" "$CLUSTER3_CSV" "$CLUSTER3_TAIL_CSV"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

if [[ -d "$OUT_DIR" ]]; then
  mv "$OUT_DIR" "${OUT_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
if [[ -f "$OUT_MANIFEST" ]]; then
  mv "$OUT_MANIFEST" "${OUT_MANIFEST}.bak.$(date +%Y%m%d_%H%M%S)"
fi

$PY src/utils/generate_green_e18b_su_bf_transition_dense.py \
  --base-manifest "$BASE_MANIFEST" \
  --out-dir "$OUT_DIR" \
  --out-manifest "$OUT_MANIFEST" \
  --dump-csv "$CLUSTER3_CSV" \
  --dump-csv "$CLUSTER3_TAIL_CSV"

echo "[DONE] E18B prepare complete"
