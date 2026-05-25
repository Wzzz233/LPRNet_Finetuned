#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python

BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.csv
BANK_JSON=/home/wzzz/LPRNet/tmp/green_e20a_cluster2_beijing_prefix_contrast_bank.json
OUT_DIR=/home/wzzz/LPRNet/generated/green_e20a_cluster2_beijing_prefix_contrast_1200
OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e20a_cluster2_beijing_prefix_contrast_1200.csv

for f in "$BASE_MANIFEST" "$BANK_JSON"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

if [[ -d "$OUT_DIR" ]]; then
  mv "$OUT_DIR" "${OUT_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
if [[ -f "$OUT_MANIFEST" ]]; then
  mv "$OUT_MANIFEST" "${OUT_MANIFEST}.bak.$(date +%Y%m%d_%H%M%S)"
fi

$PY src/utils/generate_green_e20a_cluster2_prefix_contrast.py \
  --base-manifest "$BASE_MANIFEST" \
  --bank-json "$BANK_JSON" \
  --out-dir "$OUT_DIR" \
  --out-manifest "$OUT_MANIFEST" \
  --dataset-name green_e20a_cluster2_beijing_prefix_contrast_1200 \
  --source-name e20a_cluster2_beijing_prefix_contrast_1200 \
  --bucket-ratios board_mid_occ=0.5,board_low_occ=0.35,board_extreme_tail=0.15

echo "[DONE] E20A prepare complete"
