#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python

BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e20a_cluster2_beijing_prefix_contrast_1200.csv
REPR_MANIFEST=/home/wzzz/LPRNet/tmp/green_e25a_cluster2_repr_boarddump_6000/manifests/train_manifest_green_e25a_cluster2_repr_boarddump_6000.csv
CLUSTER2_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv
OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e28_cluster2_specialist.csv
OUT_SUMMARY=/home/wzzz/LPRNet/tmp/green_e28_cluster2_specialist_manifest_summary.json

for f in "$BASE_MANIFEST" "$REPR_MANIFEST" "$CLUSTER2_CSV"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

if [[ -f "$OUT_MANIFEST" ]]; then
  mv "$OUT_MANIFEST" "${OUT_MANIFEST}.bak.$(date +%Y%m%d_%H%M%S)"
fi
if [[ -f "$OUT_SUMMARY" ]]; then
  mv "$OUT_SUMMARY" "${OUT_SUMMARY}.bak.$(date +%Y%m%d_%H%M%S)"
fi

$PY src/utils/build_green_e28_cluster2_specialist_manifest.py \
  --base-manifest "$BASE_MANIFEST" \
  --repr-manifest "$REPR_MANIFEST" \
  --cluster2-csv "$CLUSTER2_CSV" \
  --out-manifest "$OUT_MANIFEST" \
  --out-summary "$OUT_SUMMARY"

echo "[DONE] E28 prepare complete"
