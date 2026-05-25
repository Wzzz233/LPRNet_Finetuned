#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python

BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e26_cluster3_tail_boost_900.csv
E26_LOCAL_MANIFEST=/home/wzzz/LPRNet/generated/green_e26_cluster3_tail_boost_900/manifests/train_manifest_green_e26_cluster3_tail_boost_900.csv
OUT_DIR=/home/wzzz/LPRNet/generated/green_e27_cluster3_hardtail_450
OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e27_cluster3_hardtail_450.csv

for f in "$BASE_MANIFEST" "$E26_LOCAL_MANIFEST"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

if [[ -d "$OUT_DIR" ]]; then
  mv "$OUT_DIR" "${OUT_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
if [[ -f "$OUT_MANIFEST" ]]; then
  mv "$OUT_MANIFEST" "${OUT_MANIFEST}.bak.$(date +%Y%m%d_%H%M%S)"
fi

$PY src/utils/build_green_cluster_target_manifest.py \
  --base-manifest "$BASE_MANIFEST" \
  --source-manifest "$E26_LOCAL_MANIFEST" \
  --out-dir "$OUT_DIR" \
  --out-manifest "$OUT_MANIFEST" \
  --dataset-name green_e27_cluster3_hardtail_450 \
  --source-name e27_cluster3_hardtail_450 \
  --bucket-plan board_low_occ=180,board_extreme_tail=270

echo "[DONE] E27 prepare complete"
