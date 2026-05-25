#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python

BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e18b_su_bf_transition_dense_600.csv
E18B_LOCAL_MANIFEST=/home/wzzz/LPRNet/generated/green_e18b_su_bf_transition_dense_600/manifests/train_manifest_green_e18b_su_bf_transition_dense_600.csv
E19C_LOCAL_MANIFEST=/home/wzzz/LPRNet/generated/green_e19c_su_bf_low_tail_dense_240/manifests/train_manifest_green_e19c_su_bf_low_tail_dense_240.csv
OUT_DIR=/home/wzzz/LPRNet/generated/green_e26_cluster3_tail_boost_900
OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e26_cluster3_tail_boost_900.csv

for f in "$BASE_MANIFEST" "$E18B_LOCAL_MANIFEST" "$E19C_LOCAL_MANIFEST"; do
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
  --source-manifest "$E18B_LOCAL_MANIFEST" \
  --source-manifest "$E19C_LOCAL_MANIFEST" \
  --out-dir "$OUT_DIR" \
  --out-manifest "$OUT_MANIFEST" \
  --dataset-name green_e26_cluster3_tail_boost_900 \
  --source-name e26_cluster3_tail_boost_900 \
  --bucket-plan geometry_clean=90,board_mid_occ=180,board_low_occ=315,board_extreme_tail=315

echo "[DONE] E26 prepare complete"
