#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.csv
BANK_DIR=/home/wzzz/LPRNet/tmp/e17_shared_banks
BANK_JSON=$BANK_DIR/suffix_bank_cluster3.json
OUT_DIR=/home/wzzz/LPRNet/generated/green_e17c_cluster3_tail_600
OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e17c_cluster3_tail_600.csv

mkdir -p /home/wzzz/LPRNet/generated /home/wzzz/LPRNet/tmp
[[ -f "$BASE_MANIFEST" ]] || { echo "[ERR] missing base manifest: $BASE_MANIFEST" >&2; exit 1; }

if [[ ! -f "$BANK_JSON" ]]; then
  $PY src/utils/build_green_e17_shared_banks.py \
    --base-manifest "$BASE_MANIFEST" \
    --out-dir "$BANK_DIR"
fi

if [[ -d "$OUT_DIR" ]]; then
  mv "$OUT_DIR" "${OUT_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
if [[ -f "$OUT_MANIFEST" ]]; then
  mv "$OUT_MANIFEST" "${OUT_MANIFEST}.bak.$(date +%Y%m%d_%H%M%S)"
fi

$PY src/utils/generate_green_e17a_cluster2_suffixbank.py \
  --base-manifest "$BASE_MANIFEST" \
  --bank-json "$BANK_JSON" \
  --bucket-plan board_mid_occ=5,board_low_occ=10,board_extreme_tail=5 \
  --max-attempts 300 \
  --out-dir "$OUT_DIR" \
  --out-manifest "$OUT_MANIFEST" \
  --dataset-name green_e17c_cluster3_tail_600 \
  --source-name e17c_cluster3_tail_600 \
  --seed 20260418

echo "[DONE] E17C prepare complete"
