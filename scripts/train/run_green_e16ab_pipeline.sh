#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

BASE_E9C=/home/wzzz/LPRNet/manifests/unified_manifest_green_e9c_exact_template_allprov_1800.csv
E16A_OUT_DIR=/home/wzzz/LPRNet/generated/green_e16a_nonanhui_ad_balance_12k_std_v1
E16A_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e16a_nonanhui_ad_balance_12k_std_v1.csv
E16B_OUT_DIR=/home/wzzz/LPRNet/generated/green_e16b_nonanhui_ad_balance_12k_dump_v1
E16B_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e16b_nonanhui_ad_balance_12k_dump_v1.csv
CLUSTER2_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv
CLUSTER3_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv
CLUSTER3_TAIL_CSV=/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_tail_collapse_wsl.csv

for f in "$BASE_E9C" "$CLUSTER2_CSV" "$CLUSTER3_CSV" "$CLUSTER3_TAIL_CSV"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

backup_if_exists() {
  local path="$1"
  if [[ -e "$path" ]]; then
    mv "$path" "${path}.bak.$(date +%Y%m%d_%H%M%S)"
  fi
}

echo "[GEN] E16A full data"
backup_if_exists "$E16A_OUT_DIR"
backup_if_exists "$E16A_MANIFEST"
$PY src/utils/generate_green_e16a_nonanhui_ad_balance.py \
  --base-manifest "$BASE_E9C" \
  --out-dir "$E16A_OUT_DIR" \
  --out-manifest "$E16A_MANIFEST"

echo "[GEN] E16B targeted dump-text data"
backup_if_exists "$E16B_OUT_DIR"
backup_if_exists "$E16B_MANIFEST"
$PY src/utils/generate_green_e16b_cluster2_dump_from_e16a.py \
  --base-manifest "$E16A_MANIFEST" \
  --out-dir "$E16B_OUT_DIR" \
  --out-manifest "$E16B_MANIFEST" \
  --dump-csv "$CLUSTER2_CSV" \
  --dump-csv "$CLUSTER3_CSV" \
  --dump-csv "$CLUSTER3_TAIL_CSV"

echo "[TRAIN] E16A"
bash scripts/train/run_green_e16a_nonanhui_ad_balance_12k_std_v1_stage2.sh

echo "[TRAIN] E16B"
bash scripts/train/run_green_e16b_nonanhui_ad_balance_12k_dump_v1_stage2.sh

echo "[DONE] E16A/E16B pipeline complete"
