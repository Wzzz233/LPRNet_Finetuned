#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e9c_exact_template_allprov_1800.csv
E12_OUT_DIR=/home/wzzz/LPRNet/tmp/green_e12_boarddump_anti_collapse_v1/e12_5prov_1200
E12_GEN_MANIFEST=$E12_OUT_DIR/manifests/train_manifest_boarddump_exact_template.csv
E12_OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.csv
E12_SUMMARY=/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.report.json

mkdir -p /home/wzzz/LPRNet/tmp
if [[ -d "$E12_OUT_DIR" ]]; then
  mv "$E12_OUT_DIR" "${E12_OUT_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$E12_OUT_DIR"

echo "[E12] generate anti-collapse board_dump append data"
$PY src/utils/generate_green_boarddump_exact_templates.py \
  --repo_root /home/wzzz/LPRNet \
  --out_dir "$E12_OUT_DIR" \
  --province_mode five \
  --template_mode anti_collapse_balanced \
  --tail_mode anti_collapse \
  --total_count 1200 \
  --seed 20260416 \
  --param_tries_per_seed 100 \
  --extra_geom_retry 6 \
  --preview_limit 60 \
  --dataset_name green_e12_boarddump_anticollapse_5prov_1200 \
  --source_name e12_boarddump_anticollapse_5prov_1200 \
  --bucket_weights geometry_clean=1.0,board_mid_occ=0.0,board_low_occ=0.0,board_extreme_tail=0.0 \
  --avoid_text_files "$BASE_MANIFEST"

echo "[E12] append onto E9C manifest"
$PY src/manifest/build_green_e5_dumplike_append_manifest.py \
  --base-manifest "$BASE_MANIFEST" \
  --append-manifest "$E12_GEN_MANIFEST" \
  --out-manifest "$E12_OUT_MANIFEST" \
  --out-summary "$E12_SUMMARY"

echo "[E12] prepare done"
