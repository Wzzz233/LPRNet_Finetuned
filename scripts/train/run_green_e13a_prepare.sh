#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e9c_exact_template_allprov_1800.csv
E13_OUT_DIR=/home/wzzz/LPRNet/tmp/green_e13_slotalign_aa0_v1/e13a_5prov_300
E13_GEN_MANIFEST=$E13_OUT_DIR/manifests/train_manifest_boarddump_exact_template.csv
E13_OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e13a_e9c_append_slotalign_aa0_5prov_300.csv
E13_SUMMARY=/home/wzzz/LPRNet/manifests/unified_manifest_green_e13a_e9c_append_slotalign_aa0_5prov_300.report.json

mkdir -p /home/wzzz/LPRNet/tmp
if [[ -d "$E13_OUT_DIR" ]]; then
  mv "$E13_OUT_DIR" "${E13_OUT_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$E13_OUT_DIR"

for f in "$BASE_MANIFEST"; do
  [[ -f "$f" ]] || { echo "[ERR] missing required file: $f" >&2; exit 1; }
done

echo "[E13A] generate slot-alignment AA0 probe board_dump append data"
$PY src/utils/generate_green_boarddump_exact_templates.py \
  --repo_root /home/wzzz/LPRNet \
  --out_dir "$E13_OUT_DIR" \
  --province_mode five \
  --template_mode slotalign_aa0_probe \
  --tail_mode slotalign_aa0 \
  --total_count 300 \
  --seed 20260416 \
  --param_tries_per_seed 100 \
  --extra_geom_retry 6 \
  --preview_limit 60 \
  --dataset_name green_e13a_slotalign_aa0_5prov_300 \
  --source_name e13a_slotalign_aa0_5prov_300 \
  --bucket_weights geometry_clean=1.0,board_mid_occ=0.0,board_low_occ=0.0,board_extreme_tail=0.0 \
  --avoid_text_files "$BASE_MANIFEST"

echo "[E13A] append onto E9C manifest"
$PY src/manifest/build_green_e5_dumplike_append_manifest.py \
  --base-manifest "$BASE_MANIFEST" \
  --append-manifest "$E13_GEN_MANIFEST" \
  --out-manifest "$E13_OUT_MANIFEST" \
  --out-summary "$E13_SUMMARY"

echo "[E13A] prepare done"