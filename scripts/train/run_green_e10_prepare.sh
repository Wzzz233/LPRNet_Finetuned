#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

BASE_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e8c_brightness_replace_5prov.csv

E10A_OUT_DIR=/home/wzzz/LPRNet/tmp/green_e10_boarddump_exact_template_v1/e10a_5prov_1800
E10A_GEN_MANIFEST=$E10A_OUT_DIR/manifests/train_manifest_boarddump_exact_template.csv
E10A_OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e10a_boarddump_exact_template_5prov_1800_replace.csv
E10A_SUMMARY=/home/wzzz/LPRNet/manifests/unified_manifest_green_e10a_boarddump_exact_template_5prov_1800_replace.summary.json

E10B_OUT_DIR=/home/wzzz/LPRNet/tmp/green_e10_boarddump_exact_template_v1/e10b_5prov_1800
E10B_GEN_MANIFEST=$E10B_OUT_DIR/manifests/train_manifest_boarddump_exact_template.csv
E10B_OUT_MANIFEST=/home/wzzz/LPRNet/manifests/unified_manifest_green_e10b_boarddump_overflowfocus_5prov_1800_replace.csv
E10B_SUMMARY=/home/wzzz/LPRNet/manifests/unified_manifest_green_e10b_boarddump_overflowfocus_5prov_1800_replace.summary.json

rm -rf "$E10A_OUT_DIR" "$E10B_OUT_DIR"

echo "[E10] generate E10A board_dump exact-template data"
$PY src/utils/generate_green_boarddump_exact_templates.py \
  --repo_root /home/wzzz/LPRNet \
  --out_dir "$E10A_OUT_DIR" \
  --province_mode five \
  --template_mode broad_overflow \
  --total_count 1800 \
  --seed 20260416 \
  --param_tries_per_seed 100 \
  --extra_geom_retry 6 \
  --preview_limit 60 \
  --dataset_name green_e10a_boarddump_exact_template_5prov_1800 \
  --source_name e10a_boarddump_exact_template_5prov_1800 \
  --bucket_weights geometry_clean=1.0,board_mid_occ=0.0,board_low_occ=0.0,board_extreme_tail=0.0 \
  --avoid_text_files "$BASE_MANIFEST"

echo "[E10] build E10A replace manifest"
$PY src/manifest/build_green_replace_manifest.py \
  --base-manifest "$BASE_MANIFEST" \
  --generated-manifest "$E10A_GEN_MANIFEST" \
  --out-manifest "$E10A_OUT_MANIFEST" \
  --out-summary "$E10A_SUMMARY"

echo "[E10] generate E10B focused board_dump exact-template data"
$PY src/utils/generate_green_boarddump_exact_templates.py \
  --repo_root /home/wzzz/LPRNet \
  --out_dir "$E10B_OUT_DIR" \
  --province_mode five \
  --template_mode cluster1_focus \
  --total_count 1800 \
  --seed 20260417 \
  --param_tries_per_seed 100 \
  --extra_geom_retry 6 \
  --preview_limit 60 \
  --dataset_name green_e10b_boarddump_overflowfocus_5prov_1800 \
  --source_name e10b_boarddump_overflowfocus_5prov_1800 \
  --bucket_weights geometry_clean=1.0,board_mid_occ=0.0,board_low_occ=0.0,board_extreme_tail=0.0 \
  --avoid_text_files "$BASE_MANIFEST"

echo "[E10] build E10B replace manifest"
$PY src/manifest/build_green_replace_manifest.py \
  --base-manifest "$BASE_MANIFEST" \
  --generated-manifest "$E10B_GEN_MANIFEST" \
  --out-manifest "$E10B_OUT_MANIFEST" \
  --out-summary "$E10B_SUMMARY"

echo "[E10] prepare done"
