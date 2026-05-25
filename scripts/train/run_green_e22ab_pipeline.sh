#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet

echo "[PIPELINE] start E22A"
bash /home/wzzz/LPRNet/scripts/train/run_green_e22a_cluster2_province_head_e12c_stage2.sh

echo "[PIPELINE] start E22B"
bash /home/wzzz/LPRNet/scripts/train/run_green_e22b_cluster2_province_head_e20adata_stage2.sh

echo "[DONE] E22 pipeline complete"
