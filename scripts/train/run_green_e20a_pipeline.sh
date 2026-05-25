#!/usr/bin/env bash
set -euo pipefail

bash /home/wzzz/LPRNet/scripts/train/run_green_e20a_prepare.sh
bash /home/wzzz/LPRNet/scripts/train/run_green_e20a_cluster2_beijing_prefix_contrast_1200_e12c_stage2.sh

echo "[DONE] E20A pipeline complete"
