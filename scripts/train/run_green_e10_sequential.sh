#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet

bash /home/wzzz/LPRNet/scripts/train/run_green_e10_prepare.sh
bash /home/wzzz/LPRNet/scripts/train/run_green_e10a_boarddump_exact_template_5prov_1800_replace.sh
bash /home/wzzz/LPRNet/scripts/train/run_green_e10b_boarddump_overflowfocus_5prov_1800_replace.sh

echo "[DONE] E10 sequential complete"
