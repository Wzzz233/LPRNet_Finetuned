#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet
bash /home/wzzz/LPRNet/scripts/train/run_green_e13a_prepare.sh
bash /home/wzzz/LPRNet/scripts/train/run_green_e13a_e9c_append_slotalign_aa0_5prov_300_stage2.sh
