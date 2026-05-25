#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet

echo "[PREP] E19C data"
bash /home/wzzz/LPRNet/scripts/train/run_green_e19c_prepare.sh

echo "[TRAIN] E19A"
bash /home/wzzz/LPRNet/scripts/train/run_green_e19a_cluster2_green8_adapter_pos0_e12c_stage2.sh

echo "[TRAIN] E19C"
bash /home/wzzz/LPRNet/scripts/train/run_green_e19c_su_bf_low_tail_dense_240_e18b_stage2.sh

echo "[DONE] E19A/E19C pipeline complete"
