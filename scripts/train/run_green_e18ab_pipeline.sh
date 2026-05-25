#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet

echo "[PIPELINE] E18 start: prepare E18B data -> train/eval E18A -> train/eval E18B"

echo "[STEP] prepare E18B data"
bash /home/wzzz/LPRNet/scripts/train/run_green_e18b_prepare.sh

echo "[STEP] train/eval E18A"
bash /home/wzzz/LPRNet/scripts/train/run_green_e18a_cluster2_firstchar_cb_rescue_e12c_stage2.sh

echo "[STEP] train/eval E18B"
bash /home/wzzz/LPRNet/scripts/train/run_green_e18b_su_bf_transition_dense_600_stage2.sh

echo "[DONE] E18 pipeline complete"
