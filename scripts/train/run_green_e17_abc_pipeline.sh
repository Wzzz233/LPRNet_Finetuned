#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet

echo "[E17] baseline replay"
bash scripts/train/run_green_e17_baseline_replay.sh

echo "[E17] prepare/train E17A"
bash scripts/train/run_green_e17a_prepare.sh
bash scripts/train/run_green_e17a_stage2.sh

echo "[E17] prepare/train E17B"
bash scripts/train/run_green_e17b_prepare.sh
bash scripts/train/run_green_e17b_stage2.sh

echo "[E17] prepare/train E17C"
bash scripts/train/run_green_e17c_prepare.sh
bash scripts/train/run_green_e17c_stage2.sh

echo "[DONE] E17 ABC pipeline complete"
