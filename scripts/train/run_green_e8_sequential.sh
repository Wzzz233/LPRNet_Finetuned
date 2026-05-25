#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet
PY=./.conda/bin/python
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/utils:${PYTHONPATH:-}

# Sequential runner: E8A -> E8B -> E8C
# Each script already includes training + evaluation

echo "=== Starting E8A (brightness aug) ==="
bash scripts/train/run_green_e8a_brightness_aug.sh
echo "=== E8A DONE ==="

echo "=== Starting E8B (gray3 + brightness aug) ==="
bash scripts/train/run_green_e8b_gray3_brightness.sh
echo "=== E8B DONE ==="

echo "=== Starting E8C (brightness aug + board-native replace) ==="
bash scripts/train/run_green_e8c_brightness_replace.sh
echo "=== E8C DONE ==="

echo "=== ALL E8 EXPERIMENTS COMPLETE ==="