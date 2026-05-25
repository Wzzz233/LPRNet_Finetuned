#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training:${PYTHONPATH:-}
python src/training/train_tiny_province_net.py --train_manifest /home/wzzz/LPRNet/manifests/firstchar_tiny_gray_green8only_v1/train.csv --test_manifest /home/wzzz/LPRNet/manifests/firstchar_tiny_gray_green8only_v1/test.csv --save_dir /home/wzzz/LPRNet/experiments/firstchar_tiny_A3_gray_green8only_ep30 --ocr_preproc gray --epochs 30 --batch_size 256 --num_workers 8 --lr 1e-3
