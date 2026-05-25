#!/bin/bash
# Phase 2: unfreeze backbone, low lr fine-tune with freeze_bn_stats
cd /home/wzzz/LPRNet/src/training || exit 1

PHASE1_DIR="../../experiments/yellow_single_v1_phase1"
PHASE2_DIR="../../experiments/yellow_single_v1_phase2"

BEST_CKPT="${PHASE1_DIR}/best_LPRNet_model.pth"
if [ ! -f "$BEST_CKPT" ]; then
    BEST_CKPT="${PHASE1_DIR}/Final_LPRNet_model.pth"
fi
echo "Phase1 checkpoint: ${BEST_CKPT}"

python train_LPRNet.py \
  --keys_file ../../keys/yellow_keys.txt \
  --data_mode manifest \
  --train_manifest ../../manifests/yellow_train.csv \
  --test_manifest ../../manifests/yellow_real_val.csv \
  --pretrained_model "${BEST_CKPT}" \
  --backbone_lr_mult 0.1 \
  --freeze_bn_stats True \
  --learning_rate 0.0005 \
  --lr_schedule 5 10 15 \
  --max_epoch 20 \
  --save_folder "${PHASE2_DIR}/" \
  --head_mode single \
  --lpr_max_len 8 \
  --cuda True 2>&1
