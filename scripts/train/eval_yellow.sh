#!/bin/bash
# Evaluate yellow model on all test sets
# Usage: bash eval_yellow.sh <checkpoint_path>

CKPT="$1"
if [ -z "$CKPT" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

cd /home/wzzz/LPRNet/src/training || exit 1

BASE="python train_LPRNet.py --keys_file ../../keys/yellow_keys.txt \
  --data_mode manifest --head_mode single --lpr_max_len 8 --cuda True \
  --phase_train False --pretrained_model"

TRAIN_ARG="--train_manifest ../../manifests/yellow_train.csv"

echo "Checkpoint: ${CKPT}"
echo "=========================================="

# 1. Synthetic test (old baseline, includes single+double layer CBLPRD)
echo ""
echo "--- Synthetic test set (yellow_test.csv) ---"
${BASE} "${CKPT}" ${TRAIN_ARG} --test_manifest ../../manifests/yellow_test.csv \
    --save_folder /tmp/ev_yellow_syn --max_epoch 0 2>&1 | grep -E "(Test acc|Accuracy|Epoch:|Proxy)"

# 2. Real CRPD val (now has split=test after fix)
echo ""
echo "--- Real CRPD val set (yellow_real_val.csv) ---"
${BASE} "${CKPT}" ${TRAIN_ARG} --test_manifest ../../manifests/yellow_real_val.csv \
    --save_folder /tmp/ev_yellow_val --max_epoch 0 2>&1 | grep -E "(Test acc|Accuracy|Epoch:|Proxy)"

# 3. Real CRPD test
echo ""
echo "--- Real CRPD test set (yellow_real_test.csv) ---"
${BASE} "${CKPT}" ${TRAIN_ARG} --test_manifest ../../manifests/yellow_real_test.csv \
    --save_folder /tmp/ev_yellow_test --max_epoch 0 2>&1 | grep -E "(Test acc|Accuracy|Epoch:|Proxy)"
