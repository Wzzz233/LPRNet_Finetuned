#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wzzz/LPRNet
ENV_PATH=/home/wzzz/LPRNet/.conda
RUN_ROOT=${1:-$ROOT/runs/quad_refiner/stage1_r18_gt}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-20}
NUM_WORKERS=${NUM_WORKERS:-4}
LIMIT_CCPD2019=${LIMIT_CCPD2019:-0}
LIMIT_CCPD2020=${LIMIT_CCPD2020:-0}
LIMIT_CRPD=${LIMIT_CRPD:-0}
SKIP_CRPD=${SKIP_CRPD:-0}
CCPD2019_TRAIN_SPLIT_FILE=${CCPD2019_TRAIN_SPLIT_FILE:-/home/wzzz/LPRNet/datasets/CCPD2019/splits/train.txt}
CCPD2019_VAL_SPLIT_FILE=${CCPD2019_VAL_SPLIT_FILE:-/home/wzzz/LPRNet/datasets/CCPD2019/splits/val.txt}
CCPD2019_TEST_SPLIT_FILE=${CCPD2019_TEST_SPLIT_FILE:-/home/wzzz/LPRNet/datasets/CCPD2019/splits/test.txt}
CCPD2019_LABEL_DIR=${CCPD2019_LABEL_DIR:-/home/wzzz/LPRNet/prepared_labels/ccpd2019}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

mkdir -p "$RUN_ROOT"
mkdir -p "$CCPD2019_LABEL_DIR"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
python "$ROOT/src/utils/prepare_ccpd_splits.py" \
  --dataset_root /home/wzzz/LPRNet/datasets/CCPD2019 \
  --split_dir /home/wzzz/LPRNet/datasets/CCPD2019/splits \
  --output_dir "$CCPD2019_LABEL_DIR" \
  --splits train val test
TRAIN_JSONL="$RUN_ROOT/train.jsonl"
VAL_JSONL="$RUN_ROOT/val.jsonl"
TRAIN_EXTRA_ARGS=()
VAL_EXTRA_ARGS=()
if [[ -n "$CCPD2019_TRAIN_SPLIT_FILE" ]]; then
  TRAIN_EXTRA_ARGS+=(--ccpd2019-split-file "$CCPD2019_TRAIN_SPLIT_FILE")
fi
if [[ -n "$CCPD2019_VAL_SPLIT_FILE" ]]; then
  VAL_EXTRA_ARGS+=(--ccpd2019-split-file "$CCPD2019_VAL_SPLIT_FILE")
fi
if [[ "$SKIP_CRPD" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--skip-crpd)
  VAL_EXTRA_ARGS+=(--skip-crpd)
fi

python "$ROOT/scripts/build_quad_refiner_dataset.py" \
  --output-jsonl "$TRAIN_JSONL" \
  --split train \
  --limit-ccpd2019 "$LIMIT_CCPD2019" \
  --limit-ccpd2020 "$LIMIT_CCPD2020" \
  --limit-crpd "$LIMIT_CRPD" \
  "${TRAIN_EXTRA_ARGS[@]}"

python "$ROOT/scripts/build_quad_refiner_dataset.py" \
  --output-jsonl "$VAL_JSONL" \
  --split val \
  --limit-ccpd2019 "$LIMIT_CCPD2019" \
  --limit-ccpd2020 "$LIMIT_CCPD2020" \
  --limit-crpd "$LIMIT_CRPD" \
  "${VAL_EXTRA_ARGS[@]}"

python "$ROOT/scripts/run_quad_refiner_train.py" \
  --train-jsonl "$TRAIN_JSONL" \
  --val-jsonl "$VAL_JSONL" \
  --output-dir "$RUN_ROOT/exp" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"
