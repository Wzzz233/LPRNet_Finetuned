#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ROOT_DIR}/.conda/bin/python"
RUN_TAG="${1:-board_baseline_v1}"
OUT_DIR="${ROOT_DIR}/experiments/${RUN_TAG}"
WEIGHTS_DIR="${OUT_DIR}/weights"
mkdir -p "${OUT_DIR}" "${WEIGHTS_DIR}"

TRAIN_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/train_labels.txt"
VAL_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/val_labels.txt"
TEST_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/test_labels.txt"

echo "[1/4] Analyze split statistics"
"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_TXT}" \
  --val "${VAL_TXT}" \
  --test "${TEST_TXT}" \
  --out-json "${OUT_DIR}/split_stats.json" | tee "${OUT_DIR}/split_stats.stdout.json"

echo "[2/4] Start board-aligned training"
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  --train_img_dirs "${ROOT_DIR}/CCPD2019" \
  --test_img_dirs "${ROOT_DIR}/CCPD2019" \
  --train_txt_file "${TRAIN_TXT}" \
  --test_txt_file "${VAL_TXT}" \
  --data_mode ccpd_board \
  --ocr_channel_order bgr \
  --ocr_crop_mode match \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --pretrained_model "${ROOT_DIR}/weights_red_stage3/Final_LPRNet_model.pth" \
  --save_folder "${WEIGHTS_DIR}/" \
  --save_interval 2000 \
  --max_epoch 15 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --num_workers 4 \
  --cuda true | tee "${OUT_DIR}/train.log"

FINAL_MODEL="${WEIGHTS_DIR}/Final_LPRNet_model.pth"
if [[ ! -f "${FINAL_MODEL}" ]]; then
  echo "Final model not found: ${FINAL_MODEL}" >&2
  exit 1
fi

echo "[3/4] Evaluate on val"
"${ENV_PY}" "${ROOT_DIR}/eval_lpr_detailed.py" \
  --test_img_dirs "${ROOT_DIR}/CCPD2019" \
  --txt_file "${VAL_TXT}" \
  --data_mode ccpd_board \
  --ocr_channel_order bgr \
  --ocr_crop_mode match \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --pretrained_model "${FINAL_MODEL}" \
  --test_batch_size 120 \
  --num_workers 4 \
  --cuda true \
  --out_json "${OUT_DIR}/val_metrics.json" | tee "${OUT_DIR}/val_metrics.stdout.json"

echo "[4/4] Evaluate on test"
"${ENV_PY}" "${ROOT_DIR}/eval_lpr_detailed.py" \
  --test_img_dirs "${ROOT_DIR}/CCPD2019" \
  --txt_file "${TEST_TXT}" \
  --data_mode ccpd_board \
  --ocr_channel_order bgr \
  --ocr_crop_mode match \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --pretrained_model "${FINAL_MODEL}" \
  --test_batch_size 120 \
  --num_workers 4 \
  --cuda true \
  --out_json "${OUT_DIR}/test_metrics.json" | tee "${OUT_DIR}/test_metrics.stdout.json"

echo "Done. Output dir: ${OUT_DIR}"
