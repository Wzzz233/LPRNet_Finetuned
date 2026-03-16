#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ROOT_DIR}/.conda/bin/python"
RUN_TAG="${1:-crop_aligned_v1}"
OUT_DIR="${ROOT_DIR}/experiments/${RUN_TAG}"
WEIGHTS_DIR="${OUT_DIR}/weights"
mkdir -p "${OUT_DIR}" "${WEIGHTS_DIR}"

TRAIN_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/train_labels.txt"
VAL_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/val_labels.txt"
TEST_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/test_labels.txt"
PREV_MODEL="${ROOT_DIR}/experiments/first_board_baseline_v1/weights/Final_LPRNet_model.pth"
FINAL_MODEL="${WEIGHTS_DIR}/Final_LPRNet_model.pth"
RAW_PPM="${ROOT_DIR}/ocrin_0021_f000042.ppm"

echo "[1/5] Analyze split statistics"
"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_TXT}" \
  --val "${VAL_TXT}" \
  --test "${TEST_TXT}" \
  --out-json "${OUT_DIR}/split_stats.json" | tee "${OUT_DIR}/split_stats.stdout.json"

echo "[2/5] Train crop-aligned fine-tune"
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
  --train_plate_box_aug_mode jitter_refine \
  --train_plate_box_aug_prob 0.85 \
  --train_plate_box_aug_x 0.06 \
  --train_plate_box_aug_y 0.12 \
  --train_plate_box_aug_min_iou 0.75 \
  --pretrained_model "${PREV_MODEL}" \
  --learning_rate 0.0003 \
  --save_folder "${WEIGHTS_DIR}/" \
  --save_interval 2000 \
  --max_epoch 8 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --num_workers 4 \
  --cuda true | tee "${OUT_DIR}/train.log"

if [[ ! -f "${FINAL_MODEL}" ]]; then
  echo "Final model not found: ${FINAL_MODEL}" >&2
  exit 1
fi

echo "[3/5] Evaluate on val"
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

echo "[4/5] Evaluate on test"
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

if [[ -f "${RAW_PPM}" ]]; then
  echo "[5/5] Infer on raw board OCR dump"
  {
    echo "[baseline]"
    "${ENV_PY}" "${ROOT_DIR}/infer_board_dump.py" \
      --weights "${PREV_MODEL}" \
      --image "${RAW_PPM}"
    echo
    echo "[crop-aligned]"
    "${ENV_PY}" "${ROOT_DIR}/infer_board_dump.py" \
      --weights "${FINAL_MODEL}" \
      --image "${RAW_PPM}"
  } | tee "${OUT_DIR}/raw_board_dump.txt"
else
  echo "[5/5] Skip raw board OCR dump inference: ${RAW_PPM} not found"
fi

echo "Done. Output dir: ${OUT_DIR}"
