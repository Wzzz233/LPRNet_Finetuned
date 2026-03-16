#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ROOT_DIR}/.conda/bin/python"
RUN_TAG="${1:-first_char_guard_v1}"
OUT_DIR="${ROOT_DIR}/experiments/${RUN_TAG}"
WEIGHTS_DIR="${OUT_DIR}/weights"
mkdir -p "${OUT_DIR}" "${WEIGHTS_DIR}"

TRAIN_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/train_labels.txt"
VAL_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/val_labels.txt"
TEST_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/test_labels.txt"
BOARD_TXT="${ROOT_DIR}/board_anchor_labels.txt"
PREV_MODEL="${ROOT_DIR}/experiments/crop_aligned_v1/weights/Final_LPRNet_model.pth"

echo "[1/5] Analyze split statistics"
"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_TXT}" \
  --val "${VAL_TXT}" \
  --test "${TEST_TXT}" \
  --out-json "${OUT_DIR}/split_stats.json" | tee "${OUT_DIR}/split_stats.stdout.json"

echo "[2/5] Train first-char guard fine-tune"
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
  --board_anchor_img_dirs "${ROOT_DIR}" \
  --board_anchor_txt_file "${BOARD_TXT}" \
  --board_anchor_sample_weight 768 \
  --province_balance_mode inv_sqrt \
  --first_char_aux_weight 0.40 \
  --first_char_time_steps 6 \
  --selection_proxy_eval_samples 5000 \
  --pretrained_model "${PREV_MODEL}" \
  --learning_rate 0.0001 \
  --lr_schedule 3 6 8 \
  --save_folder "${WEIGHTS_DIR}/" \
  --save_interval 2000 \
  --max_epoch 8 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --num_workers 4 \
  --cuda true | tee "${OUT_DIR}/train.log"

FINAL_MODEL="${WEIGHTS_DIR}/Final_LPRNet_model.pth"
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

echo "[5/6] Evaluate board anchors"
"${ENV_PY}" "${ROOT_DIR}/eval_board_anchors.py" \
  --weights "${FINAL_MODEL}" \
  --img_dirs "${ROOT_DIR}" \
  --txt_file "${BOARD_TXT}" \
  --first_char_time_steps 6 \
  --out_json "${OUT_DIR}/board_anchor_metrics.json" | tee "${OUT_DIR}/board_anchor_metrics.stdout.json"

echo "[6/6] Write experiment report"
"${ENV_PY}" "${ROOT_DIR}/generate_experiment_report.py" \
  --experiment_name "First Char Guard V1" \
  --run_dir "${OUT_DIR}" \
  --train_txt "${TRAIN_TXT}" \
  --val_txt "${VAL_TXT}" \
  --test_txt "${TEST_TXT}" \
  --board_anchor_txt "${BOARD_TXT}" \
  --data_mode ccpd_board \
  --ocr_channel_order bgr \
  --ocr_crop_mode match \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --pretrained_model "${PREV_MODEL}" \
  --learning_rate 0.0001 \
  --lr_schedule "3,6,8" \
  --max_epoch 8 \
  --train_batch_size 64 \
  --test_batch_size 120 \
  --train_plate_box_aug_mode jitter_refine \
  --train_plate_box_aug_prob 0.85 \
  --train_plate_box_aug_x 0.06 \
  --train_plate_box_aug_y 0.12 \
  --train_plate_box_aug_min_iou 0.75 \
  --province_balance_mode inv_sqrt \
  --board_anchor_sample_weight 768 \
  --first_char_aux_weight 0.40 \
  --first_char_time_steps 6 \
  --selection_proxy_eval_samples 5000 \
  --report_path "${OUT_DIR}/EXPERIMENT_REPORT.md"

echo "Done. Output dir: ${OUT_DIR}"
