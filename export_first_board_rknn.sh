#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ENV_PY="${ROOT_DIR}/.conda/bin/python"
RKNN_ENV_PREFIX="${RKNN_ENV_PREFIX:-/root/miniconda3/envs/rknn_env}"
RUN_TAG="${1:-first_board_baseline_v1}"
OUT_DIR="${ROOT_DIR}/experiments/${RUN_TAG}"
WEIGHTS_DIR="${OUT_DIR}/weights"
WEIGHTS_PATH="${WEIGHTS_DIR}/Final_LPRNet_model.pth"
ONNX_PATH="${WEIGHTS_DIR}/LPRNet_stage3_rk3568_fp16.onnx"
RKNN_PATH="${WEIGHTS_DIR}/LPRNet_stage3_rk3568_fp16.rknn"
VAL_TXT="${ROOT_DIR}/prepared_labels/ccpd2019/val_labels.txt"

if [[ ! -x "${TRAIN_ENV_PY}" ]]; then
  echo "Training env python not found: ${TRAIN_ENV_PY}" >&2
  exit 1
fi
if [[ ! -f "${WEIGHTS_PATH}" ]]; then
  echo "Weights not found: ${WEIGHTS_PATH}" >&2
  exit 1
fi

SAMPLE_REL="$(head -n 1 "${VAL_TXT}" | awk '{print $1}')"
SAMPLE_IMG="${ROOT_DIR}/CCPD2019/${SAMPLE_REL}"

echo "[1/3] Export RKNN-compatible ONNX"
"${TRAIN_ENV_PY}" "${ROOT_DIR}/export_onnx_rknn_compatible.py" \
  --weights "${WEIGHTS_PATH}" \
  --output "${ONNX_PATH}"

echo "[2/3] Build RKNN (rk3568, fp16)"
env CONDA_NO_PLUGINS=true conda run -p "${RKNN_ENV_PREFIX}" python "${ROOT_DIR}/custom_rknn_convert.py" \
  "${ONNX_PATH}" \
  --target-platform rk3568 \
  --dtype fp \
  --output "${RKNN_PATH}"

echo "[3/3] Verify PyTorch / ONNX / RKNN consistency"
env CONDA_NO_PLUGINS=true conda run -p "${RKNN_ENV_PREFIX}" python "${ROOT_DIR}/verify_export_consistency.py" \
  --weights "${WEIGHTS_PATH}" \
  --onnx "${ONNX_PATH}" \
  --rknn "${RKNN_PATH}" \
  --image "${SAMPLE_IMG}" \
  --ocr_channel_order bgr \
  --ocr_crop_mode match \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90

echo "Done."
