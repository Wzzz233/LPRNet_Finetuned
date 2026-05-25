#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_next_tilt_finetune_from_v5.sh [options] [extra options forwarded to run_tilt_obbwarp_experiment.sh]

Options:
  --run-tag <tag>
  --hardcase-dir <path>
  --base-model <path>
  --source-train-txt <path>
  --eval-json <path>
  --selected-train <int>           (default: 10000)
  --selected-val <int>             (default: 2000)
  --stagea-hard-ratio <float>      (default: 0.65)
  --stageb-hard-ratio <float>      (default: 0.80)
  --stagec-hard-ratio <float>      (default: 0.90)
  --seed <int>                     (default: 20260319)
  --help
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ROOT_DIR}/.conda/bin/python"
DATE_TAG="$(date +%Y%m%d)"

ARM_PERSPECTIVE_OCR_CHANNEL_ORDER="bgr"
ARM_PERSPECTIVE_OCR_CROP_MODE="obb_warp"
ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO="0.0"
ARM_PERSPECTIVE_OCR_RESIZE_MODE="letterbox"
ARM_PERSPECTIVE_OCR_RESIZE_KERNEL="nn"
ARM_PERSPECTIVE_OCR_PREPROC="none"
ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO="0.90"

RUN_TAG="tilt_ocr_obbwarp_v6_from_v5_balanced_${DATE_TAG}"
HARDCASE_DIR="${ROOT_DIR}/experiments/hardcase_auto_v2_from_v5_${DATE_TAG}"
BASE_MODEL="${ROOT_DIR}/experiments/tilt_ocr_obbwarp_v5_from_v3_hardcase10k_20260319/weights_stageC/Final_LPRNet_model.pth"
SOURCE_TRAIN_TXT="${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/train_labels.txt"
EVAL_JSON=""
SELECTED_TRAIN="10000"
SELECTED_VAL="2000"
STAGEA_HARD_RATIO="0.65"
STAGEB_HARD_RATIO="0.80"
STAGEC_HARD_RATIO="0.90"
SEED="20260319"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      usage
      exit 0
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --hardcase-dir)
      HARDCASE_DIR="$2"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --source-train-txt)
      SOURCE_TRAIN_TXT="$2"
      shift 2
      ;;
    --eval-json)
      EVAL_JSON="$2"
      shift 2
      ;;
    --selected-train)
      SELECTED_TRAIN="$2"
      shift 2
      ;;
    --selected-val)
      SELECTED_VAL="$2"
      shift 2
      ;;
    --stagea-hard-ratio)
      STAGEA_HARD_RATIO="$2"
      shift 2
      ;;
    --stageb-hard-ratio)
      STAGEB_HARD_RATIO="$2"
      shift 2
      ;;
    --stagec-hard-ratio)
      STAGEC_HARD_RATIO="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "${ENV_PY}" ]]; then
  echo "Training python not found: ${ENV_PY}" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Base model not found: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_TRAIN_TXT}" ]]; then
  echo "Source train txt not found: ${SOURCE_TRAIN_TXT}" >&2
  exit 1
fi
if [[ -n "${EVAL_JSON}" && ! -f "${EVAL_JSON}" ]]; then
  echo "Eval json not found: ${EVAL_JSON}" >&2
  exit 1
fi

echo "[Config] arm_perspective_ocr=ch:${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER} crop:${ARM_PERSPECTIVE_OCR_CROP_MODE} resize:${ARM_PERSPECTIVE_OCR_RESIZE_MODE}/${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL} preproc:${ARM_PERSPECTIVE_OCR_PREPROC} min_occ:${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO} quad_pad:${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"

MINER_ARGS=(
  --model "${BASE_MODEL}"
  --source-train-txt "${SOURCE_TRAIN_TXT}"
  --out-dir "${HARDCASE_DIR}"
  --hard-val "${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/val_labels.txt"
  --hard-test "${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/test_labels.txt"
  --selected-train "${SELECTED_TRAIN}"
  --selected-val "${SELECTED_VAL}"
  --seed "${SEED}"
  --test-img-dirs "${ROOT_DIR}/CCPD2019"
  --data-mode ccpd_board
  --ocr_channel_order "${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER}"
  --ocr_crop_mode "${ARM_PERSPECTIVE_OCR_CROP_MODE}"
  --ocr_quad_pad_ratio "${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"
  --ocr_resize_mode "${ARM_PERSPECTIVE_OCR_RESIZE_MODE}"
  --ocr_resize_kernel "${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL}"
  --ocr_preproc "${ARM_PERSPECTIVE_OCR_PREPROC}"
  --ocr_min_occ_ratio "${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO}"
)
if [[ -n "${EVAL_JSON}" ]]; then
  MINER_ARGS+=(--eval-json "${EVAL_JSON}")
fi

echo "[1/2] Mine next-round hardcases from v5"
"${ENV_PY}" "${ROOT_DIR}/mine_hardcase_from_model.py" "${MINER_ARGS[@]}"

echo "[2/2] Launch next-round balanced fine-tune from v5"
"${ROOT_DIR}/run_tilt_obbwarp_experiment.sh" "${RUN_TAG}" \
  --base-model "${BASE_MODEL}" \
  --stagea-hard-ratio "${STAGEA_HARD_RATIO}" \
  --stageb-hard-ratio "${STAGEB_HARD_RATIO}" \
  --stagec-hard-ratio "${STAGEC_HARD_RATIO}" \
  --hardcase-train-txt "${HARDCASE_DIR}/hardcase_train.txt" \
  --hardcase-val-txt "${HARDCASE_DIR}/hardcase_val.txt" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
