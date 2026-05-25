#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_next_tilt_finetune_from_v6.sh [options] [extra options forwarded to run_tilt_obbwarp_experiment.sh]

Options:
  --run-tag <tag>
  --train-hardcase-dir <path>
  --val-hardcase-dir <path>
  --base-model <path>
  --source-train-txt <path>
  --source-val-txt <path>
  --selected-train <int>           (default: 12000)
  --selected-val <int>             (default: 2500)
  --stagea-hard-ratio <float>      (default: 0.70)
  --stageb-hard-ratio <float>      (default: 0.85)
  --stagec-hard-ratio <float>      (default: 0.92)
  --stagea-first-char-aux <float>  (default: 0.20)
  --stageb-first-char-aux <float>  (default: 0.18)
  --stagec-first-char-aux <float>  (default: 0.10)
  --checkpoint-reeval-mode <mode>  (default: val)
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

RUN_TAG="tilt_ocr_obbwarp_v7_from_v6_lenpos3_${DATE_TAG}"
TRAIN_HARDCASE_DIR="${ROOT_DIR}/experiments/hardcase_auto_v3_from_v6_train_${DATE_TAG}"
VAL_HARDCASE_DIR="${ROOT_DIR}/experiments/hardcase_auto_v3_from_v6_val_${DATE_TAG}"
BASE_MODEL="${ROOT_DIR}/experiments/tilt_ocr_obbwarp_v6_from_v5_balanced_20260319/weights_stageC/Final_LPRNet_model.pth"
SOURCE_TRAIN_TXT="${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/train_labels.txt"
SOURCE_VAL_TXT="${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/val_labels.txt"
SELECTED_TRAIN="12000"
SELECTED_VAL="2500"
STAGEA_HARD_RATIO="0.70"
STAGEB_HARD_RATIO="0.85"
STAGEC_HARD_RATIO="0.92"
STAGEA_FIRST_CHAR_AUX="0.20"
STAGEB_FIRST_CHAR_AUX="0.18"
STAGEC_FIRST_CHAR_AUX="0.10"
CHECKPOINT_REEVAL_MODE="val"
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
    --train-hardcase-dir)
      TRAIN_HARDCASE_DIR="$2"
      shift 2
      ;;
    --val-hardcase-dir)
      VAL_HARDCASE_DIR="$2"
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
    --source-val-txt)
      SOURCE_VAL_TXT="$2"
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
    --stagea-first-char-aux)
      STAGEA_FIRST_CHAR_AUX="$2"
      shift 2
      ;;
    --stageb-first-char-aux)
      STAGEB_FIRST_CHAR_AUX="$2"
      shift 2
      ;;
    --stagec-first-char-aux)
      STAGEC_FIRST_CHAR_AUX="$2"
      shift 2
      ;;
    --checkpoint-reeval-mode)
      CHECKPOINT_REEVAL_MODE="$2"
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
if [[ ! -f "${SOURCE_VAL_TXT}" ]]; then
  echo "Source val txt not found: ${SOURCE_VAL_TXT}" >&2
  exit 1
fi

COMMON_MINER_ARGS=(
  --model "${BASE_MODEL}"
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

echo "[Config] arm_perspective_ocr=ch:${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER} crop:${ARM_PERSPECTIVE_OCR_CROP_MODE} resize:${ARM_PERSPECTIVE_OCR_RESIZE_MODE}/${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL} preproc:${ARM_PERSPECTIVE_OCR_PREPROC} min_occ:${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO} quad_pad:${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"

echo "[1/3] Mine next-round train hardcases from v6 hard-train split"
"${ENV_PY}" "${ROOT_DIR}/mine_hardcase_from_model.py" \
  "${COMMON_MINER_ARGS[@]}" \
  --source-train-txt "${SOURCE_TRAIN_TXT}" \
  --out-dir "${TRAIN_HARDCASE_DIR}" \
  --hard-val "${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/val_labels.txt" \
  --hard-test "${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/test_labels.txt" \
  --selected-train "${SELECTED_TRAIN}" \
  --selected-val 0

echo "[2/3] Mine next-round val hardcases from v6 hard-val split"
"${ENV_PY}" "${ROOT_DIR}/mine_hardcase_from_model.py" \
  "${COMMON_MINER_ARGS[@]}" \
  --source-train-txt "${SOURCE_VAL_TXT}" \
  --out-dir "${VAL_HARDCASE_DIR}" \
  --hard-test "${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/test_labels.txt" \
  --selected-train 0 \
  --selected-val "${SELECTED_VAL}"

echo "[3/3] Launch next-round balanced fine-tune from v6"
"${ROOT_DIR}/run_tilt_obbwarp_experiment.sh" "${RUN_TAG}" \
  --base-model "${BASE_MODEL}" \
  --stagea-hard-ratio "${STAGEA_HARD_RATIO}" \
  --stageb-hard-ratio "${STAGEB_HARD_RATIO}" \
  --stagec-hard-ratio "${STAGEC_HARD_RATIO}" \
  --stagea-first-char-aux "${STAGEA_FIRST_CHAR_AUX}" \
  --stageb-first-char-aux "${STAGEB_FIRST_CHAR_AUX}" \
  --stagec-first-char-aux "${STAGEC_FIRST_CHAR_AUX}" \
  --hardcase-train-txt "${TRAIN_HARDCASE_DIR}/hardcase_train.txt" \
  --hardcase-val-txt "${VAL_HARDCASE_DIR}/hardcase_val.txt" \
  --checkpoint-reeval-mode "${CHECKPOINT_REEVAL_MODE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
