#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_green_specialist_from_official.sh [options]

Options:
  --run-tag <tag>
  --base-model <path>                    (default: ./weights_red_stage3/Final_LPRNet_model.pth)
  --green-root <path>                    (default: ./CCPD2020/ccpd_green)
  --train-cap-per-province <int>         (default: 2200)
  --train-max-major-ratio <float>        (default: 0.90)
  --val-balanced-per-province <int>      (default: 120)
  --test-balanced-per-province <int>     (default: 260)
  --stagea-epochs <int>                  (default: 10)
  --stageb-epochs <int>                  (default: 8)
  --stagec-epochs <int>                  (default: 6)
  --stagea-lr <float>                    (default: 0.00003)
  --stageb-lr <float>                    (default: 0.00001)
  --stagec-lr <float>                    (default: 0.000005)
  --stagea-first-char-aux <float>        (default: 0.20)
  --stageb-first-char-aux <float>        (default: 0.16)
  --stagec-first-char-aux <float>        (default: 0.10)
  --green-full-gain-min-pp <float>       (default: 0.20)
  --green-balanced-gain-min-pp <float>   (default: 0.20)
  --seed <int>                           (default: 20260320)
  --export-rknn <bool>                   (default: false)
  --help
USAGE
}

str_to_bool() {
  local v="${1:-}"
  case "${v,,}" in
    1|true|t|yes|y|on) echo "true" ;;
    0|false|f|no|n|off) echo "false" ;;
    *)
      echo "invalid boolean: ${v}" >&2
      return 1
      ;;
  esac
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ROOT_DIR}/.conda/bin/python"
RKNN_ENV_PREFIX="${RKNN_ENV_PREFIX:-/root/miniconda3/envs/rknn_env}"
DATE_TAG="$(date +%Y%m%d)"

RUN_TAG="green_specialist_from_official_${DATE_TAG}"
BASE_MODEL="${ROOT_DIR}/weights_red_stage3/Final_LPRNet_model.pth"
GREEN_ROOT="${ROOT_DIR}/CCPD2020/ccpd_green"

TRAIN_CAP_PER_PROVINCE="2200"
TRAIN_MAX_MAJOR_RATIO="0.90"
VAL_BALANCED_PER_PROVINCE="120"
TEST_BALANCED_PER_PROVINCE="260"

STAGEA_EPOCHS="10"
STAGEB_EPOCHS="8"
STAGEC_EPOCHS="6"

STAGEA_LR="0.00003"
STAGEB_LR="0.00001"
STAGEC_LR="0.000005"
STAGEA_LR_SCHEDULE=(4 7)
STAGEB_LR_SCHEDULE=(3 6)
STAGEC_LR_SCHEDULE=(2 4)

STAGEA_FIRST_CHAR_AUX="0.20"
STAGEB_FIRST_CHAR_AUX="0.16"
STAGEC_FIRST_CHAR_AUX="0.10"

GREEN_FULL_GAIN_MIN_PP="0.20"
GREEN_BALANCED_GAIN_MIN_PP="0.20"

USE_CUDA="${USE_CUDA:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="20260320"
EXPORT_RKNN="false"

# Lock OCR preprocessing to ARM OBB path contract.
ARM_PERSPECTIVE_OCR_CHANNEL_ORDER="bgr"
ARM_PERSPECTIVE_OCR_CROP_MODE="obb_warp"
ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO="0.0"
ARM_PERSPECTIVE_OCR_RESIZE_MODE="letterbox"
ARM_PERSPECTIVE_OCR_RESIZE_KERNEL="nn"
ARM_PERSPECTIVE_OCR_PREPROC="none"
ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO="0.90"

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
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --green-root)
      GREEN_ROOT="$2"
      shift 2
      ;;
    --train-cap-per-province)
      TRAIN_CAP_PER_PROVINCE="$2"
      shift 2
      ;;
    --train-max-major-ratio)
      TRAIN_MAX_MAJOR_RATIO="$2"
      shift 2
      ;;
    --val-balanced-per-province)
      VAL_BALANCED_PER_PROVINCE="$2"
      shift 2
      ;;
    --test-balanced-per-province)
      TEST_BALANCED_PER_PROVINCE="$2"
      shift 2
      ;;
    --stagea-epochs)
      STAGEA_EPOCHS="$2"
      shift 2
      ;;
    --stageb-epochs)
      STAGEB_EPOCHS="$2"
      shift 2
      ;;
    --stagec-epochs)
      STAGEC_EPOCHS="$2"
      shift 2
      ;;
    --stagea-lr)
      STAGEA_LR="$2"
      shift 2
      ;;
    --stageb-lr)
      STAGEB_LR="$2"
      shift 2
      ;;
    --stagec-lr)
      STAGEC_LR="$2"
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
    --green-full-gain-min-pp)
      GREEN_FULL_GAIN_MIN_PP="$2"
      shift 2
      ;;
    --green-balanced-gain-min-pp)
      GREEN_BALANCED_GAIN_MIN_PP="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --export-rknn)
      EXPORT_RKNN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

EXPORT_RKNN="$(str_to_bool "${EXPORT_RKNN}")"

OUT_DIR="${ROOT_DIR}/experiments/${RUN_TAG}"
LABELS_DIR="${OUT_DIR}/labels"
GREEN_LABEL_SRC_DIR="${OUT_DIR}/green_source_labels"
WEIGHTS_A="${OUT_DIR}/weights_stageA"
WEIGHTS_B="${OUT_DIR}/weights_stageB"
WEIGHTS_C="${OUT_DIR}/weights_stageC"
mkdir -p "${OUT_DIR}" "${LABELS_DIR}" "${GREEN_LABEL_SRC_DIR}" "${WEIGHTS_A}" "${WEIGHTS_B}" "${WEIGHTS_C}"

if [[ ! -x "${ENV_PY}" ]]; then
  echo "Training python not found: ${ENV_PY}" >&2
  exit 1
fi
if [[ ! -d "${GREEN_ROOT}" ]]; then
  echo "Green dataset root not found: ${GREEN_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Base model not found: ${BASE_MODEL}" >&2
  exit 1
fi

echo "[Config] run_tag=${RUN_TAG} out_dir=${OUT_DIR}"
echo "[Config] base_model=${BASE_MODEL}"
echo "[Config] green_root=${GREEN_ROOT}"
echo "[Config] stage_epochs=A:${STAGEA_EPOCHS} B:${STAGEB_EPOCHS} C:${STAGEC_EPOCHS}"
echo "[Config] arm_perspective_ocr=ch:${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER} crop:${ARM_PERSPECTIVE_OCR_CROP_MODE} resize:${ARM_PERSPECTIVE_OCR_RESIZE_MODE}/${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL} preproc:${ARM_PERSPECTIVE_OCR_PREPROC} min_occ:${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO} quad_pad:${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"

step=1

echo "[${step}] Build CCPD green labels"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/prepare_ccpd_green_labels.py" \
  --dataset-root "${GREEN_ROOT}" \
  --output-dir "${GREEN_LABEL_SRC_DIR}" \
  --strict | tee "${OUT_DIR}/prepare_green_labels.stdout.txt"

echo "[${step}] Build green specialist splits"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/prepare_green_specialist_splits.py" \
  --green-train "${GREEN_LABEL_SRC_DIR}/train_labels.txt" \
  --green-val "${GREEN_LABEL_SRC_DIR}/val_labels.txt" \
  --green-test "${GREEN_LABEL_SRC_DIR}/test_labels.txt" \
  --output-dir "${LABELS_DIR}" \
  --train-cap-per-province "${TRAIN_CAP_PER_PROVINCE}" \
  --train-max-major-ratio "${TRAIN_MAX_MAJOR_RATIO}" \
  --val-balanced-per-province "${VAL_BALANCED_PER_PROVINCE}" \
  --test-balanced-per-province "${TEST_BALANCED_PER_PROVINCE}" \
  --seed "${SEED}" | tee "${OUT_DIR}/split_manifest.stdout.json"

TRAIN_FULL="${LABELS_DIR}/train_green_full_labels.txt"
TRAIN_BAL="${LABELS_DIR}/train_green_balanced_labels.txt"
VAL_FULL="${LABELS_DIR}/val_green_full_labels.txt"
VAL_BAL="${LABELS_DIR}/val_green_balanced_labels.txt"
TEST_FULL="${LABELS_DIR}/test_green_full_labels.txt"
TEST_BAL="${LABELS_DIR}/test_green_balanced_labels.txt"

COMMON_EVAL_ARGS=(
  --test_img_dirs "${GREEN_ROOT}"
  --data_mode ccpd_board
  --ocr_channel_order "${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER}"
  --ocr_crop_mode "${ARM_PERSPECTIVE_OCR_CROP_MODE}"
  --ocr_quad_pad_ratio "${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"
  --ocr_resize_mode "${ARM_PERSPECTIVE_OCR_RESIZE_MODE}"
  --ocr_resize_kernel "${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL}"
  --ocr_preproc "${ARM_PERSPECTIVE_OCR_PREPROC}"
  --ocr_min_occ_ratio "${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO}"
  --test_batch_size 120
  --num_workers "${NUM_WORKERS}"
  --cuda "${USE_CUDA}"
)

run_eval() {
  local txt_file="$1"
  local model_path="$2"
  local out_json="$3"
  "${ENV_PY}" "${ROOT_DIR}/eval_lpr_detailed.py" \
    "${COMMON_EVAL_ARGS[@]}" \
    --txt_file "${txt_file}" \
    --pretrained_model "${model_path}" \
    --out_json "${out_json}" | tee "${out_json%.json}.stdout.json"
}

echo "[${step}] Baseline green metrics"
step=$((step + 1))
run_eval "${TEST_FULL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_full_test_metrics.json"
run_eval "${TEST_BAL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_balanced_test_metrics.json"

TRAIN_COMMON_ARGS=(
  --train_img_dirs "${GREEN_ROOT}"
  --test_img_dirs "${GREEN_ROOT}"
  --data_mode ccpd_board
  --ocr_channel_order "${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER}"
  --ocr_crop_mode "${ARM_PERSPECTIVE_OCR_CROP_MODE}"
  --ocr_quad_pad_ratio "${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"
  --ocr_resize_mode "${ARM_PERSPECTIVE_OCR_RESIZE_MODE}"
  --ocr_resize_kernel "${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL}"
  --ocr_preproc "${ARM_PERSPECTIVE_OCR_PREPROC}"
  --ocr_min_occ_ratio "${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO}"
  --province_balance_mode inv_sqrt
  --first_char_time_steps 6
  --selection_proxy_eval_samples 5000
  --train_plate_box_aug_mode none
  --train_plate_box_aug_prob 0.0
  --save_interval 2000
  --train_batch_size 64
  --test_batch_size 120
  --num_workers "${NUM_WORKERS}"
  --cuda "${USE_CUDA}"
)

echo "[${step}] Stage A training (full green)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_FULL}" \
  --test_txt_file "${VAL_BAL}" \
  --first_char_aux_weight "${STAGEA_FIRST_CHAR_AUX}" \
  --pretrained_model "${BASE_MODEL}" \
  --learning_rate "${STAGEA_LR}" \
  --lr_schedule "${STAGEA_LR_SCHEDULE[@]}" \
  --save_folder "${WEIGHTS_A}/" \
  --max_epoch "${STAGEA_EPOCHS}" | tee "${OUT_DIR}/train_stageA.log"

MODEL_A="${WEIGHTS_A}/Final_LPRNet_model.pth"
if [[ ! -f "${MODEL_A}" ]]; then
  echo "Stage A final model not found: ${MODEL_A}" >&2
  exit 1
fi

echo "[${step}] Stage B training (balanced green)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_BAL}" \
  --test_txt_file "${VAL_BAL}" \
  --first_char_aux_weight "${STAGEB_FIRST_CHAR_AUX}" \
  --pretrained_model "${MODEL_A}" \
  --learning_rate "${STAGEB_LR}" \
  --lr_schedule "${STAGEB_LR_SCHEDULE[@]}" \
  --save_folder "${WEIGHTS_B}/" \
  --max_epoch "${STAGEB_EPOCHS}" | tee "${OUT_DIR}/train_stageB.log"

MODEL_B="${WEIGHTS_B}/Final_LPRNet_model.pth"
if [[ ! -f "${MODEL_B}" ]]; then
  echo "Stage B final model not found: ${MODEL_B}" >&2
  exit 1
fi

echo "[${step}] Stage C training (full green)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_FULL}" \
  --test_txt_file "${VAL_FULL}" \
  --first_char_aux_weight "${STAGEC_FIRST_CHAR_AUX}" \
  --pretrained_model "${MODEL_B}" \
  --learning_rate "${STAGEC_LR}" \
  --lr_schedule "${STAGEC_LR_SCHEDULE[@]}" \
  --save_folder "${WEIGHTS_C}/" \
  --max_epoch "${STAGEC_EPOCHS}" | tee "${OUT_DIR}/train_stageC.log"

FINAL_MODEL="${WEIGHTS_C}/Final_LPRNet_model.pth"
if [[ ! -f "${FINAL_MODEL}" ]]; then
  echo "Stage C final model not found: ${FINAL_MODEL}" >&2
  exit 1
fi

echo "[${step}] Final green metrics"
step=$((step + 1))
run_eval "${TEST_FULL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_full_metrics.json"
run_eval "${TEST_BAL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_balanced_metrics.json"

echo "[${step}] Green acceptance gate"
step=$((step + 1))
set +e
"${ENV_PY}" "${ROOT_DIR}/evaluate_green_specialist_acceptance.py" \
  --baseline-green-full "${OUT_DIR}/baseline_green_full_test_metrics.json" \
  --new-green-full "${OUT_DIR}/test_green_full_metrics.json" \
  --baseline-green-balanced "${OUT_DIR}/baseline_green_balanced_test_metrics.json" \
  --new-green-balanced "${OUT_DIR}/test_green_balanced_metrics.json" \
  --green-full-gain-min-pp "${GREEN_FULL_GAIN_MIN_PP}" \
  --green-balanced-gain-min-pp "${GREEN_BALANCED_GAIN_MIN_PP}" \
  --out-json "${OUT_DIR}/acceptance.json" | tee "${OUT_DIR}/acceptance.stdout.json"
ACCEPT_RC=$?
set -e

echo "[${step}] Build scorecard"
step=$((step + 1))
"${ENV_PY}" - <<PY | tee "${OUT_DIR}/scorecard.stdout.json"
import json
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(${OUT_DIR@Q})
acceptance = json.loads((run_dir / "acceptance.json").read_text(encoding="utf-8"))
scorecard = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "base_model": ${BASE_MODEL@Q},
    "final_model": str(run_dir / "weights_stageC" / "Final_LPRNet_model.pth"),
    "acceptance": acceptance,
}
(run_dir / "scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(scorecard, ensure_ascii=False, indent=2))
PY

if [[ "${EXPORT_RKNN}" == "true" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found; cannot export RKNN." >&2
    exit 1
  fi
  echo "[${step}] Export ONNX + RKNN"
  step=$((step + 1))
  ONNX_PATH="${WEIGHTS_C}/LPRNet_stage3_rk3568_fp16.onnx"
  RKNN_PATH="${WEIGHTS_C}/LPRNet_stage3_rk3568_fp16.rknn"
  "${ENV_PY}" "${ROOT_DIR}/export_onnx_rknn_compatible.py" \
    --weights "${FINAL_MODEL}" \
    --output "${ONNX_PATH}"
  env CONDA_NO_PLUGINS=true conda run -p "${RKNN_ENV_PREFIX}" python "${ROOT_DIR}/custom_rknn_convert.py" \
    "${ONNX_PATH}" \
    --target-platform rk3568 \
    --dtype fp \
    --output "${RKNN_PATH}" \
    --input-color-order bgr \
    --model-color-order bgr
fi

echo "[${step}] Export key paths"
step=$((step + 1))
echo "run_dir=${OUT_DIR}"
echo "base_model_blue_keep=${BASE_MODEL}"
echo "final_green_model=${FINAL_MODEL}"
echo "acceptance_json=${OUT_DIR}/acceptance.json"
echo "scorecard_json=${OUT_DIR}/scorecard.json"
echo "labels_dir=${LABELS_DIR}"
echo "Done"

if [[ ${ACCEPT_RC} -ne 0 ]]; then
  echo "Acceptance gate failed. Check ${OUT_DIR}/acceptance.stdout.json" >&2
  exit ${ACCEPT_RC}
fi
