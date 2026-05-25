#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_green_from_v7.sh [options]

Options:
  --run-tag <tag>
  --base-model <path>
  --green-root <path>                  (default: ./CCPD2020/ccpd_green)
  --blue-root <path>                   (default: ./CCPD2019)
  --blue-train-txts <a,b,...>          (default: prepared_labels/ccpd2019 + hard_tilt train)
  --blue-val-normal <path>             (default: prepared_labels/ccpd2019/val_labels.txt)
  --blue-test-normal <path>            (default: prepared_labels/ccpd2019/test_labels.txt)
  --blue-val-hard <path>               (default: prepared_labels/ccpd2019_hard_tilt/val_labels.txt)
  --blue-test-hard <path>              (default: prepared_labels/ccpd2019_hard_tilt/test_labels.txt)
  --stagea-green-ratio <float>         (default: 0.60)
  --stageb-green-ratio <float>         (default: 0.80)
  --stagec-green-ratio <float>         (default: 0.70)
  --green-train-cap-per-province <int> (default: 900)
  --blue-train-cap-per-province <int>  (default: 1400)
  --green-train-max-major-ratio <float> (default: 0.40)
  --blue-train-max-major-ratio <float>  (default: 0.60)
  --green-val-balanced-per-province <int>  (default: 80)
  --green-test-balanced-per-province <int> (default: 220)
  --blue-val-gate-per-province <int>       (default: 120)
  --stagea-epochs <int>                (default: 6)
  --stageb-epochs <int>                (default: 6)
  --stagec-epochs <int>                (default: 4)
  --stagea-lr <float>                  (default: 0.00002)
  --stageb-lr <float>                  (default: 0.00001)
  --stagec-lr <float>                  (default: 0.000005)
  --stagea-first-char-aux <float>      (default: 0.15)
  --stageb-first-char-aux <float>      (default: 0.12)
  --stagec-first-char-aux <float>      (default: 0.08)
  --blue-drop-max-pp <float>           (default: 0.0)
  --green-gain-min-pp <float>          (default: 0.01)
  --seed <int>                         (default: 20260320)
  --export-rknn <bool>                 (default: false)
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

RUN_TAG="green_from_v7_${DATE_TAG}"
BASE_MODEL="${ROOT_DIR}/experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth"
GREEN_ROOT="${ROOT_DIR}/CCPD2020/ccpd_green"
BLUE_ROOT="${ROOT_DIR}/CCPD2019"
BLUE_TRAIN_TXTS="${ROOT_DIR}/prepared_labels/ccpd2019/train_labels.txt,${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/train_labels.txt"
BLUE_VAL_NORMAL="${ROOT_DIR}/prepared_labels/ccpd2019/val_labels.txt"
BLUE_TEST_NORMAL="${ROOT_DIR}/prepared_labels/ccpd2019/test_labels.txt"
BLUE_VAL_HARD="${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/val_labels.txt"
BLUE_TEST_HARD="${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/test_labels.txt"

STAGEA_GREEN_RATIO="0.60"
STAGEB_GREEN_RATIO="0.80"
STAGEC_GREEN_RATIO="0.70"

GREEN_TRAIN_CAP_PER_PROV="900"
BLUE_TRAIN_CAP_PER_PROV="1400"
GREEN_TRAIN_MAX_MAJOR_RATIO="0.40"
BLUE_TRAIN_MAX_MAJOR_RATIO="0.60"
GREEN_VAL_BALANCED_PER_PROV="80"
GREEN_TEST_BALANCED_PER_PROV="220"
BLUE_VAL_GATE_PER_PROV="120"

STAGEA_EPOCHS="6"
STAGEB_EPOCHS="6"
STAGEC_EPOCHS="4"

STAGEA_LR="0.00002"
STAGEB_LR="0.00001"
STAGEC_LR="0.000005"
STAGEA_LR_SCHEDULE=(2 4)
STAGEB_LR_SCHEDULE=(2 4)
STAGEC_LR_SCHEDULE=(2 3)

STAGEA_FIRST_CHAR_AUX="0.15"
STAGEB_FIRST_CHAR_AUX="0.12"
STAGEC_FIRST_CHAR_AUX="0.08"

USE_CUDA="${USE_CUDA:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"

BLUE_DROP_MAX_PP="0.0"
GREEN_GAIN_MIN_PP="0.01"
SEED="20260320"
EXPORT_RKNN="false"

# OCR preprocessing contract locked to ARM OBB board path.
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
    --blue-root)
      BLUE_ROOT="$2"
      shift 2
      ;;
    --blue-train-txts)
      BLUE_TRAIN_TXTS="$2"
      shift 2
      ;;
    --blue-val-normal)
      BLUE_VAL_NORMAL="$2"
      shift 2
      ;;
    --blue-test-normal)
      BLUE_TEST_NORMAL="$2"
      shift 2
      ;;
    --blue-val-hard)
      BLUE_VAL_HARD="$2"
      shift 2
      ;;
    --blue-test-hard)
      BLUE_TEST_HARD="$2"
      shift 2
      ;;
    --stagea-green-ratio)
      STAGEA_GREEN_RATIO="$2"
      shift 2
      ;;
    --stageb-green-ratio)
      STAGEB_GREEN_RATIO="$2"
      shift 2
      ;;
    --stagec-green-ratio)
      STAGEC_GREEN_RATIO="$2"
      shift 2
      ;;
    --green-train-cap-per-province)
      GREEN_TRAIN_CAP_PER_PROV="$2"
      shift 2
      ;;
    --blue-train-cap-per-province)
      BLUE_TRAIN_CAP_PER_PROV="$2"
      shift 2
      ;;
    --green-train-max-major-ratio)
      GREEN_TRAIN_MAX_MAJOR_RATIO="$2"
      shift 2
      ;;
    --blue-train-max-major-ratio)
      BLUE_TRAIN_MAX_MAJOR_RATIO="$2"
      shift 2
      ;;
    --green-val-balanced-per-province)
      GREEN_VAL_BALANCED_PER_PROV="$2"
      shift 2
      ;;
    --green-test-balanced-per-province)
      GREEN_TEST_BALANCED_PER_PROV="$2"
      shift 2
      ;;
    --blue-val-gate-per-province)
      BLUE_VAL_GATE_PER_PROV="$2"
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
    --blue-drop-max-pp)
      BLUE_DROP_MAX_PP="$2"
      shift 2
      ;;
    --green-gain-min-pp)
      GREEN_GAIN_MIN_PP="$2"
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
if [[ ! -d "${BLUE_ROOT}" ]]; then
  echo "Blue dataset root not found: ${BLUE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Base model not found: ${BASE_MODEL}" >&2
  exit 1
fi
IFS=',' read -r -a BLUE_TRAIN_ARR <<< "${BLUE_TRAIN_TXTS}"
for p in "${BLUE_TRAIN_ARR[@]}"; do
  if [[ ! -f "${p}" ]]; then
    echo "Blue train label not found: ${p}" >&2
    exit 1
  fi
done
for p in "${BLUE_VAL_NORMAL}" "${BLUE_TEST_NORMAL}" "${BLUE_VAL_HARD}" "${BLUE_TEST_HARD}"; do
  if [[ ! -f "${p}" ]]; then
    echo "Blue eval label not found: ${p}" >&2
    exit 1
  fi
done

echo "[Config] run_tag=${RUN_TAG} out_dir=${OUT_DIR}"
echo "[Config] base_model=${BASE_MODEL}"
echo "[Config] green_root=${GREEN_ROOT}"
echo "[Config] blue_root=${BLUE_ROOT}"
echo "[Config] stage_green_ratio=A:${STAGEA_GREEN_RATIO} B:${STAGEB_GREEN_RATIO} C:${STAGEC_GREEN_RATIO}"
echo "[Config] arm_perspective_ocr=ch:${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER} crop:${ARM_PERSPECTIVE_OCR_CROP_MODE} resize:${ARM_PERSPECTIVE_OCR_RESIZE_MODE}/${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL} preproc:${ARM_PERSPECTIVE_OCR_PREPROC} min_occ:${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO} quad_pad:${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"

step=1

echo "[${step}] Build CCPD green labels"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/prepare_ccpd_green_labels.py" \
  --dataset-root "${GREEN_ROOT}" \
  --output-dir "${GREEN_LABEL_SRC_DIR}" \
  --strict | tee "${OUT_DIR}/prepare_green_labels.stdout.txt"

GREEN_TRAIN_LBL="${GREEN_LABEL_SRC_DIR}/train_labels.txt"
GREEN_VAL_LBL="${GREEN_LABEL_SRC_DIR}/val_labels.txt"
GREEN_TEST_LBL="${GREEN_LABEL_SRC_DIR}/test_labels.txt"

echo "[${step}] Build strict green+blue splits"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/build_green_blue_strict_splits.py" \
  --green-train "${GREEN_TRAIN_LBL}" \
  --green-val "${GREEN_VAL_LBL}" \
  --green-test "${GREEN_TEST_LBL}" \
  --blue-train "${BLUE_TRAIN_TXTS}" \
  --blue-val-normal "${BLUE_VAL_NORMAL}" \
  --blue-test-normal "${BLUE_TEST_NORMAL}" \
  --blue-val-hard "${BLUE_VAL_HARD}" \
  --blue-test-hard "${BLUE_TEST_HARD}" \
  --output-dir "${LABELS_DIR}" \
  --stagea-green-ratio "${STAGEA_GREEN_RATIO}" \
  --stageb-green-ratio "${STAGEB_GREEN_RATIO}" \
  --stagec-green-ratio "${STAGEC_GREEN_RATIO}" \
  --green-train-cap-per-province "${GREEN_TRAIN_CAP_PER_PROV}" \
  --blue-train-cap-per-province "${BLUE_TRAIN_CAP_PER_PROV}" \
  --green-train-max-major-ratio "${GREEN_TRAIN_MAX_MAJOR_RATIO}" \
  --blue-train-max-major-ratio "${BLUE_TRAIN_MAX_MAJOR_RATIO}" \
  --green-val-balanced-per-province "${GREEN_VAL_BALANCED_PER_PROV}" \
  --green-test-balanced-per-province "${GREEN_TEST_BALANCED_PER_PROV}" \
  --blue-val-gate-per-province "${BLUE_VAL_GATE_PER_PROV}" \
  --seed "${SEED}" | tee "${OUT_DIR}/split_manifest.stdout.json"

TRAIN_A="${LABELS_DIR}/train_mix_stageA_labels.txt"
TRAIN_B="${LABELS_DIR}/train_mix_stageB_labels.txt"
TRAIN_C="${LABELS_DIR}/train_mix_stageC_labels.txt"
VAL_GATE="${LABELS_DIR}/val_gate_labels.txt"
TEST_GREEN_FULL="${LABELS_DIR}/test_green_full_labels.txt"
TEST_GREEN_BAL="${LABELS_DIR}/test_green_balanced_labels.txt"
TEST_BLUE_NORMAL="${LABELS_DIR}/test_blue_normal_labels.txt"
TEST_BLUE_HARD="${LABELS_DIR}/test_blue_hard_labels.txt"

echo "[${step}] Verify stage leakage"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_A}" \
  --val "${VAL_GATE}" \
  --test "${TEST_GREEN_FULL}" \
  --fail-on-overlap \
  --out-json "${OUT_DIR}/split_stageA_stats.json" | tee "${OUT_DIR}/split_stageA_stats.stdout.json"

"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_B}" \
  --val "${VAL_GATE}" \
  --test "${TEST_GREEN_FULL}" \
  --fail-on-overlap \
  --out-json "${OUT_DIR}/split_stageB_stats.json" | tee "${OUT_DIR}/split_stageB_stats.stdout.json"

"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_C}" \
  --val "${VAL_GATE}" \
  --test "${TEST_GREEN_FULL}" \
  --fail-on-overlap \
  --out-json "${OUT_DIR}/split_stageC_stats.json" | tee "${OUT_DIR}/split_stageC_stats.stdout.json"

COMMON_EVAL_ARGS=(
  --test_img_dirs "${GREEN_ROOT},${BLUE_ROOT}"
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

echo "[${step}] Baseline metrics (blue hard/normal + green full/balanced)"
step=$((step + 1))
run_eval "${TEST_BLUE_HARD}" "${BASE_MODEL}" "${OUT_DIR}/baseline_blue_hard_test_metrics.json"
run_eval "${TEST_BLUE_NORMAL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_blue_normal_test_metrics.json"
run_eval "${TEST_GREEN_FULL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_full_test_metrics.json"
run_eval "${TEST_GREEN_BAL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_balanced_test_metrics.json"

TRAIN_COMMON_ARGS=(
  --train_img_dirs "${GREEN_ROOT},${BLUE_ROOT}"
  --test_img_dirs "${GREEN_ROOT},${BLUE_ROOT}"
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

echo "[${step}] Stage A training"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_A}" \
  --test_txt_file "${VAL_GATE}" \
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

echo "[${step}] Stage B training"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_B}" \
  --test_txt_file "${VAL_GATE}" \
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

echo "[${step}] Stage C training"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_C}" \
  --test_txt_file "${VAL_GATE}" \
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

echo "[${step}] Final metrics (blue hard/normal + green full/balanced)"
step=$((step + 1))
run_eval "${TEST_BLUE_HARD}" "${FINAL_MODEL}" "${OUT_DIR}/test_blue_hard_metrics.json"
run_eval "${TEST_BLUE_NORMAL}" "${FINAL_MODEL}" "${OUT_DIR}/test_blue_normal_metrics.json"
run_eval "${TEST_GREEN_FULL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_full_metrics.json"
run_eval "${TEST_GREEN_BAL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_balanced_metrics.json"

echo "[${step}] Acceptance gate (blue 0-regression + green uplift)"
step=$((step + 1))
set +e
"${ENV_PY}" "${ROOT_DIR}/evaluate_green_blue_acceptance.py" \
  --baseline-blue-hard "${OUT_DIR}/baseline_blue_hard_test_metrics.json" \
  --new-blue-hard "${OUT_DIR}/test_blue_hard_metrics.json" \
  --baseline-blue-normal "${OUT_DIR}/baseline_blue_normal_test_metrics.json" \
  --new-blue-normal "${OUT_DIR}/test_blue_normal_metrics.json" \
  --baseline-green-full "${OUT_DIR}/baseline_green_full_test_metrics.json" \
  --new-green-full "${OUT_DIR}/test_green_full_metrics.json" \
  --baseline-green-balanced "${OUT_DIR}/baseline_green_balanced_test_metrics.json" \
  --new-green-balanced "${OUT_DIR}/test_green_balanced_metrics.json" \
  --blue-drop-max-pp "${BLUE_DROP_MAX_PP}" \
  --green-gain-min-pp "${GREEN_GAIN_MIN_PP}" \
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
    "targets": {
        "blue_drop_max_pp": float(${BLUE_DROP_MAX_PP@Q}),
        "green_gain_min_pp": float(${GREEN_GAIN_MIN_PP@Q}),
    },
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
echo "final_model=${FINAL_MODEL}"
echo "acceptance_json=${OUT_DIR}/acceptance.json"
echo "scorecard_json=${OUT_DIR}/scorecard.json"
echo "labels_dir=${LABELS_DIR}"
echo "Done"

if [[ ${ACCEPT_RC} -ne 0 ]]; then
  echo "Acceptance gate failed. Check ${OUT_DIR}/acceptance.stdout.json" >&2
  exit ${ACCEPT_RC}
fi
