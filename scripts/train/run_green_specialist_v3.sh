#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_green_specialist_v3.sh [options]

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
  --first-char-aux <float>               (default: 0.03)
  --second-char-aux <float>              (default: 0.10)
  --ne-type-aux <float>                  (default: 0.15)
  --province-balance-mode <mode>         (default: inv) [none|inv_sqrt|inv]
  --province-balance-clip <float>        (default: 12)
  --strata-balance-mode <mode>           (default: inv) [none|inv_sqrt|inv]
  --strata-balance-clip <float>          (default: 10)
  --adj-repeat-sample-weight <float>     (default: 2.0)
  --hard-sample-weight <float>           (default: 3.5)
  --hard-max-per-stratum <int>           (default: 600)
  --eval-decode-mode <mode>              (default: green_ctc_beam) [greedy|green_ctc_beam]
  --beam-size <int>                      (default: 30)
  --beam-topk <int>                      (default: 15)
  --green-full-gain-min-pp <float>       (default: 0.20)
  --green-balanced-gain-min-pp <float>   (default: 0.20)
  --seed <int>                           (default: 20260320)
  --help
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ROOT_DIR}/.conda/bin/python"
DATE_TAG="$(date +%Y%m%d)"

RUN_TAG="green_specialist_v3_${DATE_TAG}"
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

FIRST_CHAR_AUX="0.03"
SECOND_CHAR_AUX="0.10"
NE_TYPE_AUX="0.15"

PROVINCE_BALANCE_MODE="inv"
PROVINCE_BALANCE_CLIP="12"
STRATA_BALANCE_MODE="inv"
STRATA_BALANCE_CLIP="10"
ADJ_REPEAT_SAMPLE_WEIGHT="2.0"

HARD_SAMPLE_WEIGHT="3.5"
HARD_MAX_PER_STRATUM="600"

EVAL_DECODE_MODE="green_ctc_beam"
BEAM_SIZE="30"
BEAM_TOPK="15"

GREEN_FULL_GAIN_MIN_PP="0.20"
GREEN_BALANCED_GAIN_MIN_PP="0.20"

USE_CUDA="${USE_CUDA:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="20260320"

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
    --help) usage; exit 0 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --green-root) GREEN_ROOT="$2"; shift 2 ;;
    --train-cap-per-province) TRAIN_CAP_PER_PROVINCE="$2"; shift 2 ;;
    --train-max-major-ratio) TRAIN_MAX_MAJOR_RATIO="$2"; shift 2 ;;
    --val-balanced-per-province) VAL_BALANCED_PER_PROVINCE="$2"; shift 2 ;;
    --test-balanced-per-province) TEST_BALANCED_PER_PROVINCE="$2"; shift 2 ;;
    --stagea-epochs) STAGEA_EPOCHS="$2"; shift 2 ;;
    --stageb-epochs) STAGEB_EPOCHS="$2"; shift 2 ;;
    --stagec-epochs) STAGEC_EPOCHS="$2"; shift 2 ;;
    --stagea-lr) STAGEA_LR="$2"; shift 2 ;;
    --stageb-lr) STAGEB_LR="$2"; shift 2 ;;
    --stagec-lr) STAGEC_LR="$2"; shift 2 ;;
    --first-char-aux) FIRST_CHAR_AUX="$2"; shift 2 ;;
    --second-char-aux) SECOND_CHAR_AUX="$2"; shift 2 ;;
    --ne-type-aux) NE_TYPE_AUX="$2"; shift 2 ;;
    --province-balance-mode) PROVINCE_BALANCE_MODE="$2"; shift 2 ;;
    --province-balance-clip) PROVINCE_BALANCE_CLIP="$2"; shift 2 ;;
    --strata-balance-mode) STRATA_BALANCE_MODE="$2"; shift 2 ;;
    --strata-balance-clip) STRATA_BALANCE_CLIP="$2"; shift 2 ;;
    --adj-repeat-sample-weight) ADJ_REPEAT_SAMPLE_WEIGHT="$2"; shift 2 ;;
    --hard-sample-weight) HARD_SAMPLE_WEIGHT="$2"; shift 2 ;;
    --hard-max-per-stratum) HARD_MAX_PER_STRATUM="$2"; shift 2 ;;
    --eval-decode-mode) EVAL_DECODE_MODE="$2"; shift 2 ;;
    --beam-size) BEAM_SIZE="$2"; shift 2 ;;
    --beam-topk) BEAM_TOPK="$2"; shift 2 ;;
    --green-full-gain-min-pp) GREEN_FULL_GAIN_MIN_PP="$2"; shift 2 ;;
    --green-balanced-gain-min-pp) GREEN_BALANCED_GAIN_MIN_PP="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

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
echo "[Config] eval_decode_mode=${EVAL_DECODE_MODE} beam=${BEAM_SIZE}/${BEAM_TOPK}"
echo "[Config] province_balance=${PROVINCE_BALANCE_MODE} clip=${PROVINCE_BALANCE_CLIP} strata=${STRATA_BALANCE_MODE} strata_clip=${STRATA_BALANCE_CLIP}"
echo "[Config] aux first=${FIRST_CHAR_AUX} second=${SECOND_CHAR_AUX} ne=${NE_TYPE_AUX} hard_weight=${HARD_SAMPLE_WEIGHT}"
echo "[Config] arm_perspective_ocr=ch:${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER} crop:${ARM_PERSPECTIVE_OCR_CROP_MODE} resize:${ARM_PERSPECTIVE_OCR_RESIZE_MODE}/${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL} preproc:${ARM_PERSPECTIVE_OCR_PREPROC} min_occ:${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO} quad_pad:${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"

step=1

echo "[${step}] Build CCPD green labels"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/prepare_ccpd_green_labels.py" \
  --dataset-root "${GREEN_ROOT}" \
  --output-dir "${GREEN_LABEL_SRC_DIR}" \
  --strict | tee "${OUT_DIR}/prepare_green_labels.stdout.txt"

echo "[${step}] Build strict green splits"
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
HARD_A_TXT="${LABELS_DIR}/hard_stageA_labels.txt"
HARD_B_TXT="${LABELS_DIR}/hard_stageB_labels.txt"

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
  --decode_mode "${EVAL_DECODE_MODE}"
  --beam_size "${BEAM_SIZE}"
  --beam_topk "${BEAM_TOPK}"
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

run_eval_greedy() {
  local txt_file="$1"
  local model_path="$2"
  local out_json="$3"
  "${ENV_PY}" "${ROOT_DIR}/eval_lpr_detailed.py" \
    --test_img_dirs "${GREEN_ROOT}" \
    --data_mode ccpd_board \
    --ocr_channel_order "${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER}" \
    --ocr_crop_mode "${ARM_PERSPECTIVE_OCR_CROP_MODE}" \
    --ocr_quad_pad_ratio "${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}" \
    --ocr_resize_mode "${ARM_PERSPECTIVE_OCR_RESIZE_MODE}" \
    --ocr_resize_kernel "${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL}" \
    --ocr_preproc "${ARM_PERSPECTIVE_OCR_PREPROC}" \
    --ocr_min_occ_ratio "${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO}" \
    --decode_mode greedy \
    --test_batch_size 120 \
    --num_workers "${NUM_WORKERS}" \
    --cuda "${USE_CUDA}" \
    --txt_file "${txt_file}" \
    --pretrained_model "${model_path}" \
    --out_json "${out_json}" | tee "${out_json%.json}.stdout.json"
}

mine_hard() {
  local model_path="$1"
  local in_txt="$2"
  local out_txt="$3"
  local out_json="$4"
  "${ENV_PY}" "${ROOT_DIR}/mine_green_hard_examples.py" \
    --pretrained_model "${model_path}" \
    --img_dirs "${GREEN_ROOT}" \
    --txt_file "${in_txt}" \
    --output_txt "${out_txt}" \
    --output_json "${out_json}" \
    --data_mode ccpd_board \
    --ocr_channel_order "${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER}" \
    --ocr_crop_mode "${ARM_PERSPECTIVE_OCR_CROP_MODE}" \
    --ocr_resize_mode "${ARM_PERSPECTIVE_OCR_RESIZE_MODE}" \
    --ocr_resize_kernel "${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL}" \
    --ocr_preproc "${ARM_PERSPECTIVE_OCR_PREPROC}" \
    --ocr_min_occ_ratio "${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO}" \
    --ocr_quad_pad_ratio "${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}" \
    --decode_mode greedy \
    --max_per_stratum "${HARD_MAX_PER_STRATUM}" \
    --min_edit_distance 2 \
    --batch_size 120 \
    --num_workers "${NUM_WORKERS}" \
    --cuda "${USE_CUDA}" \
    --seed "${SEED}" | tee "${out_json%.json}.stdout.json"
}

echo "[${step}] Baseline metrics"
step=$((step + 1))
run_eval "${TEST_FULL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_full_test_metrics.json"
run_eval "${TEST_BAL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_balanced_test_metrics.json"
run_eval_greedy "${TEST_FULL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_full_test_metrics_greedy.json"

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
  --province_balance_mode "${PROVINCE_BALANCE_MODE}"
  --province_balance_clip "${PROVINCE_BALANCE_CLIP}"
  --strata_balance_mode "${STRATA_BALANCE_MODE}"
  --strata_balance_clip "${STRATA_BALANCE_CLIP}"
  --adj_repeat_sample_weight "${ADJ_REPEAT_SAMPLE_WEIGHT}"
  --first_char_aux_weight "${FIRST_CHAR_AUX}"
  --second_char_aux_weight "${SECOND_CHAR_AUX}"
  --ne_type_aux_weight "${NE_TYPE_AUX}"
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

echo "[${step}] Stage A training (balanced)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_BAL}" \
  --test_txt_file "${VAL_BAL}" \
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

echo "[${step}] Mine hard examples from Stage A"
step=$((step + 1))
mine_hard "${MODEL_A}" "${TRAIN_FULL}" "${HARD_A_TXT}" "${OUT_DIR}/hard_stageA_stats.json"

echo "[${step}] Stage B training (full + hard A)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_FULL}" \
  --test_txt_file "${VAL_BAL}" \
  --pseudo_anchor_img_dirs "${GREEN_ROOT}" \
  --pseudo_anchor_train_txt_file "${HARD_A_TXT}" \
  --pseudo_anchor_sample_weight "${HARD_SAMPLE_WEIGHT}" \
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

echo "[${step}] Mine hard examples from Stage B"
step=$((step + 1))
mine_hard "${MODEL_B}" "${TRAIN_FULL}" "${HARD_B_TXT}" "${OUT_DIR}/hard_stageB_stats.json"

echo "[${step}] Stage C training (full + hard B)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_FULL}" \
  --test_txt_file "${VAL_FULL}" \
  --pseudo_anchor_img_dirs "${GREEN_ROOT}" \
  --pseudo_anchor_train_txt_file "${HARD_B_TXT}" \
  --pseudo_anchor_sample_weight "${HARD_SAMPLE_WEIGHT}" \
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

echo "[${step}] Final metrics"
step=$((step + 1))
run_eval "${TEST_FULL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_full_metrics.json"
run_eval "${TEST_BAL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_balanced_metrics.json"
run_eval_greedy "${TEST_FULL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_full_metrics_greedy.json"

echo "[${step}] Acceptance gate (${EVAL_DECODE_MODE})"
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
    "eval_decode_mode": ${EVAL_DECODE_MODE@Q},
    "beam_size": int(${BEAM_SIZE@Q}),
    "beam_topk": int(${BEAM_TOPK@Q}),
    "province_balance_mode": ${PROVINCE_BALANCE_MODE@Q},
    "province_balance_clip": float(${PROVINCE_BALANCE_CLIP@Q}),
    "strata_balance_mode": ${STRATA_BALANCE_MODE@Q},
    "strata_balance_clip": float(${STRATA_BALANCE_CLIP@Q}),
    "adj_repeat_sample_weight": float(${ADJ_REPEAT_SAMPLE_WEIGHT@Q}),
    "hard_sample_weight": float(${HARD_SAMPLE_WEIGHT@Q}),
    "acceptance": acceptance,
}
(run_dir / "scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(scorecard, ensure_ascii=False, indent=2))
PY

echo "[${step}] Export key paths"
step=$((step + 1))
echo "run_dir=${OUT_DIR}"
echo "base_model_blue_keep=${BASE_MODEL}"
echo "final_green_model=${FINAL_MODEL}"
echo "acceptance_json=${OUT_DIR}/acceptance.json"
echo "scorecard_json=${OUT_DIR}/scorecard.json"
echo "labels_dir=${LABELS_DIR}"
echo "hard_stageA_txt=${HARD_A_TXT}"
echo "hard_stageB_txt=${HARD_B_TXT}"
echo "Done"

if [[ ${ACCEPT_RC} -ne 0 ]]; then
  echo "Acceptance gate failed. Check ${OUT_DIR}/acceptance.stdout.json" >&2
  exit ${ACCEPT_RC}
fi
