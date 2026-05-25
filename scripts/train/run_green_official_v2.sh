#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_green_official_v2.sh [options]

Options:
  --run-tag <tag>
  --base-model <path>                             (default: ./weights_red_stage3/Final_LPRNet_model.pth)
  --green-root <path>                             (default: ./CCPD2020/ccpd_green)
  --target-green-balanced-v2-exact <float>        (default: 0.50)
  --target-green-balanced-v2-macro <float>        (default: 0.35)
  --full-guardrail-min-exact <float>              (default: 0.3159848182)
  --train-balanced-v2-per-province <int>          (default: 50)
  --train-full-capped-v2-per-province <int>       (default: 300)
  --train-full-capped-v2-max-major-ratio <float>  (default: 0.35)
  --val-balanced-v2-per-province <int>            (default: 20)
  --test-balanced-v2-per-province <int>           (default: 40)
  --eval-decode-mode <mode>                       (default: green_ctc_beam) [greedy|green_ctc_beam]
  --beam-size <int>                               (default: 30)
  --beam-topk <int>                               (default: 15)
  --seed <int>                                    (default: 20260320)
  --help
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ROOT_DIR}/.conda/bin/python"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

RUN_TAG="green_official_v2_prod_${DATE_TAG}"
BASE_MODEL="${ROOT_DIR}/weights_red_stage3/Final_LPRNet_model.pth"
GREEN_ROOT="${ROOT_DIR}/CCPD2020/ccpd_green"

TARGET_GREEN_BALANCED_V2_EXACT="0.50"
TARGET_GREEN_BALANCED_V2_MACRO="0.35"
FULL_GUARDRAIL_MIN_EXACT="0.31598481821813826"

TRAIN_CAP_PER_PROVINCE="2200"
TRAIN_MAX_MAJOR_RATIO="0.90"
VAL_BALANCED_PER_PROVINCE="120"
TEST_BALANCED_PER_PROVINCE="260"

TRAIN_BALANCED_V2_PER_PROVINCE="50"
TRAIN_FULL_CAPPED_V2_PER_PROVINCE="300"
TRAIN_FULL_CAPPED_V2_MAX_MAJOR_RATIO="0.35"
VAL_BALANCED_V2_PER_PROVINCE="20"
TEST_BALANCED_V2_PER_PROVINCE="40"

STAGEA_EPOCHS="24"
STAGEB_EPOCHS="18"
STAGEC_EPOCHS="12"

STAGEA_LR="0.00003"
STAGEB_LR="0.000008"
STAGEC_LR="0.000003"
STAGEA_LR_SCHEDULE=(10 18)
STAGEB_LR_SCHEDULE=(7 13)
STAGEC_LR_SCHEDULE=(4 8)

STAGEA_FIRST_CHAR_AUX="0.05"
STAGEB_FIRST_CHAR_AUX="0.03"
STAGEC_FIRST_CHAR_AUX="0.02"
SECOND_CHAR_AUX="0.0"
NE_TYPE_AUX="0.0"

STAGEA_PROVINCE_BALANCE_MODE="inv"
STAGEB_PROVINCE_BALANCE_MODE="inv_sqrt"
STAGEC_PROVINCE_BALANCE_MODE="inv_sqrt"
STAGEA_PROVINCE_BALANCE_CLIP="4.0"
STAGEB_PROVINCE_BALANCE_CLIP="3.0"
STAGEC_PROVINCE_BALANCE_CLIP="2.5"

STRATA_BALANCE_MODE="none"
STRATA_BALANCE_CLIP="0.0"
ADJ_REPEAT_SAMPLE_WEIGHT="1.25"
STAGEC_SECONDARY_SAMPLE_WEIGHT="1.8"

USE_CUDA="${USE_CUDA:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SELECTION_EVAL_SAMPLES="5000"
EARLY_STOP_PATIENCE="4"
EARLY_STOP_REGRESSION_PATIENCE="2"
EARLY_STOP_REGRESSION_PP="5.0"
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
    --target-green-balanced-v2-exact) TARGET_GREEN_BALANCED_V2_EXACT="$2"; shift 2 ;;
    --target-green-balanced-v2-macro) TARGET_GREEN_BALANCED_V2_MACRO="$2"; shift 2 ;;
    --full-guardrail-min-exact) FULL_GUARDRAIL_MIN_EXACT="$2"; shift 2 ;;
    --train-balanced-v2-per-province) TRAIN_BALANCED_V2_PER_PROVINCE="$2"; shift 2 ;;
    --train-full-capped-v2-per-province) TRAIN_FULL_CAPPED_V2_PER_PROVINCE="$2"; shift 2 ;;
    --train-full-capped-v2-max-major-ratio) TRAIN_FULL_CAPPED_V2_MAX_MAJOR_RATIO="$2"; shift 2 ;;
    --val-balanced-v2-per-province) VAL_BALANCED_V2_PER_PROVINCE="$2"; shift 2 ;;
    --test-balanced-v2-per-province) TEST_BALANCED_V2_PER_PROVINCE="$2"; shift 2 ;;
    --eval-decode-mode) EVAL_DECODE_MODE="$2"; shift 2 ;;
    --beam-size) BEAM_SIZE="$2"; shift 2 ;;
    --beam-topk) BEAM_TOPK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

EVAL_DECODE_MODE="${EVAL_DECODE_MODE:-green_ctc_beam}"
BEAM_SIZE="${BEAM_SIZE:-30}"
BEAM_TOPK="${BEAM_TOPK:-15}"

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
echo "[Config] target_green_balanced_v2_exact=${TARGET_GREEN_BALANCED_V2_EXACT}"
echo "[Config] target_green_balanced_v2_macro=${TARGET_GREEN_BALANCED_V2_MACRO}"
echo "[Config] full_guardrail_min_exact=${FULL_GUARDRAIL_MIN_EXACT}"
echo "[Config] eval_decode_mode=${EVAL_DECODE_MODE} beam=${BEAM_SIZE}/${BEAM_TOPK}"
echo "[Config] arm_perspective_ocr=ch:${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER} crop:${ARM_PERSPECTIVE_OCR_CROP_MODE} resize:${ARM_PERSPECTIVE_OCR_RESIZE_MODE}/${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL} preproc:${ARM_PERSPECTIVE_OCR_PREPROC} min_occ:${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO} quad_pad:${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"

step=1

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

echo "[${step}] Build CCPD green labels"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/prepare_ccpd_green_labels.py" \
  --dataset-root "${GREEN_ROOT}" \
  --output-dir "${GREEN_LABEL_SRC_DIR}" \
  --strict | tee "${OUT_DIR}/prepare_green_labels.stdout.txt"

echo "[${step}] Build strict green splits (legacy + v2)"
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
  --train-balanced-v2-per-province "${TRAIN_BALANCED_V2_PER_PROVINCE}" \
  --train-full-capped-v2-per-province "${TRAIN_FULL_CAPPED_V2_PER_PROVINCE}" \
  --train-full-capped-v2-max-major-ratio "${TRAIN_FULL_CAPPED_V2_MAX_MAJOR_RATIO}" \
  --val-balanced-v2-per-province "${VAL_BALANCED_V2_PER_PROVINCE}" \
  --test-balanced-v2-per-province "${TEST_BALANCED_V2_PER_PROVINCE}" \
  --seed "${SEED}" | tee "${OUT_DIR}/split_manifest.stdout.json"

TRAIN_FULL="${LABELS_DIR}/train_green_full_labels.txt"
TRAIN_BAL="${LABELS_DIR}/train_green_balanced_labels.txt"
TEST_FULL="${LABELS_DIR}/test_green_full_labels.txt"
TEST_BAL="${LABELS_DIR}/test_green_balanced_labels.txt"
TRAIN_BAL_V2="${LABELS_DIR}/train_green_balanced_v2_labels.txt"
TRAIN_FULL_CAPPED_V2="${LABELS_DIR}/train_green_full_capped_v2_labels.txt"
VAL_BAL_V2="${LABELS_DIR}/val_green_balanced_v2_labels.txt"
TEST_BAL_V2="${LABELS_DIR}/test_green_balanced_v2_labels.txt"

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
  --strata_balance_mode "${STRATA_BALANCE_MODE}"
  --strata_balance_clip "${STRATA_BALANCE_CLIP}"
  --adj_repeat_sample_weight "${ADJ_REPEAT_SAMPLE_WEIGHT}"
  --second_char_aux_weight "${SECOND_CHAR_AUX}"
  --ne_type_aux_weight "${NE_TYPE_AUX}"
  --selection_strategy balanced_tuple
  --selection_decode_mode "${EVAL_DECODE_MODE}"
  --selection_beam_size "${BEAM_SIZE}"
  --selection_beam_topk "${BEAM_TOPK}"
  --selection_proxy_eval_samples "${SELECTION_EVAL_SAMPLES}"
  --early_stop_patience "${EARLY_STOP_PATIENCE}"
  --early_stop_regression_patience "${EARLY_STOP_REGRESSION_PATIENCE}"
  --early_stop_regression_pp "${EARLY_STOP_REGRESSION_PP}"
  --first_char_time_steps 6
  --train_plate_box_aug_mode none
  --train_plate_box_aug_prob 0.0
  --save_interval 2000
  --train_batch_size 64
  --test_batch_size 120
  --num_workers "${NUM_WORKERS}"
  --cuda "${USE_CUDA}"
)

echo "[${step}] Baseline metrics"
step=$((step + 1))
run_eval "${TEST_BAL_V2}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_balanced_v2_test_metrics.json"
run_eval "${TEST_FULL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_full_test_metrics.json"
run_eval "${TEST_BAL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_balanced_legacy_test_metrics.json"

echo "[${step}] Stage A (balanced_v2 warmup, val_balanced_v2 selection)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_BAL_V2}" \
  --test_txt_file "${VAL_BAL_V2}" \
  --province_balance_mode "${STAGEA_PROVINCE_BALANCE_MODE}" \
  --province_balance_clip "${STAGEA_PROVINCE_BALANCE_CLIP}" \
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

echo "[${step}] Stage B (full_capped_v2 fit, val_balanced_v2 selection)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_FULL_CAPPED_V2}" \
  --test_txt_file "${VAL_BAL_V2}" \
  --province_balance_mode "${STAGEB_PROVINCE_BALANCE_MODE}" \
  --province_balance_clip "${STAGEB_PROVINCE_BALANCE_CLIP}" \
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

echo "[${step}] Stage C (full_capped_v2 + balanced_v2 replay, val_balanced_v2 selection)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_FULL_CAPPED_V2}" \
  --secondary_train_img_dirs "${GREEN_ROOT}" \
  --secondary_train_txt_file "${TRAIN_BAL_V2}" \
  --secondary_train_sample_weight "${STAGEC_SECONDARY_SAMPLE_WEIGHT}" \
  --test_txt_file "${VAL_BAL_V2}" \
  --province_balance_mode "${STAGEC_PROVINCE_BALANCE_MODE}" \
  --province_balance_clip "${STAGEC_PROVINCE_BALANCE_CLIP}" \
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

echo "[${step}] Final metrics"
step=$((step + 1))
run_eval "${TEST_BAL_V2}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_balanced_v2_metrics.json"
run_eval "${TEST_FULL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_full_metrics.json"
run_eval "${TEST_BAL}" "${FINAL_MODEL}" "${OUT_DIR}/test_green_balanced_legacy_metrics.json"

echo "[${step}] Build acceptance + scorecard"
step=$((step + 1))
"${ENV_PY}" - <<PY | tee "${OUT_DIR}/scorecard.stdout.json"
import json
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(${OUT_DIR@Q})
balanced = json.loads((run_dir / "test_green_balanced_v2_metrics.json").read_text(encoding="utf-8"))
full = json.loads((run_dir / "test_green_full_metrics.json").read_text(encoding="utf-8"))
legacy_bal = json.loads((run_dir / "test_green_balanced_legacy_metrics.json").read_text(encoding="utf-8"))
baseline_balanced = json.loads((run_dir / "baseline_green_balanced_v2_test_metrics.json").read_text(encoding="utf-8"))
baseline_full = json.loads((run_dir / "baseline_green_full_test_metrics.json").read_text(encoding="utf-8"))

target_balanced_exact = float(${TARGET_GREEN_BALANCED_V2_EXACT@Q})
target_balanced_macro = float(${TARGET_GREEN_BALANCED_V2_MACRO@Q})
full_guardrail = float(${FULL_GUARDRAIL_MIN_EXACT@Q})

balanced_exact = float(balanced.get("exact_plate_acc", 0.0))
balanced_macro = float(balanced.get("province_macro_exact_acc", 0.0))
full_exact = float(full.get("exact_plate_acc", 0.0))
passed = (
    balanced_exact >= target_balanced_exact and
    balanced_macro >= target_balanced_macro and
    full_exact >= full_guardrail and
    float(balanced.get("empty_pred_rate", 0.0)) < 0.015 and
    float(balanced.get("short_pred_le4_rate", 0.0)) < 0.035
)

acceptance = {
    "target_green_balanced_v2_exact": target_balanced_exact,
    "target_green_balanced_v2_macro": target_balanced_macro,
    "full_guardrail_min_exact": full_guardrail,
    "final_green_balanced_v2_exact": balanced_exact,
    "final_green_balanced_v2_macro": balanced_macro,
    "final_green_full_exact": full_exact,
    "pass_balanced_exact": balanced_exact >= target_balanced_exact,
    "pass_balanced_macro": balanced_macro >= target_balanced_macro,
    "pass_full_guardrail": full_exact >= full_guardrail,
    "pass_empty_pred_rate": float(balanced.get("empty_pred_rate", 0.0)) < 0.015,
    "pass_short_pred_rate": float(balanced.get("short_pred_le4_rate", 0.0)) < 0.035,
    "passed": passed,
}
(run_dir / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")

scorecard = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "base_model": ${BASE_MODEL@Q},
    "final_model": ${FINAL_MODEL@Q},
    "eval_decode_mode": ${EVAL_DECODE_MODE@Q},
    "beam_size": int(${BEAM_SIZE@Q}),
    "beam_topk": int(${BEAM_TOPK@Q}),
    "targets": acceptance,
    "baseline_green_balanced_v2_exact": float(baseline_balanced.get("exact_plate_acc", 0.0)),
    "baseline_green_full_exact": float(baseline_full.get("exact_plate_acc", 0.0)),
    "final_green_balanced_v2_metrics": {
        "exact_plate_acc": balanced_exact,
        "province_macro_exact_acc": balanced_macro,
        "province_macro_first_char_acc": float(balanced.get("province_macro_first_char_acc", 0.0)),
        "non_major_province_exact_acc": float(balanced.get("non_major_province_exact_acc", 0.0)),
        "major_province": balanced.get("major_province"),
        "major_province_ratio": float(balanced.get("major_province_ratio", 0.0)),
        "empty_pred_rate": float(balanced.get("empty_pred_rate", 0.0)),
        "short_pred_le4_rate": float(balanced.get("short_pred_le4_rate", 0.0)),
    },
    "final_green_full_metrics": {
        "exact_plate_acc": full_exact,
        "province_macro_exact_acc": float(full.get("province_macro_exact_acc", 0.0)),
        "major_province": full.get("major_province"),
        "major_province_ratio": float(full.get("major_province_ratio", 0.0)),
    },
    "final_green_balanced_legacy_metrics": {
        "exact_plate_acc": float(legacy_bal.get("exact_plate_acc", 0.0)),
        "province_macro_exact_acc": float(legacy_bal.get("province_macro_exact_acc", 0.0)),
    },
}
(run_dir / "scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(scorecard, ensure_ascii=False, indent=2))
PY

echo "[${step}] Export key paths"
step=$((step + 1))
echo "run_dir=${OUT_DIR}"
echo "base_model=${BASE_MODEL}"
echo "final_green_model=${FINAL_MODEL}"
echo "acceptance_json=${OUT_DIR}/acceptance.json"
echo "scorecard_json=${OUT_DIR}/scorecard.json"
echo "labels_dir=${LABELS_DIR}"
echo "Done"

"${ENV_PY}" - <<PY
import json
from pathlib import Path
report = json.loads(Path(${OUT_DIR@Q}, "acceptance.json").read_text(encoding="utf-8"))
raise SystemExit(0 if report.get("passed", False) else 2)
PY
