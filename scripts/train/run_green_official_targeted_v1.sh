#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_green_official_targeted_v1.sh [options]

Options:
  --run-tag <tag>
  --base-model <path>                             (default: ./weights_red_stage3/Final_LPRNet_model.pth)
  --green-root <path>                             (default: ./CCPD2020/ccpd_green)
  --targeted-root <path>                          (default: ./targeted_green_missing_18)
  --target-green-balanced-exact <float>           (default: 0.17)
  --full-guardrail-min-exact <float>              (default: 0.30)
  --empty-pred-max-rate <float>                   (default: 0.05)
  --train-balanced-per-province <int>             (default: 200)
  --train-full-capped-per-province <int>          (default: 900)
  --train-full-capped-max-major-ratio <float>     (default: 0.45)
  --val-balanced-per-province <int>               (default: 40)
  --test-balanced-per-province <int>              (default: 40)
  --stagea-epochs <int>                           (default: 10)
  --stagea-lr <float>                             (default: 0.000003)
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

RUN_TAG="green_official_targeted_v1_${DATE_TAG}"
BASE_MODEL="${ROOT_DIR}/weights_red_stage3/Final_LPRNet_model.pth"
GREEN_ROOT="${ROOT_DIR}/CCPD2020/ccpd_green"
TARGETED_ROOT="${ROOT_DIR}/targeted_green_missing_18"

TARGET_GREEN_BALANCED_EXACT="0.17"
FULL_GUARDRAIL_MIN_EXACT="0.30"
EMPTY_PRED_MAX_RATE="0.05"

TRAIN_CAP_PER_PROVINCE="2200"
TRAIN_MAX_MAJOR_RATIO="0.90"
LEGACY_VAL_BALANCED_PER_PROVINCE="120"
LEGACY_TEST_BALANCED_PER_PROVINCE="260"

TRAIN_BALANCED_PER_PROVINCE="200"
TRAIN_FULL_CAPPED_PER_PROVINCE="900"
TRAIN_FULL_CAPPED_MAX_MAJOR_RATIO="0.45"
VAL_BALANCED_PER_PROVINCE="40"
TEST_BALANCED_PER_PROVINCE="40"

STAGEA_EPOCHS="10"
STAGEA_LR="0.000003"
STAGEA_LR_SCHEDULE=(3 6 8)
STAGEA_FIRST_CHAR_AUX="0.03"
STAGEA_PROVINCE_BALANCE_MODE="inv_sqrt"
STAGEA_PROVINCE_BALANCE_CLIP="2.0"

STRATA_BALANCE_MODE="none"
STRATA_BALANCE_CLIP="0.0"
ADJ_REPEAT_SAMPLE_WEIGHT="1.10"

USE_CUDA="${USE_CUDA:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SELECTION_EVAL_SAMPLES="5000"
SEED="20260320"

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
    --targeted-root) TARGETED_ROOT="$2"; shift 2 ;;
    --target-green-balanced-exact) TARGET_GREEN_BALANCED_EXACT="$2"; shift 2 ;;
    --full-guardrail-min-exact) FULL_GUARDRAIL_MIN_EXACT="$2"; shift 2 ;;
    --empty-pred-max-rate) EMPTY_PRED_MAX_RATE="$2"; shift 2 ;;
    --train-balanced-per-province) TRAIN_BALANCED_PER_PROVINCE="$2"; shift 2 ;;
    --train-full-capped-per-province) TRAIN_FULL_CAPPED_PER_PROVINCE="$2"; shift 2 ;;
    --train-full-capped-max-major-ratio) TRAIN_FULL_CAPPED_MAX_MAJOR_RATIO="$2"; shift 2 ;;
    --val-balanced-per-province) VAL_BALANCED_PER_PROVINCE="$2"; shift 2 ;;
    --test-balanced-per-province) TEST_BALANCED_PER_PROVINCE="$2"; shift 2 ;;
    --stagea-epochs) STAGEA_EPOCHS="$2"; shift 2 ;;
    --stagea-lr) STAGEA_LR="$2"; shift 2 ;;
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
GREEN_MERGED_LABEL_DIR="${OUT_DIR}/green_merged_labels"
WEIGHTS_A="${OUT_DIR}/weights_stageA"
mkdir -p "${OUT_DIR}" "${LABELS_DIR}" "${GREEN_LABEL_SRC_DIR}" "${GREEN_MERGED_LABEL_DIR}" "${WEIGHTS_A}"

if [[ ! -x "${ENV_PY}" ]]; then
  echo "Training python not found: ${ENV_PY}" >&2
  exit 1
fi
if [[ ! -d "${GREEN_ROOT}" ]]; then
  echo "Green dataset root not found: ${GREEN_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${TARGETED_ROOT}" ]]; then
  echo "Targeted green dataset root not found: ${TARGETED_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${TARGETED_ROOT}/train.txt" || ! -f "${TARGETED_ROOT}/val.txt" ]]; then
  echo "Targeted green label files not found under: ${TARGETED_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Base model not found: ${BASE_MODEL}" >&2
  exit 1
fi

IMG_ROOTS="${GREEN_ROOT},${ROOT_DIR}"

echo "[Config] run_tag=${RUN_TAG} out_dir=${OUT_DIR}"
echo "[Config] base_model=${BASE_MODEL}"
echo "[Config] green_root=${GREEN_ROOT}"
echo "[Config] targeted_root=${TARGETED_ROOT}"
echo "[Config] img_roots=${IMG_ROOTS}"
echo "[Config] target_green_balanced_exact=${TARGET_GREEN_BALANCED_EXACT}"
echo "[Config] full_guardrail_min_exact=${FULL_GUARDRAIL_MIN_EXACT}"
echo "[Config] empty_pred_max_rate=${EMPTY_PRED_MAX_RATE}"
echo "[Config] train_balanced_per_province=${TRAIN_BALANCED_PER_PROVINCE}"
echo "[Config] train_full_capped_per_province=${TRAIN_FULL_CAPPED_PER_PROVINCE}"
echo "[Config] train_full_capped_max_major_ratio=${TRAIN_FULL_CAPPED_MAX_MAJOR_RATIO}"
echo "[Config] eval_decode_mode=${EVAL_DECODE_MODE} beam=${BEAM_SIZE}/${BEAM_TOPK}"
echo "[Config] arm_perspective_ocr=ch:${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER} crop:${ARM_PERSPECTIVE_OCR_CROP_MODE} resize:${ARM_PERSPECTIVE_OCR_RESIZE_MODE}/${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL} preproc:${ARM_PERSPECTIVE_OCR_PREPROC} min_occ:${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO} quad_pad:${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"

step=1

COMMON_EVAL_ARGS=(
  --seed "${SEED}"
  --test_img_dirs "${IMG_ROOTS}"
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

echo "[${step}] Build CCPD2020 green labels"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/prepare_ccpd_green_labels.py" \
  --dataset-root "${GREEN_ROOT}" \
  --output-dir "${GREEN_LABEL_SRC_DIR}" \
  --strict | tee "${OUT_DIR}/prepare_green_labels.stdout.txt"

echo "[${step}] Merge CCPD2020 + targeted_green_missing_18 labels"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/prepare_green_merged_labels.py" \
  --ccpd-train "${GREEN_LABEL_SRC_DIR}/train_labels.txt" \
  --ccpd-val "${GREEN_LABEL_SRC_DIR}/val_labels.txt" \
  --ccpd-test "${GREEN_LABEL_SRC_DIR}/test_labels.txt" \
  --targeted-train "${TARGETED_ROOT}/train.txt" \
  --targeted-val "${TARGETED_ROOT}/val.txt" \
  --output-dir "${GREEN_MERGED_LABEL_DIR}" \
  --image-roots "${IMG_ROOTS}" \
  --strict | tee "${OUT_DIR}/merge_manifest.stdout.json"

echo "[${step}] Build strict green splits from merged labels"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/prepare_green_specialist_splits.py" \
  --green-train "${GREEN_MERGED_LABEL_DIR}/train_labels.txt" \
  --green-val "${GREEN_MERGED_LABEL_DIR}/val_labels.txt" \
  --green-test "${GREEN_MERGED_LABEL_DIR}/test_labels.txt" \
  --output-dir "${LABELS_DIR}" \
  --train-cap-per-province "${TRAIN_CAP_PER_PROVINCE}" \
  --train-max-major-ratio "${TRAIN_MAX_MAJOR_RATIO}" \
  --val-balanced-per-province "${LEGACY_VAL_BALANCED_PER_PROVINCE}" \
  --test-balanced-per-province "${LEGACY_TEST_BALANCED_PER_PROVINCE}" \
  --train-balanced-v2-per-province "${TRAIN_BALANCED_PER_PROVINCE}" \
  --train-full-capped-v2-per-province "${TRAIN_FULL_CAPPED_PER_PROVINCE}" \
  --train-full-capped-v2-max-major-ratio "${TRAIN_FULL_CAPPED_MAX_MAJOR_RATIO}" \
  --val-balanced-v2-per-province "${VAL_BALANCED_PER_PROVINCE}" \
  --test-balanced-v2-per-province "${TEST_BALANCED_PER_PROVINCE}" \
  --seed "${SEED}" | tee "${OUT_DIR}/split_manifest.stdout.json"

TEST_FULL="${LABELS_DIR}/test_green_full_labels.txt"
TEST_BAL="${LABELS_DIR}/test_green_balanced_v2_labels.txt"
VAL_BAL="${LABELS_DIR}/val_green_balanced_v2_labels.txt"
TRAIN_STAGEC="${LABELS_DIR}/train_green_stagec_v2_labels.txt"

TRAIN_COMMON_ARGS=(
  --seed "${SEED}"
  --train_img_dirs "${IMG_ROOTS}"
  --test_img_dirs "${IMG_ROOTS}"
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
  --second_char_aux_weight 0.0
  --ne_type_aux_weight 0.0
  --selection_strategy balanced_recovery
  --selection_decode_mode "${EVAL_DECODE_MODE}"
  --selection_beam_size "${BEAM_SIZE}"
  --selection_beam_topk "${BEAM_TOPK}"
  --selection_proxy_eval_samples "${SELECTION_EVAL_SAMPLES}"
  --early_stop_patience 2
  --early_stop_regression_patience 2
  --early_stop_regression_pp 0.2
  --early_stop_start_epoch 4
  --first_char_time_steps 6
  --train_plate_box_aug_mode none
  --train_plate_box_aug_prob 0.0
  --save_interval 2000
  --train_batch_size 64
  --test_batch_size 120
  --num_workers "${NUM_WORKERS}"
  --cuda "${USE_CUDA}"
)

echo "[${step}] Baseline metrics from official weights"
step=$((step + 1))
run_eval "${TEST_BAL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_balanced_test_metrics.json"
run_eval "${TEST_FULL}" "${BASE_MODEL}" "${OUT_DIR}/baseline_green_full_test_metrics.json"

echo "[${step}] Stage A training (green-only merged set)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_STAGEC}" \
  --test_txt_file "${VAL_BAL}" \
  --province_balance_mode "${STAGEA_PROVINCE_BALANCE_MODE}" \
  --province_balance_clip "${STAGEA_PROVINCE_BALANCE_CLIP}" \
  --first_char_aux_weight "${STAGEA_FIRST_CHAR_AUX}" \
  --pretrained_model "${BASE_MODEL}" \
  --learning_rate "${STAGEA_LR}" \
  --lr_schedule "${STAGEA_LR_SCHEDULE[@]}" \
  --save_folder "${WEIGHTS_A}/" \
  --max_epoch "${STAGEA_EPOCHS}" | tee "${OUT_DIR}/train_stageA.log"

CANDIDATE_MODEL="${WEIGHTS_A}/Final_LPRNet_model.pth"
if [[ ! -f "${CANDIDATE_MODEL}" ]]; then
  echo "Stage A final model not found: ${CANDIDATE_MODEL}" >&2
  exit 1
fi

echo "[${step}] Candidate metrics"
step=$((step + 1))
run_eval "${TEST_BAL}" "${CANDIDATE_MODEL}" "${OUT_DIR}/candidate_green_balanced_metrics.json"
run_eval "${TEST_FULL}" "${CANDIDATE_MODEL}" "${OUT_DIR}/candidate_green_full_metrics.json"

echo "[${step}] Build acceptance + scorecard"
step=$((step + 1))
"${ENV_PY}" - <<PY | tee "${OUT_DIR}/scorecard.stdout.json"
import json
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(${OUT_DIR@Q})
baseline_balanced = json.loads((run_dir / "baseline_green_balanced_test_metrics.json").read_text(encoding="utf-8"))
baseline_full = json.loads((run_dir / "baseline_green_full_test_metrics.json").read_text(encoding="utf-8"))
candidate_balanced = json.loads((run_dir / "candidate_green_balanced_metrics.json").read_text(encoding="utf-8"))
candidate_full = json.loads((run_dir / "candidate_green_full_metrics.json").read_text(encoding="utf-8"))

target_balanced_exact = float(${TARGET_GREEN_BALANCED_EXACT@Q})
full_guardrail = float(${FULL_GUARDRAIL_MIN_EXACT@Q})
empty_pred_max = float(${EMPTY_PRED_MAX_RATE@Q})

baseline_exact = float(baseline_balanced.get("exact_plate_acc", 0.0))
baseline_full_exact = float(baseline_full.get("exact_plate_acc", 0.0))
baseline_empty = float(baseline_balanced.get("empty_pred_rate", 0.0))

candidate_exact = float(candidate_balanced.get("exact_plate_acc", 0.0))
candidate_full_exact = float(candidate_full.get("exact_plate_acc", 0.0))
candidate_empty = float(candidate_balanced.get("empty_pred_rate", 0.0))

candidate_ok = (
    candidate_full_exact >= full_guardrail and
    candidate_empty <= empty_pred_max and
    candidate_exact >= baseline_exact
)

if candidate_ok:
    adopted_model = ${CANDIDATE_MODEL@Q}
    adopted_source = "candidate_stageA"
    adopted_balanced = candidate_balanced
    adopted_full = candidate_full
else:
    adopted_model = ${BASE_MODEL@Q}
    adopted_source = "baseline_fallback"
    adopted_balanced = baseline_balanced
    adopted_full = baseline_full

adopted_exact = float(adopted_balanced.get("exact_plate_acc", 0.0))
adopted_full_exact = float(adopted_full.get("exact_plate_acc", 0.0))
adopted_empty = float(adopted_balanced.get("empty_pred_rate", 0.0))
passed = (
    adopted_exact >= target_balanced_exact and
    adopted_full_exact >= full_guardrail and
    adopted_empty <= empty_pred_max
)

acceptance = {
    "target_green_balanced_exact": target_balanced_exact,
    "full_guardrail_min_exact": full_guardrail,
    "empty_pred_max_rate": empty_pred_max,
    "baseline_green_balanced_exact": baseline_exact,
    "baseline_green_full_exact": baseline_full_exact,
    "baseline_green_balanced_empty_pred_rate": baseline_empty,
    "candidate_green_balanced_exact": candidate_exact,
    "candidate_green_full_exact": candidate_full_exact,
    "candidate_green_balanced_empty_pred_rate": candidate_empty,
    "candidate_adopted": candidate_ok,
    "adopted_source": adopted_source,
    "adopted_green_balanced_exact": adopted_exact,
    "adopted_green_balanced_macro_first": float(adopted_balanced.get("province_macro_first_char_acc", 0.0)),
    "adopted_green_balanced_macro_exact": float(adopted_balanced.get("province_macro_exact_acc", 0.0)),
    "adopted_green_full_exact": adopted_full_exact,
    "adopted_green_balanced_empty_pred_rate": adopted_empty,
    "pass_balanced_exact": adopted_exact >= target_balanced_exact,
    "pass_full_guardrail": adopted_full_exact >= full_guardrail,
    "pass_empty_pred_rate": adopted_empty <= empty_pred_max,
    "passed": passed,
}

(run_dir / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

scorecard = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "base_model": ${BASE_MODEL@Q},
    "candidate_model": ${CANDIDATE_MODEL@Q},
    "adopted_model": adopted_model,
    "adopted_source": adopted_source,
    "eval_decode_mode": ${EVAL_DECODE_MODE@Q},
    "beam_size": int(${BEAM_SIZE@Q}),
    "beam_topk": int(${BEAM_TOPK@Q}),
    "acceptance": acceptance,
}
(run_dir / "scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(scorecard, ensure_ascii=False, indent=2))
PY

echo "[${step}] Export key paths"
step=$((step + 1))
echo "run_dir=${OUT_DIR}"
echo "base_model=${BASE_MODEL}"
echo "candidate_model=${CANDIDATE_MODEL}"
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
