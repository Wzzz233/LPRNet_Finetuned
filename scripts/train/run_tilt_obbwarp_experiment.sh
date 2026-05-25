#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_tilt_obbwarp_experiment.sh [run_tag] [options]

Options:
  --run-tag <tag>
  --base-model <path>
  --stagea-hard-ratio <float>      (default: 0.75)
  --stageb-hard-ratio <float>      (default: 0.90)
  --stagec-hard-ratio <float>      (default: 0.95)
  --stagea-first-char-aux <float>  (default: 0.25)
  --stageb-first-char-aux <float>  (default: 0.25)
  --stagec-first-char-aux <float>  (default: 0.15)
  --stagea-lr <float>              (default: 0.00003)
  --stageb-lr <float>              (default: 0.00001)
  --stagec-lr <float>              (default: 0.000005)
  --stagea-epochs <int>            (default: 8)
  --stageb-epochs <int>            (default: 8)
  --stagec-epochs <int>            (default: 6)
  --enable-jitter-refine <bool>    (default: true)
  --jitter-prob <float>            (default: 0.35)
  --hardcase-train-txt <path>      (merged into hard pool and replayed as pseudo-anchor train set)
  --hardcase-val-txt <path>        (replayed as pseudo-anchor val set and used for no-leak checks)
  --checkpoint-reeval-mode <mode>  (default: val, choices: none|val|full)
  --normal-drop-max-pp <float>     (default: 0.0)
  --hard-rel-gain-min <float>      (default: 0.10)
  --seed <int>                     (default: 20260318)
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

RUN_TAG="tilt_ocr_obbwarp_v3"
USE_CUDA="${USE_CUDA:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_MODEL_OVERRIDE=""

# Locked OCR preprocessing profile for the adjusted perspective path used on board.
ARM_PERSPECTIVE_OCR_CHANNEL_ORDER="bgr"
ARM_PERSPECTIVE_OCR_CROP_MODE="obb_warp"
ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO="0.0"
ARM_PERSPECTIVE_OCR_RESIZE_MODE="letterbox"
ARM_PERSPECTIVE_OCR_RESIZE_KERNEL="nn"
ARM_PERSPECTIVE_OCR_PREPROC="none"
ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO="0.90"

STAGEA_HARD_RATIO="0.75"
STAGEB_HARD_RATIO="0.90"
STAGEC_HARD_RATIO="0.95"

STAGEA_EPOCHS="8"
STAGEB_EPOCHS="8"
STAGEC_EPOCHS="6"

STAGEA_LR="0.00003"
STAGEB_LR="0.00001"
STAGEC_LR="0.000005"

STAGEA_LR_SCHEDULE=(3 6)
STAGEB_LR_SCHEDULE=(3 6)
STAGEC_LR_SCHEDULE=(2 4)

STAGEA_FIRST_CHAR_AUX="0.25"
STAGEB_FIRST_CHAR_AUX="0.25"
STAGEC_FIRST_CHAR_AUX="0.15"

ENABLE_JITTER_REFINE="true"
JITTER_PROB="0.35"

HARDCASE_TRAIN_TXT=""
HARDCASE_VAL_TXT=""
CHECKPOINT_REEVAL_MODE="val"

NORMAL_DROP_MAX_PP="0.0"
HARD_REL_GAIN_MIN="0.10"
SEED="20260318"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  RUN_TAG="$1"
  shift
fi

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
      BASE_MODEL_OVERRIDE="$2"
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
    --enable-jitter-refine)
      ENABLE_JITTER_REFINE="$2"
      shift 2
      ;;
    --jitter-prob)
      JITTER_PROB="$2"
      shift 2
      ;;
    --hardcase-train-txt)
      HARDCASE_TRAIN_TXT="$2"
      shift 2
      ;;
    --hardcase-val-txt)
      HARDCASE_VAL_TXT="$2"
      shift 2
      ;;
    --checkpoint-reeval-mode)
      CHECKPOINT_REEVAL_MODE="$2"
      shift 2
      ;;
    --normal-drop-max-pp)
      NORMAL_DROP_MAX_PP="$2"
      shift 2
      ;;
    --hard-rel-gain-min)
      HARD_REL_GAIN_MIN="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ENABLE_JITTER_REFINE="$(str_to_bool "${ENABLE_JITTER_REFINE}")"
case "${CHECKPOINT_REEVAL_MODE}" in
  none|val|full) ;;
  *)
    echo "invalid checkpoint reeval mode: ${CHECKPOINT_REEVAL_MODE}" >&2
    exit 1
    ;;
esac

OUT_DIR="${ROOT_DIR}/experiments/${RUN_TAG}"
WEIGHTS_A="${OUT_DIR}/weights_stageA"
WEIGHTS_B="${OUT_DIR}/weights_stageB"
WEIGHTS_C="${OUT_DIR}/weights_stageC"
LABELS_DIR="${OUT_DIR}/labels"
mkdir -p "${OUT_DIR}" "${WEIGHTS_A}" "${WEIGHTS_B}" "${WEIGHTS_C}" "${LABELS_DIR}"

HARD_TRAIN="${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/train_labels.txt"
HARD_VAL="${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/val_labels.txt"
HARD_TEST="${ROOT_DIR}/prepared_labels/ccpd2019_hard_tilt/test_labels.txt"
NORMAL_TRAIN="${ROOT_DIR}/prepared_labels/ccpd2019/train_labels.txt"
NORMAL_VAL="${ROOT_DIR}/prepared_labels/ccpd2019/val_labels.txt"
NORMAL_TEST="${ROOT_DIR}/prepared_labels/ccpd2019/test_labels.txt"

BASELINE_MODEL="${ROOT_DIR}/experiments/first_char_guard_v1/weights/Final_LPRNet_model.pth"
if [[ -n "${BASE_MODEL_OVERRIDE}" ]]; then
  BASELINE_MODEL="${BASE_MODEL_OVERRIDE}"
fi
BOARD_TXT="${ROOT_DIR}/board_anchor_labels.txt"

if [[ ! -x "${ENV_PY}" ]]; then
  echo "Training python not found: ${ENV_PY}" >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found; RKNN conversion environment cannot be used." >&2
  exit 1
fi
if [[ ! -f "${BASELINE_MODEL}" ]]; then
  echo "Baseline model not found: ${BASELINE_MODEL}" >&2
  exit 1
fi
if [[ -n "${HARDCASE_TRAIN_TXT}" && ! -f "${HARDCASE_TRAIN_TXT}" ]]; then
  echo "hardcase train txt not found: ${HARDCASE_TRAIN_TXT}" >&2
  exit 1
fi
if [[ -n "${HARDCASE_VAL_TXT}" && ! -f "${HARDCASE_VAL_TXT}" ]]; then
  echo "hardcase val txt not found: ${HARDCASE_VAL_TXT}" >&2
  exit 1
fi

if [[ "${ENABLE_JITTER_REFINE}" == "true" ]]; then
  AUG_MODE="jitter_refine"
  AUG_PROB="${JITTER_PROB}"
else
  AUG_MODE="none"
  AUG_PROB="0.0"
fi

echo "[Config] run_tag=${RUN_TAG} out_dir=${OUT_DIR}"
echo "[Config] stage_ratios=A:${STAGEA_HARD_RATIO} B:${STAGEB_HARD_RATIO} C:${STAGEC_HARD_RATIO}"
echo "[Config] stage_epochs=A:${STAGEA_EPOCHS} B:${STAGEB_EPOCHS} C:${STAGEC_EPOCHS}"
echo "[Config] stage_first_char_aux=A:${STAGEA_FIRST_CHAR_AUX} B:${STAGEB_FIRST_CHAR_AUX} C:${STAGEC_FIRST_CHAR_AUX}"
echo "[Config] stage_lr=A:${STAGEA_LR} B:${STAGEB_LR} C:${STAGEC_LR}"
echo "[Config] jitter=${AUG_MODE} prob=${AUG_PROB}"
echo "[Config] hardcase_train_txt=${HARDCASE_TRAIN_TXT:-<none>} hardcase_val_txt=${HARDCASE_VAL_TXT:-<none>}"
echo "[Config] checkpoint_reeval_mode=${CHECKPOINT_REEVAL_MODE}"
echo "[Config] arm_perspective_ocr=ch:${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER} crop:${ARM_PERSPECTIVE_OCR_CROP_MODE} resize:${ARM_PERSPECTIVE_OCR_RESIZE_MODE}/${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL} preproc:${ARM_PERSPECTIVE_OCR_PREPROC} min_occ:${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO} quad_pad:${ARM_PERSPECTIVE_OCR_QUAD_PAD_RATIO}"

step=1

echo "[${step}] Build mixed labels (strict no-leak train/val/test)"
step=$((step + 1))
MIX_ARGS=(
  --hard-train "${HARD_TRAIN}"
  --hard-val "${HARD_VAL}"
  --hard-test "${HARD_TEST}"
  --normal-train "${NORMAL_TRAIN}"
  --output-dir "${LABELS_DIR}"
  --stagea-hard-ratio "${STAGEA_HARD_RATIO}"
  --stageb-hard-ratio "${STAGEB_HARD_RATIO}"
  --stagec-hard-ratio "${STAGEC_HARD_RATIO}"
  --seed "${SEED}"
)
if [[ -n "${HARDCASE_TRAIN_TXT}" ]]; then
  MIX_ARGS+=(--hardcase-train "${HARDCASE_TRAIN_TXT}")
fi
"${ENV_PY}" "${ROOT_DIR}/prepare_tilt_mixed_labels.py" "${MIX_ARGS[@]}" | tee "${OUT_DIR}/mix_manifest.stdout.json"

TRAIN_A="${LABELS_DIR}/train_mix_stageA_labels.txt"
TRAIN_B="${LABELS_DIR}/train_mix_stageB_labels.txt"
TRAIN_C="${LABELS_DIR}/train_mix_stageC_labels.txt"
VAL_HARD="${LABELS_DIR}/val_hard_labels.txt"
TEST_HARD="${LABELS_DIR}/test_hard_labels.txt"

echo "[${step}] Assert no leakage on stageA/B/C splits"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_A}" \
  --val "${VAL_HARD}" \
  --test "${TEST_HARD}" \
  --fail-on-overlap \
  --out-json "${OUT_DIR}/split_stageA_stats.json" | tee "${OUT_DIR}/split_stageA_stats.stdout.json"

"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_B}" \
  --val "${VAL_HARD}" \
  --test "${TEST_HARD}" \
  --fail-on-overlap \
  --out-json "${OUT_DIR}/split_stageB_stats.json" | tee "${OUT_DIR}/split_stageB_stats.stdout.json"

"${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
  --train "${TRAIN_C}" \
  --val "${VAL_HARD}" \
  --test "${TEST_HARD}" \
  --fail-on-overlap \
  --out-json "${OUT_DIR}/split_stageC_stats.json" | tee "${OUT_DIR}/split_stageC_stats.stdout.json"

if [[ -n "${HARDCASE_VAL_TXT}" ]]; then
  "${ENV_PY}" "${ROOT_DIR}/analyze_ccpd_splits.py" \
    --train "${TRAIN_C}" \
    --val "${HARDCASE_VAL_TXT}" \
    --test "${TEST_HARD}" \
    --fail-on-overlap \
    --out-json "${OUT_DIR}/split_hardcase_val_stats.json" | tee "${OUT_DIR}/split_hardcase_val_stats.stdout.json"
fi

COMMON_EVAL_ARGS=(
  --test_img_dirs "${ROOT_DIR}/CCPD2019"
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

echo "[${step}] Baseline metrics on hard/normal test"
step=$((step + 1))
run_eval "${TEST_HARD}" "${BASELINE_MODEL}" "${OUT_DIR}/baseline_hard_test_metrics.json"
run_eval "${NORMAL_TEST}" "${BASELINE_MODEL}" "${OUT_DIR}/baseline_normal_test_metrics.json"

TRAIN_COMMON_ARGS=(
  --train_img_dirs "${ROOT_DIR}/CCPD2019"
  --test_img_dirs "${ROOT_DIR}/CCPD2019"
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
  --train_plate_box_aug_mode "${AUG_MODE}"
  --train_plate_box_aug_prob "${AUG_PROB}"
  --save_interval 2000
  --train_batch_size 64
  --test_batch_size 120
  --num_workers "${NUM_WORKERS}"
  --cuda "${USE_CUDA}"
)
if [[ -n "${HARDCASE_TRAIN_TXT}" ]]; then
  # Inject hard-case replay as pseudo-anchor train set so these samples are up-weighted.
  TRAIN_COMMON_ARGS+=(--pseudo_anchor_train_txt_file "${HARDCASE_TRAIN_TXT}")
fi
if [[ -n "${HARDCASE_VAL_TXT}" ]]; then
  TRAIN_COMMON_ARGS+=(--pseudo_anchor_val_txt_file "${HARDCASE_VAL_TXT}")
fi

echo "[${step}] Stage A training"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_A}" \
  --test_txt_file "${VAL_HARD}" \
  --first_char_aux_weight "${STAGEA_FIRST_CHAR_AUX}" \
  --pretrained_model "${BASELINE_MODEL}" \
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
  --test_txt_file "${VAL_HARD}" \
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
  --test_txt_file "${VAL_HARD}" \
  --first_char_aux_weight "${STAGEC_FIRST_CHAR_AUX}" \
  --pretrained_model "${MODEL_B}" \
  --learning_rate "${STAGEC_LR}" \
  --lr_schedule "${STAGEC_LR_SCHEDULE[@]}" \
  --save_folder "${WEIGHTS_C}/" \
  --max_epoch "${STAGEC_EPOCHS}" | tee "${OUT_DIR}/train_stageC.log"

FINAL_MODEL="${WEIGHTS_C}/Final_LPRNet_model.pth"
FINAL_WEIGHTS_DIR="${WEIGHTS_C}"
if [[ ! -f "${FINAL_MODEL}" ]]; then
  echo "Stage C final model not found: ${FINAL_MODEL}" >&2
  exit 1
fi

echo "[${step}] Evaluate hard val/test"
step=$((step + 1))
run_eval "${VAL_HARD}" "${FINAL_MODEL}" "${OUT_DIR}/val_metrics_hard.json"
run_eval "${TEST_HARD}" "${FINAL_MODEL}" "${OUT_DIR}/test_metrics_hard.json"

echo "[${step}] Evaluate normal val/test"
step=$((step + 1))
run_eval "${NORMAL_VAL}" "${FINAL_MODEL}" "${OUT_DIR}/val_metrics_normal.json"
run_eval "${NORMAL_TEST}" "${FINAL_MODEL}" "${OUT_DIR}/test_metrics_normal.json"

if [[ "${CHECKPOINT_REEVAL_MODE}" != "none" ]]; then
  echo "[${step}] Re-evaluate saved checkpoints"
  step=$((step + 1))
  REEVAL_ARGS=(
    --run-dir "${OUT_DIR}"
    --test-img-dirs "${ROOT_DIR}/CCPD2019"
    --data-mode ccpd_board
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
    --hard-val "${VAL_HARD}"
    --normal-val "${NORMAL_VAL}"
    --ranking-split hard_val
    --normal-guard-split normal_val
    --out-json "${OUT_DIR}/checkpoint_scoreboard.json"
  )
  if [[ "${CHECKPOINT_REEVAL_MODE}" == "full" ]]; then
    REEVAL_ARGS+=(--hard-test "${TEST_HARD}" --normal-test "${NORMAL_TEST}")
  fi
  "${ENV_PY}" "${ROOT_DIR}/reeval_lpr_checkpoints.py" "${REEVAL_ARGS[@]}" | tee "${OUT_DIR}/checkpoint_scoreboard.stdout.json"
fi

if [[ -f "${BOARD_TXT}" ]]; then
  echo "[${step}] Evaluate board anchors"
  step=$((step + 1))
  "${ENV_PY}" "${ROOT_DIR}/eval_board_anchors.py" \
    --weights "${FINAL_MODEL}" \
    --img_dirs "${ROOT_DIR}" \
    --txt_file "${BOARD_TXT}" \
    --first_char_time_steps 6 \
    --out_json "${OUT_DIR}/board_anchor_metrics.json" | tee "${OUT_DIR}/board_anchor_metrics.stdout.json"
fi

echo "[${step}] Acceptance gate"
step=$((step + 1))
set +e
"${ENV_PY}" "${ROOT_DIR}/evaluate_lpr_acceptance.py" \
  --baseline-hard-test "${OUT_DIR}/baseline_hard_test_metrics.json" \
  --new-hard-test "${OUT_DIR}/test_metrics_hard.json" \
  --baseline-normal-test "${OUT_DIR}/baseline_normal_test_metrics.json" \
  --new-normal-test "${OUT_DIR}/test_metrics_normal.json" \
  --hard-rel-gain-min "${HARD_REL_GAIN_MIN}" \
  --normal-drop-max-pp "${NORMAL_DROP_MAX_PP}" \
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
baseline_hard = json.loads((run_dir / "baseline_hard_test_metrics.json").read_text(encoding="utf-8"))
baseline_normal = json.loads((run_dir / "baseline_normal_test_metrics.json").read_text(encoding="utf-8"))
new_hard = json.loads((run_dir / "test_metrics_hard.json").read_text(encoding="utf-8"))
new_normal = json.loads((run_dir / "test_metrics_normal.json").read_text(encoding="utf-8"))
acceptance = json.loads((run_dir / "acceptance.json").read_text(encoding="utf-8"))

hard_exact = float(new_hard.get("exact_plate_acc", 0.0))
normal_exact = float(new_normal.get("exact_plate_acc", 0.0))
baseline_normal_exact = float(baseline_normal.get("exact_plate_acc", 0.0))

scorecard = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "targets": {
        "hard_exact_primary": 0.50,
        "hard_exact_stretch": 0.80,
        "normal_exact_floor": baseline_normal_exact,
    },
    "results": {
        "hard_exact": hard_exact,
        "hard_char_acc": float(new_hard.get("char_acc", 0.0)),
        "normal_exact": normal_exact,
        "normal_char_acc": float(new_normal.get("char_acc", 0.0)),
        "hard_sample_count": int(new_hard.get("sample_count", 0)),
        "normal_sample_count": int(new_normal.get("sample_count", 0)),
        "baseline_hard_exact": float(baseline_hard.get("exact_plate_acc", 0.0)),
        "baseline_normal_exact": baseline_normal_exact,
    },
    "gates": {
        "acceptance_passed": bool(acceptance.get("passed", False)),
        "normal_no_regression": normal_exact >= baseline_normal_exact,
        "hard_ge_50": hard_exact >= 0.50,
        "hard_ge_80": hard_exact >= 0.80,
    },
}

if scorecard["gates"]["hard_ge_80"] and scorecard["gates"]["normal_no_regression"]:
    decision = "target_80_reached"
elif scorecard["gates"]["hard_ge_50"] and scorecard["gates"]["normal_no_regression"]:
    decision = "target_50_reached_continue_to_80"
else:
    decision = "continue_finetune"

scorecard["decision"] = decision
(run_dir / "scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(scorecard, ensure_ascii=False, indent=2))
PY

echo "[${step}] Write report"
step=$((step + 1))
TOTAL_EPOCHS=$((STAGEA_EPOCHS + STAGEB_EPOCHS + STAGEC_EPOCHS))
REPORT_ARGS=(
  --experiment_name "Tilt OBBWarp V3"
  --run_dir "${OUT_DIR}"
  --train_txt "${TRAIN_C}"
  --val_txt "${VAL_HARD}"
  --test_txt "${TEST_HARD}"
  --board_anchor_txt "${BOARD_TXT}"
  --data_mode ccpd_board
  --ocr_channel_order "${ARM_PERSPECTIVE_OCR_CHANNEL_ORDER}"
  --ocr_crop_mode "${ARM_PERSPECTIVE_OCR_CROP_MODE}"
  --ocr_resize_mode "${ARM_PERSPECTIVE_OCR_RESIZE_MODE}"
  --ocr_resize_kernel "${ARM_PERSPECTIVE_OCR_RESIZE_KERNEL}"
  --ocr_preproc "${ARM_PERSPECTIVE_OCR_PREPROC}"
  --ocr_min_occ_ratio "${ARM_PERSPECTIVE_OCR_MIN_OCC_RATIO}"
  --pretrained_model "${BASELINE_MODEL}"
  --learning_rate "${STAGEA_LR}"
  --lr_schedule "A:${STAGEA_LR_SCHEDULE[*]} B:${STAGEB_LR_SCHEDULE[*]} C:${STAGEC_LR_SCHEDULE[*]}"
  --max_epoch "${TOTAL_EPOCHS}"
  --train_batch_size 64
  --test_batch_size 120
  --province_balance_mode inv_sqrt
  --first_char_aux_weight "${STAGEA_FIRST_CHAR_AUX}"
  --first_char_time_steps 6
  --selection_proxy_eval_samples 5000
  --report_path "${OUT_DIR}/EXPERIMENT_REPORT.md"
)
if [[ -n "${HARDCASE_TRAIN_TXT}" ]]; then
  REPORT_ARGS+=(--pseudo_anchor_train_txt "${HARDCASE_TRAIN_TXT}")
fi
if [[ -n "${HARDCASE_VAL_TXT}" ]]; then
  REPORT_ARGS+=(--pseudo_anchor_val_txt "${HARDCASE_VAL_TXT}")
fi
"${ENV_PY}" "${ROOT_DIR}/generate_experiment_report.py" "${REPORT_ARGS[@]}"

echo "[${step}] Export ONNX"
step=$((step + 1))
ONNX_PATH="${FINAL_WEIGHTS_DIR}/LPRNet_stage3_rk3568_fp16.onnx"
RKNN_PATH="${FINAL_WEIGHTS_DIR}/LPRNet_stage3_rk3568_fp16.rknn"
"${ENV_PY}" "${ROOT_DIR}/export_onnx_rknn_compatible.py" \
  --weights "${FINAL_MODEL}" \
  --output "${ONNX_PATH}"

echo "[${step}] Build RKNN"
step=$((step + 1))
env CONDA_NO_PLUGINS=true conda run -p "${RKNN_ENV_PREFIX}" python "${ROOT_DIR}/custom_rknn_convert.py" \
  "${ONNX_PATH}" \
  --target-platform rk3568 \
  --dtype fp \
  --output "${RKNN_PATH}" \
  --input-color-order bgr \
  --model-color-order bgr

echo "[${step}] Export key paths"
step=$((step + 1))
echo "run_dir=${OUT_DIR}"
echo "final_model=${FINAL_MODEL}"
echo "onnx_model=${ONNX_PATH}"
echo "rknn_model=${RKNN_PATH}"
echo "rknn_env_prefix=${RKNN_ENV_PREFIX}"
echo "acceptance_json=${OUT_DIR}/acceptance.json"
echo "scorecard_json=${OUT_DIR}/scorecard.json"

echo "Done"
if [[ ${ACCEPT_RC} -ne 0 ]]; then
  echo "Acceptance gate failed. Check ${OUT_DIR}/acceptance.stdout.json" >&2
  exit ${ACCEPT_RC}
fi
