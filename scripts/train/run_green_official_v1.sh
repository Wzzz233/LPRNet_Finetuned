#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_green_official_v1.sh [options]

Options:
  --run-tag <tag>
  --base-model <path>                    (default: ./weights_red_stage3/Final_LPRNet_model.pth)
  --green-root <path>                    (default: ./CCPD2020/ccpd_green)
  --target-green-full-exact <float>      (default: 0.50)
  --enable-hard-round <true|false>       (default: true)
  --train-cap-per-province <int>         (default: 2200)
  --train-max-major-ratio <float>        (default: 0.90)
  --val-balanced-per-province <int>      (default: 120)
  --test-balanced-per-province <int>     (default: 260)
  --stagea-epochs <int>                  (default: 18)
  --stageb-epochs <int>                  (default: 14)
  --stagec-epochs <int>                  (default: 10)
  --stagea-lr <float>                    (default: 0.00003)
  --stageb-lr <float>                    (default: 0.00001)
  --stagec-lr <float>                    (default: 0.000003)
  --stagea-first-char-aux <float>        (default: 0.04)
  --stageb-first-char-aux <float>        (default: 0.03)
  --stagec-first-char-aux <float>        (default: 0.02)
  --province-balance-mode <mode>         (default: inv_sqrt) [none|inv_sqrt|inv]
  --province-balance-clip <float>        (default: 8.0)
  --adj-repeat-sample-weight <float>     (default: 1.5)
  --eval-decode-mode <mode>              (default: green_ctc_beam) [greedy|green_ctc_beam]
  --beam-size <int>                      (default: 30)
  --beam-topk <int>                      (default: 15)
  --hard-sample-weight <float>           (default: 1.3)
  --hard-max-per-stratum <int>           (default: 200)
  --hard-min-edit-distance <int>         (default: 3)
  --round2-stageb-epochs <int>           (default: 8)
  --round2-stagec-epochs <int>           (default: 6)
  --round2-stageb-lr <float>             (default: 0.000008)
  --round2-stagec-lr <float>             (default: 0.000003)
  --seed <int>                           (default: 20260320)
  --help
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ROOT_DIR}/.conda/bin/python"
DATE_TAG="$(date +%Y%m%d)"

RUN_TAG="green_official_v1_${DATE_TAG}"
BASE_MODEL="${ROOT_DIR}/weights_red_stage3/Final_LPRNet_model.pth"
GREEN_ROOT="${ROOT_DIR}/CCPD2020/ccpd_green"

TARGET_GREEN_FULL_EXACT="0.50"
ENABLE_HARD_ROUND="true"

TRAIN_CAP_PER_PROVINCE="2200"
TRAIN_MAX_MAJOR_RATIO="0.90"
VAL_BALANCED_PER_PROVINCE="120"
TEST_BALANCED_PER_PROVINCE="260"

STAGEA_EPOCHS="18"
STAGEB_EPOCHS="14"
STAGEC_EPOCHS="10"

STAGEA_LR="0.00003"
STAGEB_LR="0.00001"
STAGEC_LR="0.000003"
STAGEA_LR_SCHEDULE=(8 14)
STAGEB_LR_SCHEDULE=(6 10)
STAGEC_LR_SCHEDULE=(4 7)

STAGEA_FIRST_CHAR_AUX="0.04"
STAGEB_FIRST_CHAR_AUX="0.03"
STAGEC_FIRST_CHAR_AUX="0.02"
SECOND_CHAR_AUX="0.0"
NE_TYPE_AUX="0.0"

PROVINCE_BALANCE_MODE="inv_sqrt"
PROVINCE_BALANCE_CLIP="8.0"
STRATA_BALANCE_MODE="none"
STRATA_BALANCE_CLIP="0.0"
ADJ_REPEAT_SAMPLE_WEIGHT="1.5"

EVAL_DECODE_MODE="green_ctc_beam"
BEAM_SIZE="30"
BEAM_TOPK="15"
HARD_DECODE_MODE="green_ctc_beam"

HARD_SAMPLE_WEIGHT="1.3"
HARD_MAX_PER_STRATUM="200"
HARD_MIN_EDIT_DISTANCE="3"

ROUND2_STAGEB_EPOCHS="8"
ROUND2_STAGEC_EPOCHS="6"
ROUND2_STAGEB_LR="0.000008"
ROUND2_STAGEC_LR="0.000003"
ROUND2_STAGEB_LR_SCHEDULE=(3 6)
ROUND2_STAGEC_LR_SCHEDULE=(2 4)

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
    --target-green-full-exact) TARGET_GREEN_FULL_EXACT="$2"; shift 2 ;;
    --enable-hard-round) ENABLE_HARD_ROUND="$2"; shift 2 ;;
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
    --stagea-first-char-aux) STAGEA_FIRST_CHAR_AUX="$2"; shift 2 ;;
    --stageb-first-char-aux) STAGEB_FIRST_CHAR_AUX="$2"; shift 2 ;;
    --stagec-first-char-aux) STAGEC_FIRST_CHAR_AUX="$2"; shift 2 ;;
    --province-balance-mode) PROVINCE_BALANCE_MODE="$2"; shift 2 ;;
    --province-balance-clip) PROVINCE_BALANCE_CLIP="$2"; shift 2 ;;
    --adj-repeat-sample-weight) ADJ_REPEAT_SAMPLE_WEIGHT="$2"; shift 2 ;;
    --eval-decode-mode) EVAL_DECODE_MODE="$2"; shift 2 ;;
    --beam-size) BEAM_SIZE="$2"; shift 2 ;;
    --beam-topk) BEAM_TOPK="$2"; shift 2 ;;
    --hard-sample-weight) HARD_SAMPLE_WEIGHT="$2"; shift 2 ;;
    --hard-max-per-stratum) HARD_MAX_PER_STRATUM="$2"; shift 2 ;;
    --hard-min-edit-distance) HARD_MIN_EDIT_DISTANCE="$2"; shift 2 ;;
    --round2-stageb-epochs) ROUND2_STAGEB_EPOCHS="$2"; shift 2 ;;
    --round2-stagec-epochs) ROUND2_STAGEC_EPOCHS="$2"; shift 2 ;;
    --round2-stageb-lr) ROUND2_STAGEB_LR="$2"; shift 2 ;;
    --round2-stagec-lr) ROUND2_STAGEC_LR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

ENABLE_HARD_ROUND="$(echo "${ENABLE_HARD_ROUND}" | tr '[:upper:]' '[:lower:]')"

OUT_DIR="${ROOT_DIR}/experiments/${RUN_TAG}"
LABELS_DIR="${OUT_DIR}/labels"
GREEN_LABEL_SRC_DIR="${OUT_DIR}/green_source_labels"
WEIGHTS_A_R1="${OUT_DIR}/weights_stageA_r1"
WEIGHTS_B_R1="${OUT_DIR}/weights_stageB_r1"
WEIGHTS_C_R1="${OUT_DIR}/weights_stageC_r1"
WEIGHTS_B_R2="${OUT_DIR}/weights_stageB_r2"
WEIGHTS_C_R2="${OUT_DIR}/weights_stageC_r2"
mkdir -p "${OUT_DIR}" "${LABELS_DIR}" "${GREEN_LABEL_SRC_DIR}" "${WEIGHTS_A_R1}" "${WEIGHTS_B_R1}" "${WEIGHTS_C_R1}" "${WEIGHTS_B_R2}" "${WEIGHTS_C_R2}"

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
echo "[Config] target_green_full_exact=${TARGET_GREEN_FULL_EXACT}"
echo "[Config] enable_hard_round=${ENABLE_HARD_ROUND}"
echo "[Config] eval_decode_mode=${EVAL_DECODE_MODE} beam=${BEAM_SIZE}/${BEAM_TOPK}"
echo "[Config] hard_decode_mode=${HARD_DECODE_MODE} min_edit=${HARD_MIN_EDIT_DISTANCE} hard_weight=${HARD_SAMPLE_WEIGHT}"
echo "[Config] province_balance=${PROVINCE_BALANCE_MODE} clip=${PROVINCE_BALANCE_CLIP} strata=${STRATA_BALANCE_MODE}"
echo "[Config] aux first=(A:${STAGEA_FIRST_CHAR_AUX},B:${STAGEB_FIRST_CHAR_AUX},C:${STAGEC_FIRST_CHAR_AUX}) second=${SECOND_CHAR_AUX} ne=${NE_TYPE_AUX}"
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

assert_split_no_leak() {
  local manifest_json="$1"
  "${ENV_PY}" - <<PY
import json
from pathlib import Path

manifest = Path(${manifest_json@Q})
data = json.loads(manifest.read_text(encoding="utf-8"))
overlap = data.get("overlap", {})
bad = {k: int(v) for k, v in overlap.items() if int(v) != 0}
if bad:
    raise SystemExit(f"split leakage detected: {bad}")
print("[LeakCheck] split overlap clean:", overlap)
PY
}

assert_hard_no_leak() {
  local train_txt="$1"
  local val_txt="$2"
  local test_txt="$3"
  local hard_txt="$4"
  local tag="$5"
  "${ENV_PY}" - <<PY
from pathlib import Path

def read_paths(p):
    rows = set()
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            rows.add(parts[0])
    return rows

train_set = read_paths(${train_txt@Q})
val_set = read_paths(${val_txt@Q})
test_set = read_paths(${test_txt@Q})
hard_set = read_paths(${hard_txt@Q})
tag = ${tag@Q}

if not hard_set:
    raise SystemExit(f"{tag} hard set is empty")
if not hard_set.issubset(train_set):
    missing = len(hard_set - train_set)
    raise SystemExit(f"{tag} hard set has {missing} samples outside train_full")
iv = len(hard_set & val_set)
it = len(hard_set & test_set)
if iv or it:
    raise SystemExit(f"{tag} hard leakage detected: with val={iv}, with test={it}")
print(f"[LeakCheck] {tag} hard set clean: size={len(hard_set)} val_overlap=0 test_overlap=0")
PY
}

extract_green_exact() {
  local metrics_json="$1"
  "${ENV_PY}" - <<PY
import json
from pathlib import Path

data = json.loads(Path(${metrics_json@Q}).read_text(encoding="utf-8"))
print(float(data.get("exact_plate_acc", 0.0)))
PY
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
    --decode_mode "${HARD_DECODE_MODE}" \
    --beam_size "${BEAM_SIZE}" \
    --beam_topk "${BEAM_TOPK}" \
    --max_per_stratum "${HARD_MAX_PER_STRATUM}" \
    --min_edit_distance "${HARD_MIN_EDIT_DISTANCE}" \
    --batch_size 120 \
    --num_workers "${NUM_WORKERS}" \
    --cuda "${USE_CUDA}" \
    --seed "${SEED}" | tee "${out_json%.json}.stdout.json"
}

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

echo "[${step}] Verify split leakage constraints"
step=$((step + 1))
assert_split_no_leak "${OUT_DIR}/split_manifest.stdout.json"

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
  --second_char_aux_weight "${SECOND_CHAR_AUX}"
  --ne_type_aux_weight "${NE_TYPE_AUX}"
  --selection_decode_mode "${EVAL_DECODE_MODE}"
  --selection_beam_size "${BEAM_SIZE}"
  --selection_beam_topk "${BEAM_TOPK}"
  --selection_proxy_eval_samples 5000
  --first_char_time_steps 6
  --train_plate_box_aug_mode none
  --train_plate_box_aug_prob 0.0
  --save_interval 2000
  --train_batch_size 64
  --test_batch_size 120
  --num_workers "${NUM_WORKERS}"
  --cuda "${USE_CUDA}"
)

echo "[${step}] Round1 Stage A (balanced warmup, val_full selection)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_BAL}" \
  --test_txt_file "${VAL_FULL}" \
  --first_char_aux_weight "${STAGEA_FIRST_CHAR_AUX}" \
  --pretrained_model "${BASE_MODEL}" \
  --learning_rate "${STAGEA_LR}" \
  --lr_schedule "${STAGEA_LR_SCHEDULE[@]}" \
  --save_folder "${WEIGHTS_A_R1}/" \
  --max_epoch "${STAGEA_EPOCHS}" | tee "${OUT_DIR}/train_stageA_r1.log"

MODEL_A_R1="${WEIGHTS_A_R1}/Final_LPRNet_model.pth"
if [[ ! -f "${MODEL_A_R1}" ]]; then
  echo "Round1 StageA final model not found: ${MODEL_A_R1}" >&2
  exit 1
fi

echo "[${step}] Round1 Stage B (full fit, val_full selection)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_FULL}" \
  --test_txt_file "${VAL_FULL}" \
  --first_char_aux_weight "${STAGEB_FIRST_CHAR_AUX}" \
  --pretrained_model "${MODEL_A_R1}" \
  --learning_rate "${STAGEB_LR}" \
  --lr_schedule "${STAGEB_LR_SCHEDULE[@]}" \
  --save_folder "${WEIGHTS_B_R1}/" \
  --max_epoch "${STAGEB_EPOCHS}" | tee "${OUT_DIR}/train_stageB_r1.log"

MODEL_B_R1="${WEIGHTS_B_R1}/Final_LPRNet_model.pth"
if [[ ! -f "${MODEL_B_R1}" ]]; then
  echo "Round1 StageB final model not found: ${MODEL_B_R1}" >&2
  exit 1
fi

echo "[${step}] Round1 Stage C (full refine, val_full selection)"
step=$((step + 1))
"${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
  "${TRAIN_COMMON_ARGS[@]}" \
  --train_txt_file "${TRAIN_FULL}" \
  --test_txt_file "${VAL_FULL}" \
  --first_char_aux_weight "${STAGEC_FIRST_CHAR_AUX}" \
  --pretrained_model "${MODEL_B_R1}" \
  --learning_rate "${STAGEC_LR}" \
  --lr_schedule "${STAGEC_LR_SCHEDULE[@]}" \
  --save_folder "${WEIGHTS_C_R1}/" \
  --max_epoch "${STAGEC_EPOCHS}" | tee "${OUT_DIR}/train_stageC_r1.log"

MODEL_C_R1="${WEIGHTS_C_R1}/Final_LPRNet_model.pth"
if [[ ! -f "${MODEL_C_R1}" ]]; then
  echo "Round1 StageC final model not found: ${MODEL_C_R1}" >&2
  exit 1
fi

echo "[${step}] Round1 final metrics"
step=$((step + 1))
run_eval "${TEST_FULL}" "${MODEL_C_R1}" "${OUT_DIR}/test_green_full_metrics_r1.json"
run_eval "${TEST_BAL}" "${MODEL_C_R1}" "${OUT_DIR}/test_green_balanced_metrics_r1.json"
run_eval_greedy "${TEST_FULL}" "${MODEL_C_R1}" "${OUT_DIR}/test_green_full_metrics_greedy_r1.json"

ROUND1_EXACT="$(extract_green_exact "${OUT_DIR}/test_green_full_metrics_r1.json")"
echo "[Gate] round1 green_full_exact=${ROUND1_EXACT}, target=${TARGET_GREEN_FULL_EXACT}"

ROUND2_RAN="false"
FINAL_MODEL="${MODEL_C_R1}"
FINAL_METRICS_JSON="${OUT_DIR}/test_green_full_metrics_r1.json"
ROUND2_EXACT=""

if "${ENV_PY}" - <<PY
target = float(${TARGET_GREEN_FULL_EXACT@Q})
score = float(${ROUND1_EXACT@Q})
raise SystemExit(0 if score >= target else 2)
PY
then
  echo "[Gate] round1 passed"
else
  if [[ "${ENABLE_HARD_ROUND}" != "true" && "${ENABLE_HARD_ROUND}" != "1" && "${ENABLE_HARD_ROUND}" != "yes" ]]; then
    echo "[Gate] round1 failed and hard round disabled" >&2
    exit 2
  fi

  ROUND2_RAN="true"
  echo "[${step}] Round2 mine hard from StageA_r1 (beam + strict edit)"
  step=$((step + 1))
  mine_hard "${MODEL_A_R1}" "${TRAIN_FULL}" "${HARD_A_TXT}" "${OUT_DIR}/hard_stageA_stats.json"
  assert_hard_no_leak "${TRAIN_FULL}" "${VAL_FULL}" "${TEST_FULL}" "${HARD_A_TXT}" "stageA_hard"

  echo "[${step}] Round2 Stage B (full + hardA)"
  step=$((step + 1))
  "${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
    "${TRAIN_COMMON_ARGS[@]}" \
    --train_txt_file "${TRAIN_FULL}" \
    --test_txt_file "${VAL_FULL}" \
    --first_char_aux_weight "${STAGEB_FIRST_CHAR_AUX}" \
    --pseudo_anchor_img_dirs "${GREEN_ROOT}" \
    --pseudo_anchor_train_txt_file "${HARD_A_TXT}" \
    --pseudo_anchor_sample_weight "${HARD_SAMPLE_WEIGHT}" \
    --pretrained_model "${MODEL_A_R1}" \
    --learning_rate "${ROUND2_STAGEB_LR}" \
    --lr_schedule "${ROUND2_STAGEB_LR_SCHEDULE[@]}" \
    --save_folder "${WEIGHTS_B_R2}/" \
    --max_epoch "${ROUND2_STAGEB_EPOCHS}" | tee "${OUT_DIR}/train_stageB_r2.log"

  MODEL_B_R2="${WEIGHTS_B_R2}/Final_LPRNet_model.pth"
  if [[ ! -f "${MODEL_B_R2}" ]]; then
    echo "Round2 StageB final model not found: ${MODEL_B_R2}" >&2
    exit 1
  fi

  echo "[${step}] Round2 mine hard from StageB_r2"
  step=$((step + 1))
  mine_hard "${MODEL_B_R2}" "${TRAIN_FULL}" "${HARD_B_TXT}" "${OUT_DIR}/hard_stageB_stats.json"
  assert_hard_no_leak "${TRAIN_FULL}" "${VAL_FULL}" "${TEST_FULL}" "${HARD_B_TXT}" "stageB_hard"

  echo "[${step}] Round2 Stage C (full + hardB)"
  step=$((step + 1))
  "${ENV_PY}" "${ROOT_DIR}/train_LPRNet.py" \
    "${TRAIN_COMMON_ARGS[@]}" \
    --train_txt_file "${TRAIN_FULL}" \
    --test_txt_file "${VAL_FULL}" \
    --first_char_aux_weight "${STAGEC_FIRST_CHAR_AUX}" \
    --pseudo_anchor_img_dirs "${GREEN_ROOT}" \
    --pseudo_anchor_train_txt_file "${HARD_B_TXT}" \
    --pseudo_anchor_sample_weight "${HARD_SAMPLE_WEIGHT}" \
    --pretrained_model "${MODEL_B_R2}" \
    --learning_rate "${ROUND2_STAGEC_LR}" \
    --lr_schedule "${ROUND2_STAGEC_LR_SCHEDULE[@]}" \
    --save_folder "${WEIGHTS_C_R2}/" \
    --max_epoch "${ROUND2_STAGEC_EPOCHS}" | tee "${OUT_DIR}/train_stageC_r2.log"

  MODEL_C_R2="${WEIGHTS_C_R2}/Final_LPRNet_model.pth"
  if [[ ! -f "${MODEL_C_R2}" ]]; then
    echo "Round2 StageC final model not found: ${MODEL_C_R2}" >&2
    exit 1
  fi

  echo "[${step}] Round2 final metrics"
  step=$((step + 1))
  run_eval "${TEST_FULL}" "${MODEL_C_R2}" "${OUT_DIR}/test_green_full_metrics_r2.json"
  run_eval "${TEST_BAL}" "${MODEL_C_R2}" "${OUT_DIR}/test_green_balanced_metrics_r2.json"
  run_eval_greedy "${TEST_FULL}" "${MODEL_C_R2}" "${OUT_DIR}/test_green_full_metrics_greedy_r2.json"

  ROUND2_EXACT="$(extract_green_exact "${OUT_DIR}/test_green_full_metrics_r2.json")"
  echo "[Gate] round2 green_full_exact=${ROUND2_EXACT}, target=${TARGET_GREEN_FULL_EXACT}"

  FINAL_MODEL="${MODEL_C_R2}"
  FINAL_METRICS_JSON="${OUT_DIR}/test_green_full_metrics_r2.json"
fi

echo "[${step}] Build acceptance + scorecard"
step=$((step + 1))
"${ENV_PY}" - <<PY | tee "${OUT_DIR}/scorecard.stdout.json"
import json
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(${OUT_DIR@Q})
target = float(${TARGET_GREEN_FULL_EXACT@Q})
round1 = float(${ROUND1_EXACT@Q})
round2_ran = ${ROUND2_RAN@Q} == "true"
round2 = float(${ROUND2_EXACT@Q}) if round2_ran else None
final_exact = round2 if round2_ran else round1
passed = final_exact >= target

acceptance = {
    "target_green_full_exact": target,
    "round1_green_full_exact": round1,
    "round2_ran": round2_ran,
    "round2_green_full_exact": round2,
    "final_green_full_exact": final_exact,
    "passed": passed,
}
(run_dir / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")

scorecard = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "base_model": ${BASE_MODEL@Q},
    "final_model": ${FINAL_MODEL@Q},
    "target_green_full_exact": target,
    "eval_decode_mode": ${EVAL_DECODE_MODE@Q},
    "beam_size": int(${BEAM_SIZE@Q}),
    "beam_topk": int(${BEAM_TOPK@Q}),
    "province_balance_mode": ${PROVINCE_BALANCE_MODE@Q},
    "province_balance_clip": float(${PROVINCE_BALANCE_CLIP@Q}),
    "strata_balance_mode": ${STRATA_BALANCE_MODE@Q},
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
echo "base_model=${BASE_MODEL}"
echo "final_green_model=${FINAL_MODEL}"
echo "final_metrics_json=${FINAL_METRICS_JSON}"
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
