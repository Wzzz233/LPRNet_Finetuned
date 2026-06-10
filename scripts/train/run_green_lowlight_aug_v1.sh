#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wzzz/LPRNet"
PY="${ROOT}/.conda/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi

MANIFEST_DIR="${ROOT}/manifests_rebased/green_lowlight_aug_v1_20260610"
BASE_MODEL="${ROOT}/experiments/a_ratio_r50_20260510/best_LPRNet_model.pth"
TEST_MANIFEST="${MANIFEST_DIR}/lowlight_heldout_6000.csv"
VARIANTS=(lowlight10 lowlight15 lowlight25)
MAX_EPOCH=6
TRAIN_BATCH_SIZE=128
TEST_BATCH_SIZE=120
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_CUDA="${USE_CUDA:-true}"
PREFLIGHT_ONLY=false

usage() {
  cat <<'USAGE'
Usage: scripts/train/run_green_lowlight_aug_v1.sh [options]

Options:
  --variant <lowlight10|lowlight15|lowlight25|all>  default: all
  --base-model <path>                               default: R50 best checkpoint
  --manifest-dir <path>                             default: green_lowlight_aug_v1_20260610
  --max-epoch <n>                                   default: 6
  --preflight-only                                  build model/data and exit
  --cuda <true|false>                               default: USE_CUDA env or true
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --variant)
      if [[ "$2" == "all" ]]; then VARIANTS=(lowlight10 lowlight15 lowlight25); else VARIANTS=("$2"); fi
      shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --manifest-dir) MANIFEST_DIR="$2"; TEST_MANIFEST="${MANIFEST_DIR}/lowlight_heldout_6000.csv"; shift 2 ;;
    --max-epoch) MAX_EPOCH="$2"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=true; shift ;;
    --cuda) USE_CUDA="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Base model not found: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${TEST_MANIFEST}" ]]; then
  echo "Lowlight heldout manifest not found: ${TEST_MANIFEST}" >&2
  echo "Run: ${PY} ${ROOT}/tmp_scripts/build_green_lowlight_aug_v1_dataset.py" >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"

for variant in "${VARIANTS[@]}"; do
  TRAIN_MANIFEST="${MANIFEST_DIR}/train_v1_${variant}.csv"
  EXP_DIR="${ROOT}/experiments/green_lowlight_aug_v1_${variant}_20260610"
  if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
    echo "Train manifest not found: ${TRAIN_MANIFEST}" >&2
    exit 1
  fi
  mkdir -p "${EXP_DIR}"
  echo "[green-lowlight] variant=${variant} train=${TRAIN_MANIFEST} test=${TEST_MANIFEST} out=${EXP_DIR}"

  COMMON_ARGS=(
    --data_mode manifest
    --train_manifest "${TRAIN_MANIFEST}"
    --test_manifest "${TEST_MANIFEST}"
    --dataset_root "${ROOT}"
    --pretrained_model "${BASE_MODEL}"
    --save_folder "${EXP_DIR}/"
    --head_mode multihead --enhanced_green_head expD
    --trainable_families green8
    --freeze_backbone True --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20
    --max_epoch "${MAX_EPOCH}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --test_batch_size "${TEST_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --learning_rate 0.00002 --lr_schedule 3 5
    --lpr_max_len 8
    --ocr_crop_mode obb_warp --ocr_resize_mode letterbox --ocr_resize_kernel nn
    --ocr_preproc none --ocr_channel_order bgr --ocr_quad_pad_ratio 0.0
    --province_balance_mode inv_sqrt
    --first_char_aux_weight 0.4
    --save_interval 2000 --test_interval 2000
    --cuda "${USE_CUDA}" --phase_train True
    --seed 20260610
  )

  if [[ "${PREFLIGHT_ONLY}" == true ]]; then
    "${PY}" -u src/training/train_LPRNet.py "${COMMON_ARGS[@]}" --preflight_only True 2>&1 | tee "${EXP_DIR}/preflight.log"
  else
    "${PY}" -u src/training/train_LPRNet.py "${COMMON_ARGS[@]}" 2>&1 | tee "${EXP_DIR}/train.log"
  fi
done
