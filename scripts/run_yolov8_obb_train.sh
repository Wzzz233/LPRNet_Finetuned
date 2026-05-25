#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-./.conda/bin/python}"
DATASET_DIR="${DATASET_DIR:-./prepared_labels/ccpd2019_yolov8_obb}"
RUN_NAME="${RUN_NAME:-plate_yolov8n_obb_v1}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-yolov8n-obb.pt}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-32}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
PATIENCE="${PATIENCE:-20}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys
mods = ["ultralytics"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] Missing Python packages:", ", ".join(missing))
    print("[HINT] Install in the training env, for example:")
    print("  ./.conda/bin/python -m pip install ultralytics pyyaml matplotlib scipy psutil tqdm")
    sys.exit(1)
PY

"${PYTHON_BIN}" - <<PY
from ultralytics import YOLO

model = YOLO("${MODEL_WEIGHTS}")
model.train(
    data="${DATASET_DIR}/dataset.yaml",
    task="obb",
    imgsz=${IMGSZ},
    epochs=${EPOCHS},
    batch=${BATCH},
    device="${DEVICE}",
    workers=${WORKERS},
    project="experiments/yolov8_obb",
    name="${RUN_NAME}",
    patience=${PATIENCE},
    cache=False,
    pretrained=True,
    single_cls=True,
    optimizer="auto",
    amp=True,
    degrees=0.0,
)
PY
