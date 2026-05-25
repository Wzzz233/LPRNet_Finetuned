#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "./.conda/bin/python" ]]; then
    PYTHON_BIN="./.conda/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

"${PYTHON_BIN}" validate_yolov8_obb_rk3568.py \
  --weights "${1:-yolov8n-obb.pt}" \
  --onnx "${2:-./artifacts/yolov8n_obb/yolov8n-obb.onnx}" \
  --imgsz 640 \
  --opset 12 \
  --target-platform rk3568 \
  --verbose
