# YOLOv8n-OBB RK3568 Validation Workflow

## Goal

Validate whether `YOLOv8n-OBB` can replace the current `YOLOv5` plate detector on `RK3568`, and whether the rotated output improves OCR crops for the existing `LPRNet` path.

This workflow is split into two parts:

1. `YOLOv8n-OBB -> ONNX -> RKNN` compatibility validation
2. `bbox crop` vs `quad crop` OCR-input comparison

## 1. RK3568 Compatibility Check

Install the required packages in your chosen Python environment:

```bash
pip install ultralytics onnx onnxsim rknn-toolkit2
```

In this repo, `./.conda/bin/python` already has the image-processing stack used by the existing OCR scripts, so prefer that interpreter when available.

Run the combined export + audit + RKNN probe:

```bash
./.conda/bin/python validate_yolov8_obb_rk3568.py \
  --weights yolov8n-obb.pt \
  --onnx ./artifacts/yolov8n_obb/yolov8n-obb.onnx \
  --imgsz 640 \
  --opset 12 \
  --target-platform rk3568 \
  --verbose
```

Outputs:

- `artifacts/yolov8n_obb/yolov8n-obb.onnx`
- `artifacts/yolov8n_obb/yolov8n_obb_rk3568_report.json`

The report separates:

- package versions
- ONNX focus ops such as `NMSRotated` and `GatherElements`
- RKNN `config/load_onnx/build` return codes

If you only want to inspect an existing ONNX file:

```bash
./.conda/bin/python inspect_onnx_ops.py ./artifacts/yolov8n_obb/yolov8n-obb.onnx
```

## 2. OCR Crop Comparison

Use the current axis-aligned bbox path and the OBB quad path on the same image:

```bash
./.conda/bin/python compare_plate_crops.py \
  --image /path/to/image.jpg \
  --bbox 120,180,260,220 \
  --quad 122,184,258,176,262,220,126,228 \
  --output-dir ./artifacts/obb_crop_compare
```

Optional OCR evaluation with the existing LPRNet checkpoint:

```bash
./.conda/bin/python compare_plate_crops.py \
  --image /path/to/image.jpg \
  --bbox 120,180,260,220 \
  --quad 122,184,258,176,262,220,126,228 \
  --weights ./experiments/first_char_guard_v1/weights/Final_LPRNet_model.pth \
  --label 京N8P8F8 \
  --output-dir ./artifacts/obb_crop_compare
```

Outputs:

- `overlay.jpg`
- `bbox_raw.jpg`
- `bbox_prepared.jpg`
- `quad_raw.jpg`
- `quad_prepared.jpg`
- `summary.json`

`summary.json` includes occupancy and optional OCR predictions so you can judge whether OBB improves the OCR input distribution rather than only the detector geometry.

## Decision Rule

- `load_onnx/build` fails on RK3568: do not replace the current `YOLOv5` detector directly.
- backbone/head converts but rotated postprocess does not: keep NPU inference and move OBB decode/NMS to ARM for timing evaluation.
- OBB converts and `quad_prepared` crops improve OCR predictions: proceed to board-side integration.
