# YOLOv8-OBB Plate Training

## Goal

Train a real plate-only `YOLOv8n-OBB` model from CCPD instead of using the official 15-class OBB checkpoint.

This is required because the official `yolov8n-obb.pt` is not a plate detector. It is only useful for RK3568 compatibility validation.

## 1. Prepare CCPD OBB Dataset

The CCPD filename already contains the 4-point plate polygon, so we can convert it into YOLOv8-OBB labels.

Run:

```bash
./.conda/bin/python prepare_ccpd_yolov8_obb.py \
  --dataset_root ./CCPD2019 \
  --output_dir ./prepared_labels/ccpd2019_yolov8_obb
```

Outputs:

- `prepared_labels/ccpd2019_yolov8_obb/images/{train,val,test}/...`
- `prepared_labels/ccpd2019_yolov8_obb/labels/{train,val,test}/...`
- `prepared_labels/ccpd2019_yolov8_obb/{train,val,test}.txt`
- `prepared_labels/ccpd2019_yolov8_obb/dataset.yaml`

The generated OBB label format is:

```text
0 x1 y1 x2 y2 x3 y3 x4 y4
```

with coordinates normalized to image width/height.

## 2. Training

Install `ultralytics` and its user-space dependencies in the training environment:

```bash
./.conda/bin/python -m pip install ultralytics pyyaml matplotlib scipy psutil tqdm
```

Then start training:

```bash
./run_yolov8_obb_train.sh
```

Useful overrides:

```bash
RUN_NAME=plate_yolov8n_obb_v1 \
MODEL_WEIGHTS=yolov8n-obb.pt \
EPOCHS=80 \
BATCH=32 \
DEVICE=0 \
./run_yolov8_obb_train.sh
```

## 3. Export For RK3568

After training, export the trained weights to ONNX:

```bash
./.conda/bin/python validate_yolov8_obb_rk3568.py \
  --weights /path/to/best.pt \
  --onnx ./artifacts/yolov8n_obb/plate_yolov8n_obb.onnx \
  --imgsz 640 \
  --opset 12 \
  --skip-build
```

Then use the safe RK3568 split:

- preferred ONNX boundary: `484 / 415 / 464`
- preferred board model: `plate_yolov8n_obb_middecode_rk3568_fp16.rknn`

The reason is documented in:

- `artifacts/yolov8n_obb/REGTASK_ROOT_CAUSE.md`

## 4. Board Integration

Board-side requirements:

- detector type: `yolov8_obb_rknn`
- OCR crop mode: `obb_warp`
- OCR model: keep the current fine-tuned `LPRNet_stage3_rk3568_fp16.rknn`

This preserves the current OCR contract while replacing only:

- plate detector
- rotated crop construction
