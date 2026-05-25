# YOLOv8-OBB Training Handoff

## Goal

Train a real **plate-only** `YOLOv8n-OBB` model from CCPD, then export a board-testable RK3568 detector artifact for real-time testing with the existing fine-tuned `LPRNet_stage3_rk3568_fp16.rknn`.

Do **not** use the official `yolov8n-obb.pt` as the final detector model.  
It is only a compatibility baseline and is **not** a plate detector.

## Current Repo State

Use this repo revision:

- commit: `b57560b9268de422c4071e6132049df35b493d15`

Relevant files already prepared:

- `prepare_ccpd_yolov8_obb.py`
- `run_yolov8_obb_train.sh`
- `docs/YOLOV8_OBB_PLATE_TRAINING.md`
- `validate_yolov8_obb_rk3568.py`
- `rewrite_onnx_outputs.py`
- `artifacts/yolov8n_obb/REGTASK_ROOT_CAUSE.md`

## Dataset And Split Rules

The OBB dataset is already separated from the LPRNet OCR labels.  
Do **not** touch or overwrite the old OCR split files under `prepared_labels/ccpd2019`.

Use only:

- `prepared_labels/ccpd2019_yolov8_obb/train.txt`
- `prepared_labels/ccpd2019_yolov8_obb/val.txt`
- `prepared_labels/ccpd2019_yolov8_obb/test.txt`

Current split sizes:

- train: `100000`
- val: `99996`
- test: `141982`

These must stay separated. Do not merge train/val, and do not regenerate mixed splits.

## Training Environment

Known-good package versions:

- Python `3.10.20`
- torch `2.5.1+cu124`
- torchvision `0.20.1+cu124`
- ultralytics `8.4.23`
- numpy `2.2.6`
- opencv-python `4.13.0`
- onnx `1.20.1`
- pyyaml `6.0.3`
- scipy `1.15.3`
- psutil `7.2.2`
- tqdm `4.67.3`
- matplotlib `3.10.8`
- requests `2.32.5`
- polars `1.39.0`

Recommended machine:

- Linux
- NVIDIA GPU with at least `16 GB` VRAM
- CUDA compatible with `torch 2.5.1+cu124`
- at least `100 GB` free disk

## What To Train

Train:

- model family: `YOLOv8n-OBB`
- task: `obb`
- classes: single class only
- class name: `plate`

Use:

- starting weights: `yolov8n-obb.pt`
- dataset yaml: `prepared_labels/ccpd2019_yolov8_obb/dataset.yaml`

Do not change the dataset to multi-class.  
The board integration expects a plate detector, not a generic 15-class OBB model.

## Training Procedure

### 1. Sanity check the dataset

Before full training, run a tiny probe:

```bash
python prepare_ccpd_yolov8_obb.py \
  --dataset_root ./CCPD2019 \
  --output_dir ./prepared_labels/ccpd2019_yolov8_obb
```

Then confirm:

```bash
wc -l prepared_labels/ccpd2019_yolov8_obb/train.txt \
      prepared_labels/ccpd2019_yolov8_obb/val.txt \
      prepared_labels/ccpd2019_yolov8_obb/test.txt
```

Expected:

- `100000`
- `99996`
- `141982`

### 2. Run a short probe training first

Do not jump straight to a long run.  
First verify the machine can complete a short training epoch cleanly.

Example:

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolov8n-obb.pt')
model.train(
    data='prepared_labels/ccpd2019_yolov8_obb/dataset.yaml',
    task='obb',
    imgsz=640,
    epochs=1,
    fraction=0.01,
    batch=16,
    device='0',
    workers=4,
    project='experiments/yolov8_obb',
    name='probe',
    single_cls=True,
    pretrained=True,
    optimizer='auto',
    amp=True,
    degrees=0.0,
)
PY
```

If that passes, move to full training.

### 3. Full training

Start with a practical run such as:

```bash
RUN_NAME=plate_yolov8n_obb_v1 \
MODEL_WEIGHTS=yolov8n-obb.pt \
DATASET_DIR=./prepared_labels/ccpd2019_yolov8_obb \
EPOCHS=20 \
BATCH=32 \
DEVICE=0 \
WORKERS=8 \
PATIENCE=10 \
./run_yolov8_obb_train.sh
```

If VRAM allows, try higher batch sizes after a probe.  
Do not assume `auto batch` is stable; explicit batch size is preferred.

### 4. If the first full run is weak, keep iterating

This handoff is for a **finished artifact**, not a half-step.

If the first run is weak:

- continue training longer
- adjust batch size
- retry with a fresh run name
- compare validation metrics

Do not stop at “it trains”.

## Model Selection

The final deliverable must be selected from validation performance, not from convenience.

At minimum, collect:

- best checkpoint path
- validation mAP50
- validation recall
- a few qualitative predictions on rotated plate samples

Do not use the official `yolov8n-obb.pt` as fallback unless training completely fails and you explicitly call that out as a blocker.

## Export For RK3568

After selecting the best `best.pt`, export to ONNX:

```bash
python validate_yolov8_obb_rk3568.py \
  --weights /path/to/best.pt \
  --onnx ./artifacts/yolov8n_obb/plate_yolov8n_obb.onnx \
  --imgsz 640 \
  --opset 12 \
  --skip-build
```

Then create the **safe RK3568 split**.

Important board constraint:

- do **not** use full decoded `output0`
- do **not** ship the full decode tail as the RKNN boundary

Use the preferred RK3568-safe boundary documented in:

- `artifacts/yolov8n_obb/REGTASK_ROOT_CAUSE.md`

Preferred output boundary:

- `484`
- `415`
- `464`

Meaning:

- `484`: decoded `ltrb`
- `415`: class logits
- `464`: decoded angle scalar

Then export the board model:

- target artifact name: `plate_yolov8n_obb_middecode_rk3568_fp16.rknn`

Optional conservative fallback:

- raw head boundary `372 / 415 / 459`

But preferred is `middecode`.

## Board Integration Contract

The board-side driver is expected to run:

- detector type: `yolov8_obb_rknn`
- OCR crop mode: `obb_warp`
- OCR model: existing fine-tuned `LPRNet_stage3_rk3568_fp16.rknn`

The OCR preprocessing contract must stay:

- BGR
- letterbox
- nearest-neighbor
- normalize `(x - 127.5) / 128.0`

Do not retrain OCR as part of this task.

## Required Deliverables

The task is only complete when all of these exist:

1. A trained plate-only `YOLOv8n-OBB` checkpoint
2. Exported ONNX from the trained checkpoint
3. RK3568-safe `middecode` RKNN artifact
4. A short result summary containing:
   - training config
   - best checkpoint
   - validation metrics
   - export command
   - final RKNN artifact path
5. A few qualitative prediction examples on rotated CCPD images

## Acceptance Standard

The output must be good enough to hand to the engineering side for live board testing.

That means:

- not the official generic 15-class OBB weight
- not an unfinished training run
- not only a PyTorch checkpoint without RKNN export
- not only an RKNN artifact without proof the training run produced a meaningful plate detector

## Known Pitfalls

- Do not overwrite old LPRNet OCR label files or experiments.
- Do not mix train and val splits.
- Do not ship full decoded `output0` RKNN for RK3568.
- Do not stop after a probe run.
- Do not assume “training started” means “artifact ready”.

## Final Response Format

When done, report:

1. Best model path
2. ONNX export path
3. RKNN export path
4. Validation metrics
5. Exact command used for board testing
6. Any remaining risks
