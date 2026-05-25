#!/usr/bin/env python3
"""Convert YOLOv8n-pose ONNX model to RKNN for RK3568 (FP16)."""

import os, sys
from pathlib import Path

ONNX_PATH = "/home/wzzz/LPRNet/experiments/yolov8n-pos/weights/best.onnx"
RKNN_PATH = "/home/wzzz/LPRNet/experiments/yolov8n-pos/weights/best_fp16.rknn"

from rknn.api import RKNN

# Create RKNN object
rknn = RKNN(verbose=False)

# Pre-process config
print("Configuring RKNN...")
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3568",
)

# Load ONNX model
print(f"Loading ONNX model: {ONNX_PATH}")
ret = rknn.load_onnx(model=ONNX_PATH)
if ret != 0:
    print(f"ERROR: Failed to load ONNX model (ret={ret})")
    sys.exit(1)

# Build (quantize=False = FP16)
print("Building RKNN model (FP16)...")
ret = rknn.build(do_quantization=False)
if ret != 0:
    print(f"ERROR: Failed to build RKNN model (ret={ret})")
    sys.exit(1)

# Export RKNN
print(f"Exporting RKNN model: {RKNN_PATH}")
ret = rknn.export_rknn(RKNN_PATH)
if ret != 0:
    print(f"ERROR: Failed to export RKNN (ret={ret})")
    sys.exit(1)

rknn.release()

# Verify
size_mb = os.path.getsize(RKNN_PATH) / 1024 / 1024
print(f"\nSUCCESS: RKNN model exported ({size_mb:.1f} MB)")
print(f"  Path: {RKNN_PATH}")
print(f"  Target: RK3568, FP16")
