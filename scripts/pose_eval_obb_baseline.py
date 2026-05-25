#!/usr/bin/env python3
"""OBB baseline evaluation v2: use r.obb.xyxyxyxy for accurate quad extraction."""

import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

OBB_WEIGHT = "/home/wzzz/LPRNet/datasets/downloaded_green_clone_success/best.pt"
DATASET_DIR = Path("/home/wzzz/LPRNet/datasets/plate_true_quad_pose")
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = Path("/home/wzzz/LPRNet/reports/pose_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(OBB_WEIGHT)

def parse_pose_label(label_path):
    content = open(label_path).read().strip().splitlines()
    for line in content:
        parts = line.split()
        coords = list(map(float, parts[5:]))
        kp = np.array([(coords[i], coords[i+1]) for i in range(0, len(coords), 3)])
        return kp
    return None

def order_quad_points(pts):
    quad = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = quad.sum(axis=1)
    diffs = np.diff(quad, axis=1).reshape(-1)
    ordered[0] = quad[np.argmin(sums)]
    ordered[2] = quad[np.argmax(sums)]
    ordered[1] = quad[np.argmin(diffs)]
    ordered[3] = quad[np.argmax(diffs)]
    return ordered

# Get green plate images from test.txt
test_paths = (DATASET_DIR / "test.txt").read_text().strip().splitlines()
green_paths = [p for p in test_paths if "c2020" in p]
print(f"Green plate images: {len(green_paths)}")

all_errors = []
per_slice = {}
n_missed = 0
n_obb_fail = 0

for i, rel_path in enumerate(green_paths):
    img_path = str(DATASET_DIR / rel_path)
    fname = Path(img_path).name
    label_name = fname.replace(".jpg", ".txt")
    
    # Find label
    label_path = None
    for sd in ["val", "test", "train"]:
        c = LABELS_DIR / sd / label_name
        if c.exists():
            label_path = c
            break
    if label_path is None:
        continue
    
    # GT
    gt_kps_norm = parse_pose_label(str(label_path))
    if gt_kps_norm is None:
        continue
    
    w, h = Image.open(img_path).size
    gt_kps_px = gt_kps_norm * np.array([w, h])
    
    # Run OBB
    r = model(img_path, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    
    if r.obb is None or len(r.obb) == 0:
        n_missed += 1
        continue
    
    # Use xyxyxyxy (4 corners in pixels)
    obb_corners = r.obb.xyxyxyxy.cpu().numpy()  # (n, 4, 2)
    
    if len(obb_corners) == 0:
        n_obb_fail += 1
        continue
    
    # Pick highest confidence detection
    confs = r.obb.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    pred_kps_px = order_quad_points(obb_corners[best_idx])
    
    # Error
    errs = np.linalg.norm(pred_kps_px - gt_kps_px, axis=1)
    mean_err = float(errs.mean())
    all_errors.append(mean_err)
    
    diff = "unknown"
    if "c2020_ge" in rel_path: diff = "green_easy"
    elif "c2020_gm" in rel_path: diff = "green_mid"
    elif "c2020_gh" in rel_path: diff = "green_hard"
    per_slice.setdefault(diff, []).append(mean_err)
    
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(green_paths)}")

# Report
print("\n" + "=" * 70)
print("OBB BASELINE - GREEN PLATE CORNER ERROR (pixels)")
print("=" * 70)

data = {}
for diff in ["green_easy", "green_mid", "green_hard"]:
    if diff in per_slice and per_slice[diff]:
        errs = np.array(per_slice[diff])
        data[diff] = errs
        print(f"  {diff:<15} n={len(errs):>5}  "
              f"mean={errs.mean():.2f}px  "
              f"median={float(np.median(errs)):.2f}px  "
              f"p90={float(np.percentile(errs, 90)):.2f}px  "
              f"max={errs.max():.2f}px  "
              f"min={errs.min():.2f}px")

if all_errors:
    a = np.array(all_errors)
    print(f"  {'OBB ALL GREEN':<15} n={len(a):>5}  "
          f"mean={a.mean():.2f}px  "
          f"median={float(np.median(a)):.2f}px  "
          f"p90={float(np.percentile(a, 90)):.2f}px  "
          f"max={a.max():.2f}px  "
          f"missed={n_missed}  obb_fail={n_obb_fail}")

# Also compute CCPD2019 for reference
print("\n" + "=" * 70)
print("POSE MODEL (for comparison)")
print("=" * 70)
for diff, label in [("green_easy", "green_easy"), ("green_mid", "green_mid"), ("green_hard", "green_hard")]:
    if diff in per_slice:
        errs = np.array(per_slice[diff])
        m = np.median(errs)
        print(f"  OBB {diff:<15} median={m:.2f}px")
