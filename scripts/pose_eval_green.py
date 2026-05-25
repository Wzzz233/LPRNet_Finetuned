#!/usr/bin/env python3
"""Quick eval: run pose model on green plates only, report corner error."""

import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

POSE_WEIGHT = "/home/wzzz/LPRNet/experiments/yolov8n-pos/weights/best.pt"
DATASET_DIR = Path("/home/wzzz/LPRNet/datasets/plate_true_quad_pose")
LABELS_DIR = DATASET_DIR / "labels"

# Get green plate images from test.txt
test_paths = (DATASET_DIR / "test.txt").read_text().strip().splitlines()
green_paths = [p for p in test_paths if "c2020" in p]
print(f"Green plate images: {len(green_paths)}")

model = YOLO(POSE_WEIGHT)

def parse_label(label_path):
    content = open(label_path).read().strip().splitlines()
    kps = []
    for line in content:
        parts = line.split()
        coords = list(map(float, parts[5:]))
        kp = np.array([(coords[i], coords[i+1]) for i in range(0, len(coords), 3)])
        kps.append(kp)
    return kps  # list of (4,2) arrays, one per plate in image

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

all_errors = []
per_slice = {}

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
    
    # Get GT
    gts = parse_label(str(label_path))
    gt = gts[0]  # single plate
    
    # Get image size
    w, h = Image.open(img_path).size
    
    # Run inference
    r = model(img_path, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    
    # Extract best detection
    if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
        continue
    
    pred_kps = r.keypoints.xy.cpu().numpy()
    if pred_kps.ndim != 3 or pred_kps.shape[0] == 0 or pred_kps.shape[1] != 4:
        continue
        
    if r.keypoints.conf is not None:
        kp_conf = r.keypoints.conf.cpu().numpy()
        det_conf = kp_conf.mean(axis=1) if kp_conf.ndim > 1 else kp_conf
        best = int(np.argmax(det_conf))
    else:
        best = 0
    pred_kps_best = pred_kps[best]
    pred_kps_norm = order_quad_points(pred_kps_best / np.array([w, h]))
    
    # Error in pixels
    pred_px = pred_kps_norm * np.array([w, h])
    gt_px = gt * np.array([w, h])
    errs = np.linalg.norm(pred_px - gt_px, axis=1)
    mean_err = float(errs.mean())
    all_errors.append(mean_err)
    
    # Per difficulty
    diff = "unknown"
    if "c2020_ge" in rel_path: diff = "green_easy"
    elif "c2020_gm" in rel_path: diff = "green_mid"
    elif "c2020_gh" in rel_path: diff = "green_hard"
    per_slice.setdefault(diff, []).append(mean_err)
    
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(green_paths)}")

# Report
print("\n" + "=" * 70)
print("GREEN PLATE CORNER ERROR (pixels)")
print("=" * 70)

for diff in ["green_easy", "green_mid", "green_hard"]:
    if diff in per_slice and per_slice[diff]:
        errs = np.array(per_slice[diff])
        print(f"  {diff:<15} n={len(errs):>5}  "
              f"mean={errs.mean():.2f}px  "
              f"median={float(np.median(errs)):.2f}px  "
              f"p90={float(np.percentile(errs, 90)):.2f}px  "
              f"max={errs.max():.2f}px  "
              f"min={errs.min():.2f}px")

if all_errors:
    a = np.array(all_errors)
    print(f"  {'ALL GREEN':<15} n={len(a):>5}  "
          f"mean={a.mean():.2f}px  "
          f"median={float(np.median(a)):.2f}px  "
          f"p90={float(np.percentile(a, 90)):.2f}px  "
          f"max={a.max():.2f}px")
