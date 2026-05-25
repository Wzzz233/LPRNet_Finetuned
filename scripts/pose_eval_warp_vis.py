#!/usr/bin/env python3
"""Warp comparison: pose vs GT on hard green plates."""

import json, sys, os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

POSE_WEIGHT = "/home/wzzz/LPRNet/experiments/yolov8n-pos/weights/best.pt"
DATASET_DIR = Path("/home/wzzz/LPRNet/datasets/plate_true_quad_pose")
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = Path("/home/wzzz/LPRNet/reports/pose_eval/warp_vis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OCR crop dimensions (from board pipeline)
OCR_W, OCR_H = 150, 50

model = YOLO(POSE_WEIGHT)

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

def warp_quad_to_rect(img, quad, dst_w=OCR_W, dst_h=OCR_H):
    """Perspective warp from quad to rect."""
    dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_pts)
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    return warped

def parse_label(label_path):
    content = open(label_path).read().strip().splitlines()
    for line in content:
        parts = line.split()
        coords = list(map(float, parts[5:]))
        kp = np.array([(coords[i], coords[i+1]) for i in range(0, len(coords), 3)])
        return kp
    return None

# Get hard green images
test_paths = (DATASET_DIR / "test.txt").read_text().strip().splitlines()
hard_green = [p for p in test_paths if "c2020_gh" in p]
print(f"Hard green images: {len(hard_green)}")

# Pick top 10 by difficulty (hardest = last in slice)
# Also get a few easy ones for comparison
import random
random.seed(42)
samples = random.sample(hard_green, min(20, len(hard_green)))

results_rows = []

for i, rel_path in enumerate(samples):
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
    
    # GT quad (normalized)
    gt_kps_norm = parse_label(str(label_path))
    if gt_kps_norm is None:
        continue
    
    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]
    
    # GT quad in pixels
    gt_kps_px = gt_kps_norm * np.array([w, h])
    
    # Run pose model
    r = model(img_path, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    
    if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
        continue
    
    pred_kps_tensor = r.keypoints.xy.cpu().numpy()
    if pred_kps_tensor.ndim != 3 or pred_kps_tensor.shape[0] == 0 or pred_kps_tensor.shape[1] != 4:
        continue
    
    # Pick best detection
    if r.keypoints.conf is not None:
        kp_conf = r.keypoints.conf.cpu().numpy()
        det_conf = kp_conf.mean(axis=1) if kp_conf.ndim > 1 else kp_conf
        best = int(np.argmax(det_conf))
    else:
        best = 0
    
    pred_kps_px = pred_kps_tensor[best]
    
    # Warps
    gt_warp = warp_quad_to_rect(img_bgr, gt_kps_px)
    pose_warp = warp_quad_to_rect(img_bgr, pred_kps_px)
    
    # Save comparison
    # Create comparison image with dynamic sizing
    scale = 5
    small_w, small_h = w // scale, h // scale
    vis_h = max(OCR_H * 2 + 20, small_h + 30)
    vis_w = OCR_W * 2 + 15 + small_w
    vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    
    # GT warp and Pose warp side by side
    vis[:OCR_H, :OCR_W] = gt_warp
    vis[:OCR_H, OCR_W + 5:2 * OCR_W + 5] = pose_warp
    
    # Warp labels
    cv2.putText(vis, "GT warp", (5, OCR_H + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(vis, "Pose warp", (OCR_W + 10, OCR_H + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Original image with GT and Pose quad drawn
    orig_small = cv2.resize(img_bgr, (small_w, small_h))
    ox = 2 * OCR_W + 15
    oy = max(0, (vis_h - small_h) // 2)
    vis[oy:oy + small_h, ox:ox + small_w] = orig_small
    
    # Draw GT (green) and Pose (blue) quads on the small image
    gt_scaled = (gt_kps_px / scale).astype(np.int32)
    pose_scaled = (pred_kps_px / scale).astype(np.int32)
    cv2.polylines(orig_small, [gt_scaled.reshape(-1, 1, 2)], True, (0, 255, 0), 1)
    cv2.polylines(orig_small, [pose_scaled.reshape(-1, 1, 2)], True, (255, 0, 0), 1)
    vis[oy:oy + small_h, ox:ox + small_w] = orig_small
    
    # Error
    err = np.mean(np.linalg.norm(pred_kps_px - gt_kps_px, axis=1))
    cv2.putText(vis, f"err={err:.1f}px", (ox, h // 4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    out_path = str(OUTPUT_DIR / f"hard_green_{i:02d}_err{err:.0f}.jpg")
    cv2.imwrite(out_path, vis)
    
    results_rows.append({
        "idx": i, "rel_path": rel_path, "error_px": round(err, 2),
        "out_path": out_path
    })
    
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(samples)}")

# Save results
with open(OUTPUT_DIR / "warp_results.json", "w") as f:
    json.dump(results_rows, f, indent=2)

print(f"\nDone. {len(results_rows)} samples. Output in {OUTPUT_DIR}")
