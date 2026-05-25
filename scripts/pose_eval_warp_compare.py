#!/usr/bin/env python3
"""Generate OBB vs Pose warp comparison visuals for hard green samples."""

import json, sys, os, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

OBB_WEIGHT = "/home/wzzz/LPRNet/datasets/downloaded_green_clone_success/best.pt"
POSE_WEIGHT = "/home/wzzz/LPRNet/experiments/yolov8n-pos/weights/best.pt"
DATASET_DIR = Path("/home/wzzz/LPRNet/datasets/plate_true_quad_pose")
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = Path("/home/wzzz/LPRNet/reports/pose_eval/warp_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OCR_W, OCR_H = 150, 50

obb_model = YOLO(OBB_WEIGHT)
pose_model = YOLO(POSE_WEIGHT)

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

def warp_quad(img, quad, dw=OCR_W, dh=OCR_H):
    dst = np.array([[0,0],[dw,0],[dw,dh],[0,dh]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (dw, dh), flags=cv2.INTER_LINEAR)

def parse_label(path):
    for line in open(path).read().strip().splitlines():
        parts = line.split()
        coords = list(map(float, parts[5:]))
        return np.array([(coords[i], coords[i+1]) for i in range(0, len(coords), 3)])
    return None

# Get hard green samples
test_paths = (DATASET_DIR / "test.txt").read_text().strip().splitlines()
hard_green = [p for p in test_paths if "c2020_gh" in p]

import random
random.seed(42)
samples = random.sample(hard_green, min(25, len(hard_green)))
print(f"Generating {len(samples)} warp comparisons...")

for i, rel_path in enumerate(samples):
    img_path = str(DATASET_DIR / rel_path)
    fname = Path(img_path).name
    label_name = fname.replace(".jpg", ".txt")
    
    label_path = None
    for sd in ["val", "test", "train"]:
        c = LABELS_DIR / sd / label_name
        if c.exists():
            label_path = c
            break
    if label_path is None:
        continue
    
    gt_norm = parse_label(str(label_path))
    if gt_norm is None:
        continue
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]
    gt_px = gt_norm * np.array([w, h])
    
    # OBB
    r_obb = obb_model(img_path, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    obb_quad = None
    if r_obb.obb is not None and len(r_obb.obb) > 0:
        obb_corners = r_obb.obb.xyxyxyxy.cpu().numpy()
        confs = r_obb.obb.conf.cpu().numpy() if r_obb.obb.conf is not None else None
        if len(obb_corners) > 0:
            best = int(confs.argmax()) if confs is not None else 0
            obb_quad = order_quad_points(obb_corners[best])
    
    # Pose
    r_pose = pose_model(img_path, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    pose_quad = None
    if r_pose.keypoints is not None and r_pose.keypoints.xy is not None:
        kps = r_pose.keypoints.xy.cpu().numpy()
        if kps.ndim == 3 and kps.shape[0] > 0 and kps.shape[1] == 4:
            confs = r_pose.keypoints.conf.cpu().numpy() if r_pose.keypoints.conf is not None else None
            best = int(np.argmax(confs.mean(axis=1))) if confs is not None and confs.ndim > 1 else 0
            pose_quad = order_quad_points(kps[best])
    
    # Generate warps
    gt_warp = warp_quad(img_bgr, gt_px)
    obb_warp = warp_quad(img_bgr, obb_quad) if obb_quad is not None else np.zeros((OCR_H, OCR_W, 3), dtype=np.uint8)
    pose_warp = warp_quad(img_bgr, pose_quad) if pose_quad is not None else np.zeros((OCR_H, OCR_W, 3), dtype=np.uint8)
    
    # Create 2x2 grid: row0=GT+OBB, row1=Pose+original
    cell_w, cell_h = max(OCR_W, w//6), max(OCR_H, h//6)
    grid = np.zeros((cell_h * 2 + 40, cell_w * 2, 3), dtype=np.uint8)
    grid = np.zeros((cell_h * 2 + 40, cell_w * 2, 3), dtype=np.uint8)
    
    # Row 0: GT warp | OBB warp
    grid[:OCR_H, :OCR_W] = gt_warp
    grid[:OCR_H, cell_w:cell_w + OCR_W] = obb_warp
    cv2.putText(grid, "GT", (5, OCR_H + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    cv2.putText(grid, "OBB", (cell_w + 5, OCR_H + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
    
    # Row 1: Pose warp | original with quads
    grid[cell_h + 20:cell_h + 20 + OCR_H, :OCR_W] = pose_warp
    cv2.putText(grid, "Pose", (5, cell_h + 20 + OCR_H + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1)
    
    # Put original image at bottom-right, centered
    orig = cv2.resize(img_bgr, (cell_w, cell_h + 20))
    grid[cell_h + 20:2*(cell_h + 20), cell_w:2*cell_w] = orig[:cell_h+20, :cell_w]
    
    # Draw quads on the resized canvas directly
    scale = max(w, h) / max(cell_w, cell_h + 20)
    if obb_quad is not None:
        q = (obb_quad / scale).astype(np.int32)
        q[:, 0] += cell_w  
        q[:, 1] += cell_h + 20
        cv2.polylines(grid, [q.reshape(-1,1,2)], True, (0,255,255), 1)
    if pose_quad is not None:
        q = (pose_quad / scale).astype(np.int32)
        q[:, 0] += cell_w
        q[:, 1] += cell_h + 20
        cv2.polylines(grid, [q.reshape(-1,1,2)], True, (255,0,0), 1)
    q = (gt_px / scale).astype(np.int32)
    q[:, 0] += cell_w
    q[:, 1] += cell_h + 20
    cv2.polylines(grid, [q.reshape(-1,1,2)], True, (0,255,0), 1)
    
    # Error labels
    obb_err = np.mean(np.linalg.norm(obb_quad - gt_px, axis=1)) if obb_quad is not None else 0
    pose_err = np.mean(np.linalg.norm(pose_quad - gt_px, axis=1)) if pose_quad is not None else 0
    cv2.putText(grid, f"OBB err:{obb_err:.0f}px", (cell_w, cell_h//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    cv2.putText(grid, f"Pose err:{pose_err:.0f}px GT err:0px", (cell_w, cell_h//2 + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    
    out = str(OUTPUT_DIR / f"warp_{i:02d}_obb{obb_err:.0f}_pose{pose_err:.0f}.jpg")
    cv2.imwrite(out, grid)
    
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{len(samples)}")

print(f"\nDone. {len(samples)} images in {OUTPUT_DIR}")
# Copy to Windows
import shutil
win_dir = Path("/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pose_obb_comparison")
win_dir.mkdir(parents=True, exist_ok=True)
for f in OUTPUT_DIR.glob("*.jpg"):
    shutil.copy(str(f), str(win_dir / f.name))
print(f"Copied to Windows: {win_dir}")
