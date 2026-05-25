#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 evaluation: run pose model on val set, compare predicted vs GT keypoints.

Outputs:
  reports/pose_eval/corner_error_stats.json  — per-slice corner error
  reports/pose_eval/val_predictions.jsonl    — per-image corner errors + metadata
  reports/pose_eval/vis/                     — visual comparison montages
"""

import json
import os
import sys
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────
POSE_WEIGHT = "/home/wzzz/LPRNet/experiments/yolov8n-pos/weights/best.pt"
LPRNET_ROOT = Path("/home/wzzz/LPRNet")
DATASET_DIR = LPRNET_ROOT / "datasets" / "plate_true_quad_pose"
VAL_TXT = DATASET_DIR / "val.txt"
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = LPRNET_ROOT / "reports" / "pose_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "vis").mkdir(exist_ok=True)
(OUTPUT_DIR / "vis_failures").mkdir(exist_ok=True)

# ── Label parsing ──────────────────────────────────────────────────
def parse_pose_label(label_path: str) -> np.ndarray:
    """Parse YOLO pose label file. Returns (4,2) normalized keypoints."""
    content = open(label_path).read().strip()
    parts = content.split()
    # Format: class cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
    coords = list(map(float, parts[5:]))
    kps = []
    for i in range(0, len(coords), 3):
        kps.append((coords[i], coords[i + 1]))
    return np.array(kps, dtype=np.float32)


def compute_corner_error(pred_kps: np.ndarray, gt_kps: np.ndarray, img_w: int, img_h: int) -> dict:
    """
    Compute per-corner and mean corner error in pixels.
    pred_kps, gt_kps: (4, 2) in normalized [0, 1] coordinates.
    """
    pred_px = pred_kps * np.array([img_w, img_h], dtype=np.float32)
    gt_px = gt_kps * np.array([img_w, img_h], dtype=np.float32)
    errors = np.linalg.norm(pred_px - gt_px, axis=1)
    return {
        "corner_errors_px": errors.tolist(),
        "mean_error_px": float(errors.mean()),
        "max_error_px": float(errors.max()),
        "median_error_px": float(np.median(errors)),
    }


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Reorder to TL/TR/BR/BL using geometric heuristic."""
    quad = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = quad.sum(axis=1)
    diffs = np.diff(quad, axis=1).reshape(-1)
    ordered[0] = quad[np.argmin(sums)]
    ordered[2] = quad[np.argmax(sums)]
    ordered[1] = quad[np.argmin(diffs)]
    ordered[3] = quad[np.argmax(diffs)]
    return ordered


def infer_slice_from_label(label_path: str) -> str:
    """Extract slice tag from label filename (e.g., 'c2019_nb', 'c2020_gh')."""
    stem = Path(label_path).stem
    return "_".join(stem.split("_")[:2])  # e.g., "c2019_nb"


def infer_image_size(img_path: str) -> tuple:
    """Get image dimensions from file."""
    with Image.open(img_path) as im:
        return im.size  # (width, height)


# ── Main ───────────────────────────────────────────────────────────
def main():
    print("Loading pose model...")
    model = YOLO(POSE_WEIGHT)

    # Read val images — keep relative paths, do NOT resolve symlinks
    val_lines = VAL_TXT.read_text().strip().splitlines()
    # Use the path as stored in the dataset (relative to DATASET_DIR)
    # Important: do NOT resolve symlinks — the filename in labels/ uses the
    # short ID (e.g., "c2019_nb_00305705.jpg"), not the original CCPD filename.
    abs_val_paths = [str(DATASET_DIR / line) for line in val_lines]

    print(f"Val set: {len(abs_val_paths)} images")

    # Results accumulator
    all_results = []
    slice_stats = defaultdict(lambda: {"errors": [], "count": 0})

    batch_size = 128
    n_batches = math.ceil(len(abs_val_paths) / batch_size)

    for batch_idx in range(n_batches):
        batch_paths = abs_val_paths[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        # Run inference
        results = model(batch_paths, imgsz=640, conf=0.25, iou=0.5, device=0, verbose=False)

        for i, r in enumerate(results):
            img_path = batch_paths[i]
            img_w, img_h = infer_image_size(img_path)

            # GT label path
            fname = Path(img_path).name
            label_name = fname.replace(".jpg", ".txt").replace(".png", ".txt")
            # Find label (it could be in train/val/test labels dir)
            label_path = None
            for split_dir in ["val", "test", "train"]:
                candidate = LABELS_DIR / split_dir / label_name
                if candidate.exists():
                    label_path = candidate
                    break
            if label_path is None:
                continue

            gt_kps_norm = parse_pose_label(str(label_path))
            slice_name = infer_slice_from_label(str(label_path))

            # Extract predictions (take highest confidence detection)
            pred_kps_norm = None
            pred_conf = 0
            num_detections = 0
            if r.keypoints is not None and r.keypoints.xy is not None:
                num_detections = len(r.keypoints.xy)
                if num_detections > 0:
                    # keypoints.xy shape: (n_detections, 4, 2)
                    # keypoints.conf shape: (n_detections, 4)  — per-keypoint conf
                    # Use mean keypoint confidence as detection confidence
                    if r.keypoints.conf is not None:
                        kp_conf = r.keypoints.conf.cpu().numpy()
                        det_confidences = kp_conf.mean(axis=1) if kp_conf.ndim > 1 else kp_conf
                        best_idx = int(np.argmax(det_confidences))
                        pred_conf = float(det_confidences[best_idx])
                    else:
                        best_idx = 0
                        pred_conf = 1.0
                    pred_kps = r.keypoints.xy[best_idx].cpu().numpy()  # (4, 2) in pixels
                    pred_kps_norm = pred_kps / np.array([img_w, img_h])

            # Ensure order TL/TR/BR/BL
            if pred_kps_norm is not None and len(pred_kps_norm) == 4:
                pred_kps_norm = order_quad_points(pred_kps_norm)

            # Compute error
            entry = {
                "img_path": img_path,
                "slice": slice_name,
                "num_detections": num_detections,
                "pred_conf": pred_conf,
                "gt_kps_norm": gt_kps_norm.tolist(),
            }

            if pred_kps_norm is not None and len(pred_kps_norm) == 4 and np.all(pred_kps_norm >= 0):
                corner_err = compute_corner_error(pred_kps_norm, gt_kps_norm, img_w, img_h)
                entry.update({
                    "pred_kps_norm": pred_kps_norm.tolist(),
                    "mean_error_px": corner_err["mean_error_px"],
                    "max_error_px": corner_err["max_error_px"],
                    "median_error_px": corner_err["median_error_px"],
                    "corner_errors_px": corner_err["corner_errors_px"],
                })
                slice_stats[slice_name]["errors"].append(corner_err["mean_error_px"])
            else:
                entry.update({
                    "pred_kps_norm": None,
                    "mean_error_px": None,
                    "max_error_px": None,
                    "median_error_px": None,
                    "corner_errors_px": None,
                })

            slice_stats[slice_name]["count"] += 1
            all_results.append(entry)

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{n_batches} ({len(all_results)} images processed)")

    # ── Compute summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CORNER ERROR SUMMARY (pixels)")
    print("=" * 70)

    summary = {}
    for slice_name in sorted(slice_stats.keys()):
        stats = slice_stats[slice_name]
        errors = np.array(stats["errors"])
        n = len(errors)
        n_missed = stats["count"] - n
        summary[slice_name] = {
            "count": stats["count"],
            "with_prediction": n,
            "missed": n_missed,
            "missed_pct": round(100 * n_missed / max(stats["count"], 1), 1),
            "mean_error_px": float(errors.mean()) if n > 0 else None,
            "median_error_px": float(np.median(errors)) if n > 0 else None,
            "std_error_px": float(errors.std()) if n > 0 else None,
            "p90_error_px": float(np.percentile(errors, 90)) if n > 0 else None,
            "max_error_px": float(errors.max()) if n > 0 else None,
            "min_error_px": float(errors.min()) if n > 0 else None,
        }
        print(f"\n  {slice_name:<20} n={stats['count']:>5}  "
              f"mean={float(errors.mean()):.2f}px  "
              f"median={float(np.median(errors)):.2f}px  "
              f"p90={float(np.percentile(errors, 90)):.2f}px  "
              f"max={float(errors.max()):.2f}px  "
              f"missed={n_missed}")

    # Overall
    all_errors = []
    for slice_name in slice_stats:
        all_errors.extend(slice_stats[slice_name]["errors"])
    all_errors = np.array(all_errors)
    summary["_overall"] = {
        "total_images": sum(s["count"] for s in slice_stats.values()),
        "total_with_prediction": len(all_errors),
        "total_missed": sum(s["count"] for s in slice_stats.values()) - len(all_errors),
        "mean_error_px": float(all_errors.mean()),
        "median_error_px": float(np.median(all_errors)),
        "p90_error_px": float(np.percentile(all_errors, 90)),
        "max_error_px": float(all_errors.max()),
        "std_error_px": float(all_errors.std()),
    }
    print(f"\n  {'OVERALL':<20} n={sum(s['count'] for s in slice_stats.values()):>5}  "
          f"mean={float(all_errors.mean()):.2f}px  "
          f"median={float(np.median(all_errors)):.2f}px  "
          f"p90={float(np.percentile(all_errors, 90)):.2f}px  "
          f"max={float(all_errors.max()):.2f}px  "
          f"missed={sum(s['count'] for s in slice_stats.values()) - len(all_errors)}")

    # Save results
    json_path = OUTPUT_DIR / "corner_error_stats.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")

    jsonl_path = OUTPUT_DIR / "val_predictions.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved: {jsonl_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
