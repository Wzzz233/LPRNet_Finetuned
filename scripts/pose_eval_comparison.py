#!/usr/bin/env python3
"""Full comparison: OBB vs V2b refiner vs Pose on green plates. Reports side-by-side."""

import json, sys, math, os
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────
LPRNET_ROOT = Path("/home/wzzz/LPRNet")
OBB_WEIGHT = str(LPRNET_ROOT / "datasets/downloaded_green_clone_success/best.pt")
POSE_WEIGHT = str(LPRNET_ROOT / "experiments/yolov8n-pos/weights/best.pt")
V2B_WEIGHT = str(LPRNET_ROOT / "runs/quad_refiner/true_quad_refiner_v2b_offset/best.pt")
DATASET_DIR = LPRNET_ROOT / "datasets/plate_true_quad_pose"
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = LPRNET_ROOT / "reports" / "pose_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(LPRNET_ROOT / "src"))
from quad_refiner.model import QuadHeatmapRefiner
from quad_refiner.geometry import build_patch_box_from_quad, map_quad_from_patch, gate_refined_quad
from quad_refiner.decode import (
    decode_corner_heatmaps,
    decode_corner_heatmaps_with_offset,
)

# ── Load models ────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading OBB model...")
obb_model = YOLO(OBB_WEIGHT)

print("Loading Pose model...")
pose_model = YOLO(POSE_WEIGHT)

print("Loading V2b refiner...")
refiner = QuadHeatmapRefiner(pretrained=False, enable_offset=True).to(device)
ckpt = torch.load(V2B_WEIGHT, map_location=device, weights_only=False)
refiner.load_state_dict(ckpt.get("state_dict", ckpt))
refiner.eval()

# ── Helpers ────────────────────────────────────────────────────────
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

def compute_error(pred_px, gt_px):
    errs = np.linalg.norm(pred_px - gt_px, axis=1)
    return float(errs.mean()), float(errs.max()), float(np.median(errs))

def refine_quad_simple(image, coarse_quad):
    """Run V2b refiner on an image+coarse quad."""
    h, w = image.shape[:2]
    INPUT_W, INPUT_H = 256, 128
    PAD_X, PAD_Y = 0.20, 0.25
    
    patch_box = build_patch_box_from_quad(coarse_quad, img_w=w, img_h=h, pad_x=PAD_X, pad_y=PAD_Y)
    patch = image[patch_box.y1:patch_box.y2 + 1, patch_box.x1:patch_box.x2 + 1]
    patch = cv2.resize(patch, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(patch.transpose(2, 0, 1)).float()[None] / 255.0
    
    with torch.no_grad():
        out = refiner(x.to(device))
    
    heatmaps = torch.sigmoid(out["heatmaps"])[0].cpu().numpy()
    offsets = out.get("offsets")
    if offsets is not None:
        offsets = offsets[0].cpu().numpy()
        pred_patch, confs = decode_corner_heatmaps_with_offset(heatmaps, offsets, in_w=INPUT_W, in_h=INPUT_H)
    else:
        pred_patch, confs = decode_corner_heatmaps(heatmaps, in_w=INPUT_W, in_h=INPUT_H)
    
    pred_quad = map_quad_from_patch(pred_patch, patch_box, in_w=INPUT_W, in_h=INPUT_H)
    # Gate: use refined if accepted, else fallback to coarse
    gate = gate_refined_quad(coarse_quad, pred_quad, confs,
                              patch_diag=math.hypot(INPUT_W, INPUT_H))
    final_quad = pred_quad if gate.accepted else coarse_quad
    return final_quad, gate.accepted

# ── Get green plate images ─────────────────────────────────────────
test_paths = (DATASET_DIR / "test.txt").read_text().strip().splitlines()
green_paths = [p for p in test_paths if "c2020" in p]
print(f"\nGreen plate images: {len(green_paths)}")

# ── Accumulators ───────────────────────────────────────────────────
results = {
    "obb": {"errors": [], "missed": 0},
    "v2b": {"errors": [], "missed": 0, "gate_rejected": 0},
    "pose": {"errors": [], "missed": 0},
}
per_slice = {"obb": {}, "v2b": {}, "pose": {}}

for i, rel_path in enumerate(green_paths):
    img_path = str(DATASET_DIR / rel_path)
    fname = Path(img_path).name
    label_name = fname.replace(".jpg", ".txt")
    
    # Label
    label_path = None
    for sd in ["val", "test", "train"]:
        c = LABELS_DIR / sd / label_name
        if c.exists():
            label_path = c
            break
    if label_path is None:
        continue
    
    gt_kps_norm = parse_pose_label(str(label_path))
    if gt_kps_norm is None:
        continue
    
    w, h = Image.open(img_path).size
    gt_px = gt_kps_norm * np.array([w, h])
    
    diff = "easy" if "ge" in rel_path else ("mid" if "gm" in rel_path else "hard")
    
    # ── OBB ───────────────────────────────────────────────────────
    r_obb = obb_model(img_path, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    if r_obb.obb is not None and len(r_obb.obb) > 0:
        obb_corners = r_obb.obb.xyxyxyxy.cpu().numpy()
        confs = r_obb.obb.conf.cpu().numpy()
        if len(obb_corners) > 0:
            best = int(confs.argmax())
            obb_quad_px = order_quad_points(obb_corners[best])
            err_mean, err_max, err_med = compute_error(obb_quad_px, gt_px)
            results["obb"]["errors"].append(err_mean)
            per_slice["obb"].setdefault(diff, []).append(err_mean)
        else:
            results["obb"]["missed"] += 1
            continue
    else:
        results["obb"]["missed"] += 1
        continue
    
    # ── V2b Refiner ───────────────────────────────────────────────
    img_bgr = cv2.imread(img_path)
    if img_bgr is not None:
        refined_quad_px, gate_accepted = refine_quad_simple(img_bgr, obb_quad_px)
        err_mean, err_max, err_med = compute_error(refined_quad_px, gt_px)
        results["v2b"]["errors"].append(err_mean)
        per_slice["v2b"].setdefault(diff, []).append(err_mean)
        if not gate_accepted:
            results["v2b"]["gate_rejected"] += 1
    else:
        # Use OBB as fallback for V2b
        results["v2b"]["errors"].append(
            np.mean(np.linalg.norm(obb_quad_px - gt_px, axis=1))
        )
        per_slice["v2b"].setdefault(diff, []).append(
            np.mean(np.linalg.norm(obb_quad_px - gt_px, axis=1))
        )
    
    # ── Pose ──────────────────────────────────────────────────────
    r_pose = pose_model(img_path, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    if r_pose.keypoints is not None and r_pose.keypoints.xy is not None and len(r_pose.keypoints.xy) > 0:
        pred_kps = r_pose.keypoints.xy.cpu().numpy()
        if pred_kps.ndim == 3 and pred_kps.shape[0] > 0 and pred_kps.shape[1] == 4:
            if r_pose.keypoints.conf is not None:
                kp_conf = r_pose.keypoints.conf.cpu().numpy()
                det_conf = kp_conf.mean(axis=1) if kp_conf.ndim > 1 else kp_conf
                best = int(np.argmax(det_conf))
            else:
                best = 0
            pose_quad_px = order_quad_points(pred_kps[best])
            err_mean, err_max, err_med = compute_error(pose_quad_px, gt_px)
            results["pose"]["errors"].append(err_mean)
            per_slice["pose"].setdefault(diff, []).append(err_mean)
        else:
            results["pose"]["missed"] += 1
    else:
        results["pose"]["missed"] += 1
    
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(green_paths)}")

# ── Print comparison table ─────────────────────────────────────────
print("\n" + "=" * 75)
print("GREEN PLATE CORNER ERROR COMPARISON (pixels)")
print("=" * 75)
print(f"{'Method':<8} {'Difficulty':<10} {'n':>6} {'Mean':>8} {'Median':>8} {'p90':>8} {'Max':>8} {'Missed':>7}")
print("-" * 75)

for diff_label in ["easy", "mid", "hard"]:
    for method in ["obb", "v2b", "pose"]:
        errs = np.array(per_slice[method].get(diff_label, []))
        if len(errs) == 0:
            continue
        n = len(errs)
        mean = errs.mean()
        med = np.median(errs)
        p90 = np.percentile(errs, 90)
        mx = errs.max()
        missed = results[method].get("missed", 0)
        print(f"{method:<8} {diff_label:<10} {n:>6} {mean:>8.2f} {med:>8.2f} {p90:>8.2f} {mx:>8.2f} {missed:>7}")

print("-" * 75)
# Overall
for method in ["obb", "v2b", "pose"]:
    all_e = np.array(results[method]["errors"])
    if len(all_e) > 0:
        n = len(all_e)
        mean = all_e.mean()
        med = np.median(all_e)
        p90 = np.percentile(all_e, 90)
        mx = all_e.max()
        missed = results[method].get("missed", 0)
        extra = ""
        if method == "v2b":
            extra = f"  gate_rej={results['v2b']['gate_rejected']}"
        print(f"{method:<8} {'ALL':<10} {n:>6} {mean:>8.2f} {med:>8.2f} {p90:>8.2f} {mx:>8.2f} {missed:>7}{extra}")

# ── Save ───────────────────────────────────────────────────────────
summary = {}
for method in ["obb", "v2b", "pose"]:
    summary[method] = {}
    for diff_label in ["easy", "mid", "hard"]:
        errs = np.array(per_slice[method].get(diff_label, []))
        if len(errs) > 0:
            summary[method][diff_label] = {
                "n": int(len(errs)),
                "mean_px": float(errs.mean()),
                "median_px": float(np.median(errs)),
                "p90_px": float(np.percentile(errs, 90)),
                "max_px": float(errs.max()),
            }
    all_e = np.array(results[method]["errors"])
    if len(all_e) > 0:
        summary[method]["_all"] = {
            "n": int(len(all_e)),
            "mean_px": float(all_e.mean()),
            "median_px": float(np.median(all_e)),
            "p90_px": float(np.percentile(all_e, 90)),
            "max_px": float(all_e.max()),
            "missed": results[method]["missed"],
        }
    if method == "v2b":
        summary[method]["gate_rejected"] = results["v2b"]["gate_rejected"]

json_path = OUTPUT_DIR / "comparison_green.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {json_path}")
print("Done.")
