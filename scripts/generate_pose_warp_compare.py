#!/usr/bin/env python3
"""
Compare GT quad warp vs Pose quad warp on extreme CCPD2020 green samples.
Saves individual per-sample files (no complex compositing).
"""

import csv, os, sys, math
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import parse_ccpd_quad_from_name, order_quad_points, \
    prepare_board_ocr_input_from_quad_bgr888

POSE_WEIGHT = ROOT / 'experiments/yolov8n-pos/weights/best.pt'
CCPD2020_LABEL = ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv'
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pose_quad_warp_compare')
WIN_QA.mkdir(parents=True, exist_ok=True)

print("Loading pose model...")
pose_model = YOLO(str(POSE_WEIGHT))

def angle_score(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i-1)%4]; b = p[i]; c = p[(i+1)%4]
        v1, v2 = a - b, c - b
        d = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angles.append(abs(math.degrees(math.acos(max(-1,min(1,d)))) - 90))
    return float(np.mean(angles))

# Load and sort by angle
samples = []
with open(CCPD2020_LABEL, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        p = row['img_path']
        if not os.path.exists(p):
            continue
        q = parse_ccpd_quad_from_name(p)
        if q is None:
            continue
        a = angle_score(q)
        samples.append({'path': p, 'text': row['text'], 'gt_quad': q, 'angle': a})

samples.sort(key=lambda x: -x['angle'])
qa_samples = samples[:12]
print(f"Angle range: {qa_samples[0]['angle']:.1f} ~ {qa_samples[-1]['angle']:.1f}")

for idx, s in enumerate(qa_samples):
    print(f"  [{idx+1}/12] {s['text'][:8]} angle={s['angle']:.1f}")
    
    img = cv2.imread(s['path'])
    if img is None:
        continue
    h, w = img.shape[:2]
    
    gt_quad = order_quad_points(s['gt_quad'])
    
    # Pose
    r = pose_model(s['path'], imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    pose_quad = None
    if r.keypoints is not None and r.keypoints.xy is not None:
        kps = r.keypoints.xy.cpu().numpy()
        if kps.ndim == 3 and kps.shape[0] > 0 and kps.shape[1] == 4:
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            best = int(np.argmax(confs.mean(axis=1))) if confs is not None and confs.ndim > 1 else 0
            pose_quad = order_quad_points(kps[best])
    if pose_quad is None:
        continue
    
    # Warps (original plate, no replacement)
    _, _, gt_raw, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, gt_quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='none', channel_order='bgr', quad_pad_ratio=0.0)
    gt_g3, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, gt_quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    _, _, pose_raw, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, pose_quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='none', channel_order='bgr', quad_pad_ratio=0.0)
    pose_g3, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, pose_quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    
    # Create output: top row = GT warp, bottom row = Pose warp
    # Each row: raw warp (94x24) | gray3 (94x24)
    COL_W, COL_H = 94, 24
    LABEL_H = 18
    GAP = 4
    
    # Create cell helpers
    def make_cell(img_bgr, label):
        img_h, img_w = img_bgr.shape[:2]
        cell = np.ones((img_h + LABEL_H + GAP, img_w + GAP*2, 3), dtype=np.uint8) * 30
        cell[GAP:img_h+GAP, GAP:img_w+GAP] = img_bgr
        cv2.putText(cell, label, (GAP, img_h+LABEL_H-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
        return cell
    
    gt_raw_cell = make_cell(gt_raw, 'GT raw')
    gt_g3_cell = make_cell(gt_g3, 'GT gray3')
    pose_raw_cell = make_cell(pose_raw, 'Pose raw')
    pose_g3_cell = make_cell(pose_g3, 'Pose gray3')
    
    # Overlay image with quads
    sf = min(200 / max(h, w), 1.0)
    overlay = cv2.resize(img, (int(w*sf), int(h*sf)))
    cv2.polylines(overlay, [(gt_quad*sf).round().astype(np.int32).reshape(-1,1,2)], True, (0,255,0), 2)
    cv2.polylines(overlay, [(pose_quad*sf).round().astype(np.int32).reshape(-1,1,2)], True, (255,0,0), 2)
    overlay_cell = make_cell(overlay, f'{s["text"][:8]} a={s["angle"]:.0f} G/P')
    
    # Stack: overlay | GT raw | GT gray3 | Pose raw | Pose gray3
    cells = [overlay_cell, gt_raw_cell, gt_g3_cell, pose_raw_cell, pose_g3_cell]
    
    # Compute total width precisely
    cell_widths = [c.shape[1] for c in cells]
    total_w = sum(cell_widths) + GAP * (len(cells) - 1)
    max_h = max(c.shape[0] for c in cells)
    out = np.ones((max_h, total_w, 3), dtype=np.uint8) * 20
    
    cx = 0
    for c in cells:
        ch, cw = c.shape[:2]
        y_off = (max_h - ch) // 2
        out[y_off:y_off+ch, cx:cx+cw] = c
        cx += cw + GAP
    
    win_path = str(WIN_QA / f'extreme_{idx:02d}_a{s["angle"]:.0f}.jpg')
    cv2.imwrite(win_path, out)
    
    if (idx+1) % 4 == 0:
        print(f"  ... {idx+1}/12")

print(f"\nDone. {len(qa_samples)} images in {WIN_QA}")
