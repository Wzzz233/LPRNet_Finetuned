#!/usr/bin/env python3
"""Run green OBB detector on CCPD2020 images, collect OBB quads."""
import csv, json, os
from pathlib import Path

import cv2
import numpy as np
import ultralytics

ROOT = Path('/home/wzzz/LPRNet')
LABEL_PATH = ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv'
MODEL_PATH = ROOT / 'datasets/downloaded_green_clone_success/best.pt'
OUT_PATH = ROOT / 'datasets/ccpd2020_replace_obbquad_v1/obb_quads.json'
IMG_DIR = ROOT / 'datasets/CCPD2020/ccpd_green/test'

print("Loading OBB model...")
model = ultralytics.YOLO(str(MODEL_PATH))

print("Loading CCPD2020 test labels...")
with open(LABEL_PATH, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
print(f"  Total: {len(rows)}")

results = []
no_det = 0
for i, row in enumerate(rows):
    path = row['img_path']
    if not os.path.exists(path):
        print(f"  [{i+1}/{len(rows)}] MISSING: {path}")
        continue
    
    img = cv2.imread(path)
    if img is None:
        print(f"  [{i+1}/{len(rows)}] CAN'T READ: {path}")
        continue
    
    out = model(img, verbose=False)[0]
    obb = out.obb
    
    if obb is not None and len(obb) > 0:
        # Take highest confidence detection
        best_idx = int(obb.conf.argmax())
        quad = obb.xyxyxyxy[best_idx].cpu().numpy().astype(np.float32)  # (4,2)
        conf = float(obb.conf[best_idx])
        results.append({
            'img_path': path,
            'obb_quad': quad.tolist(),
            'conf': conf,
        })
    else:
        no_det += 1
        results.append({
            'img_path': path,
            'obb_quad': None,
            'conf': 0.0,
        })
    
    if (i + 1) % 200 == 0:
        det_rate = (i + 1 - no_det) / (i + 1) * 100
        print(f"  [{i+1}/{len(rows)}] detected={i+1-no_det}/{i+1} ({det_rate:.1f}%)")

detected = sum(1 for r in results if r['obb_quad'] is not None)
print(f"\nDone: {detected}/{len(results)} detected, {no_det} no detection")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print(f"Saved: {OUT_PATH}")
