#!/usr/bin/env python3
"""Step 4: Auto QA check on new manifest. Verify Pose quad quality."""

import csv, json, sys, random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

MANIFEST_PATH = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2D_pose_quad' / 'train_pose_quad.csv'
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pose_quad_manifest_qa')
WIN_QA.mkdir(parents=True, exist_ok=True)

random.seed(20260503)

# ── Check functions ──────────────────────────────────────────────
errors = []
samples_checked = 0

def check_quad_order(quad, label, img_path):
    """Verify quad is TL/TR/BR/BL - TL should have smallest sum."""
    q = np.array(quad, dtype=np.float32)
    sums = q.sum(axis=1)
    if sums[0] > sums[2]:
        errors.append(f"ORDER: {label} TL sum {sums[0]:.0f} > BR sum {sums[2]:.0f} in {img_path}")
        return False
    return True

def check_quad_bounds(quad, w, h, label, img_path):
    """Verify quad is within image bounds with tolerance."""
    xs, ys = quad[:, 0], quad[:, 1]
    tol = -10
    if xs.min() < tol or ys.min() < tol or xs.max() > w + 10 or ys.max() > h + 10:
        errors.append(f"BOUNDS: {label} out of bounds ({int(w)}x{int(h)}) in {img_path}")
        return False
    return True

def check_warp_quality(warp_94x24, label, img_path):
    """Verify warp is not blank, has content."""
    if warp_94x24.max() < 10 and warp_94x24.min() > -10:
        errors.append(f"BLANK: {label} warp appears blank in {img_path}")
        return False
    return True

# ── Read manifest and sample ─────────────────────────────────────
print("Reading manifest...")
rows = []
with open(MANIFEST_PATH, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
print(f"  {len(rows)} rows")

# Stratify sample: green8 + normal7
green8_rows = [r for r in rows if r['family'] == 'green8']
normal7_rows = [r for r in rows if r['family'] == 'normal7']
print(f"  green8: {len(green8_rows)}, normal7: {len(normal7_rows)}")

# Sample 6 green8 + 2 normal7
sample_green = random.sample(green8_rows, min(6, len(green8_rows)))
sample_normal = random.sample(normal7_rows, min(2, len(normal7_rows)))

for idx, row in enumerate(sample_green + sample_normal):
    img_path = row['img_path']
    family = row['family']
    text = row['text']
    source = row['source']
    
    print(f"  [{idx+1}] {family} {source} {text[:10]}...")
    
    if not Path(img_path).exists():
        errors.append(f"MISSING: {img_path}")
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        errors.append(f"UNREADABLE: {img_path}")
        continue
    h, w = img.shape[:2]
    samples_checked += 1
    
    # Parse quad from manifest
    try:
        quad = np.array([
            [float(row['quad_1x']), float(row['quad_1y'])],
            [float(row['quad_2x']), float(row['quad_2y'])],
            [float(row['quad_3x']), float(row['quad_3y'])],
            [float(row['quad_4x']), float(row['quad_4y'])],
        ], dtype=np.float32)
    except (ValueError, KeyError):
        errors.append(f"NO_QUAD: {img_path}")
        continue
    
    # Checks
    check_quad_order(quad, f'{family}/{source}', img_path)
    check_quad_bounds(quad, w, h, f'{family}/{source}', img_path)
    
    # Generate training input
    try:
        g3, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
            img, quad, 94, 24,
            resize_mode='letterbox', resize_kernel='nn',
            preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
        check_warp_quality(g3, f'{family}/{source}', img_path)
    except Exception as e:
        errors.append(f"WARP_ERROR: {img_path}: {e}")
        continue
    
    # Visual QA — save side panels
    sf = 200 / max(h, w)
    orig_small = cv2.resize(img, (int(w*sf), int(h*sf)))
    cv2.polylines(orig_small, [(quad*sf).round().astype(np.int32).reshape(-1,1,2)], True, (255,0,0), 2)
    
    g3_display = cv2.resize(g3, (188, 48))  # 2x for visibility
    
    # Create output
    out = np.zeros((max(orig_small.shape[0], 60), orig_small.shape[1] + 200, 3), dtype=np.uint8)
    out[:orig_small.shape[0], :orig_small.shape[1]] = orig_small
    out[:g3_display.shape[0], orig_small.shape[1]+10:orig_small.shape[1]+10+g3_display.shape[1]] = g3_display
    cv2.putText(out, f'{family} {source} {text}', (5, orig_small.shape[0]+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    
    fname = f'qa_{idx:02d}_{family}_{source}_{text[:6]}.jpg'
    cv2.imwrite(str(WIN_QA / fname), out)

# ── Report ────────────────────────────────────────────────────────
print(f"\n{'=' * 50}")
print("AUTO QA REPORT")
print(f"{'=' * 50}")
print(f"  Samples checked: {samples_checked}")
print(f"  Errors found:    {len(errors)}")
if errors:
    print(f"\n  Errors:")
    for e in errors:
        print(f"    {e}")
    print(f"\n  → Manual review needed")
else:
    print(f"  → All checks passed. Ready for training.")
print(f"\n  QA images: {WIN_QA}")
