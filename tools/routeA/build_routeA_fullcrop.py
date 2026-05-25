#!/usr/bin/env python3
"""Phase 5B: Export full-crop plate images from current R50 data pool.

Reads each row from the balanced R50 manifest, warp-crops the plate region
using quad coordinates, resizes to full-crop size (171x64), saves as PNG.

Output:
  datasets/routeA_fullcrop_20260512/images/train/*.png
  manifests_rebased/routeA_firstchar_r50_20260512/train_r50_fullcrop_bal31_v1.csv
  manifests_rebased/routeA_firstchar_r50_20260512/test_real_holdout_fullcrop_v1.csv
  manifests_rebased/routeA_firstchar_r50_20260512/test_province_stress_fullcrop_v1.csv

Usage:
  python tools/routeA/build_routeA_fullcrop.py
"""
import csv, json, os, sys, cv2, numpy as np
from pathlib import Path
from collections import Counter
import argparse

ROOT = Path('/home/wzzz/LPRNet')
OUT_IMG_DIR = ROOT / 'datasets/routeA_fullcrop_20260512/images/train'
OUT_MANIFEST_DIR = ROOT / 'manifests_rebased/routeA_firstchar_r50_20260512'
IN_DIR = ROOT / 'manifests_rebased/routeA_firstchar_r50_20260512'

# Full-crop target size (matches train_tiny_province_net.py full_crop defaults)
TARGET_H = 64
TARGET_W = int(round(TARGET_H * 128 / 48))  # 171

def warp_crop_plate(img, row):
    """Extract plate region using quad coordinates from manifest row."""
    try:
        pts = np.array([
            [float(row['quad_1x']), float(row['quad_1y'])],
            [float(row['quad_2x']), float(row['quad_2y'])],
            [float(row['quad_3x']), float(row['quad_3y'])],
            [float(row['quad_4x']), float(row['quad_4y'])],
        ], dtype='float32')
    except (KeyError, ValueError):
        return None

    # Get bounding rect for crop
    x_min = max(0, int(pts[:, 0].min()))
    y_min = max(0, int(pts[:, 1].min()))
    x_max = min(img.shape[1], int(pts[:, 0].max()))
    y_max = min(img.shape[0], int(pts[:, 1].max()))

    if x_max <= x_min or y_max <= y_min:
        return None

    # Crop and warp to target size
    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    # Perspective warp to straighten the plate
    dst_pts = np.array([
        [0, 0],
        [TARGET_W - 1, 0],
        [TARGET_W - 1, TARGET_H - 1],
        [0, TARGET_H - 1],
    ], dtype='float32')

    # Adjust source points relative to crop
    src_pts = pts.copy()
    src_pts[:, 0] -= x_min
    src_pts[:, 1] -= y_min

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(crop, matrix, (TARGET_W, TARGET_H))
    return warped


def process_manifest(in_csv, out_csv, out_dir, max_n=None):
    """Process a manifest: export full-crop images and create new manifest."""
    rows = []
    exported = 0
    skipped = 0
    failed_paths = []

    fieldnames = None
    with open(in_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    success_count = 0

    for idx, row in enumerate(rows):
        if max_n and idx >= max_n:
            break

        img_path = row.get('img_path', '').strip()
        text = row.get('text', '').strip()

        if not img_path or not text:
            skipped += 1
            continue

        # Resolve full path
        full_path = img_path
        if not os.path.isabs(img_path):
            full_path = str(ROOT / img_path)
        if not Path(full_path).exists():
            failed_paths.append(img_path)
            skipped += 1
            continue

        img = cv2.imread(full_path)
        if img is None:
            skipped += 1
            continue

        # Warp-crop plate region
        plate = warp_crop_plate(img, row)
        if plate is None:
            skipped += 1
            continue

        # Save as PNG (lossless)
        safe_name = f'{idx:08d}_{text}.png'.replace('/', '_')
        out_path = out_dir / safe_name
        cv2.imwrite(str(out_path), plate)

        # Create new row
        new_row = dict(row)
        new_row['img_path'] = str(out_path)
        out_rows.append(new_row)
        success_count += 1

        if success_count % 1000 == 0:
            print(f'  Exported {success_count}/{len(rows)} (skipped {skipped})')

    # Write new manifest
    out_rows = out_rows[:max_n] if max_n else out_rows
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f'  Done: {success_count} exported, {skipped} skipped, {len(failed_paths)} path errors')
    return {
        'total_input': len(rows),
        'exported': success_count,
        'skipped': skipped,
        'path_errors': len(failed_paths),
        'fieldnames': fieldnames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_n', type=int, default=None, help='Limit rows for testing')
    args = ap.parse_args()

    print('=' * 60)
    print('Phase 5B: Export full-crop plate images')
    print('=' * 60)
    print(f'Target size: {TARGET_W}x{TARGET_H}')

    OUT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Balanced train set
    print('\n[1] Processing balanced train manifest...')
    in_csv = IN_DIR / 'train_r50_bal31_v1.csv'
    out_csv = IN_DIR / 'train_r50_fullcrop_bal31_v1.csv'
    img_dir = OUT_IMG_DIR
    stats = process_manifest(in_csv, out_csv, img_dir, max_n=args.max_n)
    print(f'  Train fullcrop: {stats["exported"]} images -> {out_csv}')

    # 2. Real holdout test
    print('\n[2] Processing real holdout test...')
    in_csv = IN_DIR / 'test_real_holdout_v1.csv'
    out_csv = IN_DIR / 'test_real_holdout_fullcrop_v1.csv'
    img_dir_holdout = ROOT / 'datasets/routeA_fullcrop_20260512/images/test'
    stats2 = process_manifest(in_csv, out_csv, img_dir_holdout, max_n=args.max_n)
    print(f'  Holdout fullcrop: {stats2["exported"]} images -> {out_csv}')

    # 3. Province stress test
    print('\n[3] Processing province stress test...')
    in_csv = IN_DIR / 'test_province_stress_v1.csv'
    out_csv = IN_DIR / 'test_province_stress_fullcrop_v1.csv'
    stats3 = process_manifest(in_csv, out_csv, img_dir_holdout, max_n=args.max_n)
    print(f'  Province stress fullcrop: {stats3["exported"]} images -> {out_csv}')

    # Summary
    print(f'\n{"=" * 60}')
    print('Export complete')
    print(f'  Train: {stats.get("exported", 0)}')
    print(f'  Holdout: {stats2.get("exported", 0)}')
    print(f'  Province stress: {stats3.get("exported", 0)}')
    print(f'  Output dir: {OUT_IMG_dir}' if 'OUT_IMG_dir' in dir() else f'  Output dir: {OUT_IMG_DIR}')


if __name__ == '__main__':
    main()
