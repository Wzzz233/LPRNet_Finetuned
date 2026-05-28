#!/usr/bin/env python3
"""Fast sample-based ARM color routing probe (200 samples per split)."""
import csv, os, random
from collections import defaultdict
import cv2
import numpy as np

PLATE_COLOR_UNKNOWN = 0
PLATE_COLOR_BLUE = 1
PLATE_COLOR_GREEN = 2
PLATE_COLOR_YELLOW = 3
COLOR_NAMES = {0: 'UNKNOWN', 1: 'BLUE', 2: 'GREEN', 3: 'YELLOW'}

def classify_plate_color_rgb(rgb, b):
    x1 = b[0] + (b[2] - b[0]) // 6
    x2 = b[2] - (b[2] - b[0]) // 6
    y1 = b[1] + (b[3] - b[1]) // 6
    y2 = b[3] - (b[3] - b[1]) // 6
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    total = blue_cnt = green_cnt = yellow_cnt = dark_cnt = 0
    for y in range(y1, y2+1):
        for x in range(x1, x2+1):
            p = rgb[y, x].astype(np.float32) / 255.0
            r, g, bch = p[2], p[1], p[0]
            mx = max(r, max(g, bch))
            mn = min(r, min(g, bch))
            d = mx - mn
            h_deg = 0.0
            s = 0.0 if mx == 0 else d / mx
            v = mx
            if v < 0.20: dark_cnt += 1
            if d > 1e-6:
                if mx == r: h_deg = 60.0 * ((g - bch) / d % 6.0)
                elif mx == g: h_deg = 60.0 * (((bch - r) / d) + 2.0)
                else: h_deg = 60.0 * (((r - g) / d) + 4.0)
            if h_deg < 0: h_deg += 360.0
            total += 1
            if 190.0 <= h_deg <= 260.0 and s > 0.23 and v > 0.16: blue_cnt += 1
            elif 75.0 <= h_deg <= 155.0 and s > 0.20 and v > 0.16: green_cnt += 1
            elif 15.0 <= h_deg <= 55.0 and s > 0.15 and v > 0.16: yellow_cnt += 1
    
    if total == 0: return PLATE_COLOR_UNKNOWN
    if float(blue_cnt)/total >= 0.20 and blue_cnt > green_cnt + int(0.05*total): return PLATE_COLOR_BLUE
    if float(green_cnt)/total >= 0.20 and green_cnt > blue_cnt + int(0.05*total): return PLATE_COLOR_GREEN
    if float(yellow_cnt)/total >= 0.18 and float(dark_cnt)/total < 0.50: return PLATE_COLOR_YELLOW
    return PLATE_COLOR_UNKNOWN

def bbox_from_quad(quad):
    return [int(min(quad[:,0])), int(min(quad[:,1])), int(max(quad[:,0])), int(max(quad[:,1]))]

def probe(csv_path, label, dataset_root, n_sample=200):
    rows = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    
    random.seed(42)
    sampled = random.sample(rows, min(n_sample, len(rows)))
    
    results = defaultdict(int)
    errors = 0
    for row in sampled:
        img_path = row['img_path']
        full_path = os.path.join(dataset_root, img_path)
        img = cv2.imread(full_path)
        if img is None:
            errors += 1
            continue
        try:
            quad = np.array([
                [float(row['quad_1x']), float(row['quad_1y'])],
                [float(row['quad_2x']), float(row['quad_2y'])],
                [float(row['quad_3x']), float(row['quad_3y'])],
                [float(row['quad_4x']), float(row['quad_4y'])],
            ], dtype=np.float32)
        except:
            errors += 1
            continue
        bbox = bbox_from_quad(quad)
        color = classify_plate_color_rgb(img, bbox)
        results[COLOR_NAMES[color]] += 1
    
    total = n_sample - errors
    print(f"\n{'='*60}")
    print(f"  {label} (n={total}/{n_sample}, errors={errors})")
    print(f"{'='*60}")
    for cname in ['UNKNOWN', 'BLUE', 'GREEN', 'YELLOW']:
        cnt = results.get(cname, 0)
        pct = cnt / total * 100
        bar = '#' * int(pct / 2) + '.' * (50 - int(pct / 2))
        print(f"  {cname:8s}: {cnt:4d} ({pct:5.1f}%) {bar}")
    return results

def main():
    dataset_root = '/home/wzzz/LPRNet'
    manifest_dir = '/home/wzzz/LPRNet/manifests_rebased/special_split_20260526'
    
    print("ARM classify_plate_color_rgb — empirical probe (200 samples each)")
    print("=" * 60)
    
    for fname, label in [
        ('train_police_only.csv', 'POLICE'),
        ('train_embassy_only.csv', 'EMBASSY'),
        ('train_yellow_single_routecheck_only.csv', 'YELLOW_SINGLE'),
    ]:
        probe(os.path.join(manifest_dir, fname), label, dataset_root)

if __name__ == '__main__':
    main()
