#!/usr/bin/env python3
"""Compare OCR crop distribution between green_val/green_simple and cvr_val.
Sampled from each eval set, compute warp 94x24 metrics."""

import csv, json, sys, random, numpy as np, cv2
from pathlib import Path
from collections import defaultdict
from load_data import CHARS, UnifiedManifestDataset, prepare_board_ocr_input_from_quad_bgr888

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
random.seed(20260509)

OCR_PARAMS = dict(ocr_crop_mode='obb_warp', ocr_resize_mode='letterbox', ocr_resize_kernel='nn',
                  ocr_preproc='none', ocr_channel_order='bgr', ocr_quad_pad_ratio=0.0)

SAMPLES_PER_SET = 200

SETS = {
    'cvr_val': ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/val_cvreplace_v2.csv',
    'green_val': ROOT / 'manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv',
    'green_simple': ROOT / 'manifests_rebased/curriculum_gray3/test_green_simple.csv',
    'green_hard': ROOT / 'manifests_rebased/curriculum_gray3/test_green_hard.csv',
}

metrics = {}

for sname, spath in SETS.items():
    ds = UnifiedManifestDataset(str(spath), [94, 24], 8, split_filter='test',
                                 dataset_root=str(ROOT), **OCR_PARAMS)
    records = getattr(ds, 'records', [])
    indices = random.sample(range(len(ds)), min(SAMPLES_PER_SET, len(ds)))
    
    set_metrics = defaultdict(list)
    
    for idx in indices:
        sample = ds[idx]
        img_chw = sample[0]  # (3,24,94) normalized float32
        # Denormalize for display metrics
        img_vis = ((img_chw / 0.0078125) + 127.5).clip(0, 255).astype(np.uint8)
        img_hwc = img_vis.transpose(1, 2, 0)  # (24,94,3) BGR
        
        gray = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Brightness (mean of gray)
        set_metrics['brightness'].append(float(gray.mean()))
        # Contrast (std of gray)
        set_metrics['contrast'].append(float(gray.std()))
        # Sharpness
        set_metrics['sharpness'].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        
        # Green hue/saturation (only on green pixels)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        green_mask = (s > 30) & (v > 40) & (h >= 45) & (h <= 100)
        if green_mask.sum() > 10:
            set_metrics['green_hue'].append(float(h[green_mask].mean()))
            set_metrics['green_sat'].append(float(s[green_mask].mean()))
        
        # Character pixel ratio (darker pixels in gray = text)
        char_mask = gray < 100  # dark text on light background
        set_metrics['char_ratio'].append(float(char_mask.mean()))
        
        # Estimate stroke width from horizontal gradient
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges = np.abs(sobel_x) > 50
        if edges.sum() > 10:
            # Mean distance between opposite edges as stroke proxy
            set_metrics['edge_density'].append(float(edges.mean()))
        
        # Plate area from manifest
        if records and idx < len(records):
            rec = records[idx]
            try:
                q = np.array([[float(rec['quad_1x']), float(rec['quad_1y'])],
                              [float(rec['quad_2x']), float(rec['quad_2y'])],
                              [float(rec['quad_3x']), float(rec['quad_3y'])],
                              [float(rec['quad_4x']), float(rec['quad_4y'])]], dtype=np.float32)
                w_top = np.linalg.norm(q[1]-q[0]); w_bot = np.linalg.norm(q[2]-q[3])
                h_left = np.linalg.norm(q[3]-q[0]); h_right = np.linalg.norm(q[2]-q[1])
                set_metrics['plate_area'].append(float(max(w_top,w_bot)*max(h_left,h_right)))
            except:
                pass
    
    # Aggregate
    metrics[sname] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                           'p5': float(np.percentile(v,5)), 'p95': float(np.percentile(v,95))}
                       for k, v in set_metrics.items() if len(v) > 5}

# Print comparison table
print(f"{'Metric':25s} {'cvr_val':>20s} {'green_val':>20s} {'green_simple':>20s} {'green_hard':>20s}")
print('-' * 105)

metric_names = ['brightness', 'contrast', 'sharpness', 'green_hue', 'green_sat', 'char_ratio', 'edge_density', 'plate_area']
metric_labels = {
    'brightness': 'Brightness (mean gray)',
    'contrast': 'Contrast (gray std)',
    'sharpness': 'Sharpness (Lap var)',
    'green_hue': 'Green Hue (HSV)',
    'green_sat': 'Green Saturation',
    'char_ratio': 'Char pixel ratio',
    'edge_density': 'Edge density',
    'plate_area': 'Plate area (px²)',
}

for m in metric_names:
    row = f"{metric_labels.get(m,m):25s}"
    for sname in ['cvr_val', 'green_val', 'green_simple', 'green_hard']:
        if m in metrics.get(sname, {}):
            d = metrics[sname][m]
            row += f" {d['mean']:7.1f} ±{d['std']:5.1f}  [{d['p5']:6.0f}-{d['p95']:5.0f}] "
        else:
            row += f" {'N/A':>18s} "
    print(row)

# Save
json.dump(metrics, open('/tmp/ocr_crop_domain_compare.json', 'w'), ensure_ascii=False, indent=2)
print(f"\nRaw: /tmp/ocr_crop_domain_compare.json")
