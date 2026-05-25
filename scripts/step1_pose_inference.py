#!/usr/bin/env python3
"""Step 1: Run Pose inference on all CCPD2020 green plates (train+test).
Saves pose quads, confidences, and failure reports."""

import csv, json, os, sys, math
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import parse_ccpd_quad_from_name, order_quad_points

POSE_WEIGHT = ROOT / 'experiments/yolov8n-pos/weights/best.pt'
OUT_DIR = ROOT / 'datasets' / 'ccpd2020_pose_quads'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_FILES = [
    ROOT / 'labels/curriculum_gray3/ccpd2020_train.csv',
    ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv',
]

print("Loading pose model...")
model = YOLO(str(POSE_WEIGHT))

def load_samples(csv_path):
    samples = []
    split = 'train' if 'train' in csv_path.stem else 'test'
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            p = row['img_path']
            if not os.path.exists(p):
                continue
            q = parse_ccpd_quad_from_name(p)
            if q is None:
                continue
            samples.append({
                'img_path': p,
                'text': row['text'],
                'gt_quad': q,
                'split': split,
            })
    return samples

all_samples = []
for lf in LABEL_FILES:
    samples = load_samples(lf)
    all_samples.extend(samples)
    print(f"  {lf.stem}: {len(samples)} samples")

print(f"\nTotal: {len(all_samples)} samples")

results = []
failures = []
batch_size = 128

for i in range(0, len(all_samples), batch_size):
    batch = all_samples[i:i+batch_size]
    batch_paths = [s['img_path'] for s in batch]
    
    try:
        r_batch = model(batch_paths, imgsz=640, conf=0.25, iou=0.5, verbose=False)
    except Exception as e:
        for s in batch:
            failures.append({
                'img_path': s['img_path'],
                'text': s['text'],
                'split': s['split'],
                'reason': f'inference_error: {e}',
            })
        continue
    
    for idx, s in enumerate(batch):
        r = r_batch[idx]
        pose_quad = None
        confidence = 0.0
        num_dets = 0
        success = False
        reason = ''
        
        if r.keypoints is not None and r.keypoints.xy is not None:
            kps = r.keypoints.xy.cpu().numpy()
            num_dets = len(kps)
            if kps.ndim == 3 and kps.shape[0] > 0 and kps.shape[1] == 4:
                if r.keypoints.conf is not None:
                    kp_conf = r.keypoints.conf.cpu().numpy()
                    det_conf = kp_conf.mean(axis=1) if kp_conf.ndim > 1 else kp_conf
                    best = int(np.argmax(det_conf))
                    confidence = float(det_conf[best])
                else:
                    best = 0
                    confidence = 1.0
                pose_quad = order_quad_points(kps[best])
                success = True
            else:
                reason = f'bad_kps_shape: {kps.shape}'
        else:
            reason = 'no_keypoints'
        
        if success:
            with Image.open(s['img_path']) as im:
                w, h = im.size
            # Validate quad is in image bounds
            xs = pose_quad[:, 0]
            ys = pose_quad[:, 1]
            in_bounds = (xs.min() >= -10 and xs.max() <= w + 10 and
                         ys.min() >= -10 and ys.max() <= h + 10)
            if not in_bounds:
                success = False
                reason = f'out_of_bounds: x[{xs.min():.0f},{xs.max():.0f}] y[{ys.min():.0f},{ys.max():.0f}] img={w}x{h}'
                failures.append({
                    'img_path': s['img_path'],
                    'text': s['text'],
                    'split': s['split'],
                    'reason': reason,
                    'gt_quad': s['gt_quad'].tolist(),
                })
            else:
                results.append({
                    'img_path': s['img_path'],
                    'text': s['text'],
                    'split': s['split'],
                    'gt_quad': s['gt_quad'].tolist(),
                    'pose_quad': pose_quad.tolist(),
                    'confidence': round(confidence, 4),
                    'num_detections': int(num_dets),
                })
        else:
            failures.append({
                'img_path': s['img_path'],
                'text': s['text'],
                'split': s['split'],
                'reason': reason,
                'gt_quad': s['gt_quad'].tolist(),
            })
    
    if (i // batch_size + 1) % 5 == 0:
        pct = (i + len(batch)) / len(all_samples) * 100
        print(f"  {i + len(batch)}/{len(all_samples)} ({pct:.0f}%)  "
              f"ok={len(results)} fail={len(failures)}")

# Save results
out_path = OUT_DIR / 'pose_quads.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"\nSaved: {out_path} ({len(results)} entries)")

fail_path = OUT_DIR / 'pose_quads_failures.jsonl'
with open(fail_path, 'w', encoding='utf-8') as f:
    for r in failures:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"Saved: {fail_path} ({len(failures)} entries)")

# Summary report
print(f"\n{'=' * 50}")
print(f"POSE INFERENCE REPORT")
print(f"{'=' * 50}")
print(f"  Total samples:    {len(all_samples)}")
print(f"  Success:          {len(results)} ({len(results)/max(len(all_samples),1)*100:.1f}%)")
print(f"  Failures:         {len(failures)} ({len(failures)/max(len(all_samples),1)*100:.1f}%)")
if failures:
    print(f"\n  Failure reasons:")
    reasons = {}
    for f in failures:
        r = f['reason'].split(':')[0]
        reasons[r] = reasons.get(r, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r}: {c}")
    print(f"\n  Failure details saved to: {fail_path}")

# Per-split stats
for split in ['train', 'test']:
    total = sum(1 for s in all_samples if s['split'] == split)
    ok = sum(1 for r in results if r['split'] == split)
    fail = total - ok
    print(f"  {split}: {ok}/{total} ({ok/max(total,1)*100:.1f}%)")

print(f"\nDone. Results in {OUT_DIR}")
