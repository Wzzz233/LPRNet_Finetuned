#!/usr/bin/env python3
"""Step 0: Run Pose inference on CCPD2019 ccpd_base.
Output pose quads for use in green plate generation."""

import csv, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import parse_ccpd_quad_from_name, order_quad_points

# CCPD2019 text decoding (same as tilt/db/challenge)
CCPD2019_PROVS = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                  '苏','浙','京','闽','赣','鲁','豫','鄂','湘','粤',
                  '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
CCPD2019_ADS = ['A','B','C','D','E','F','G','H','J','K',
                'L','M','N','P','Q','R','S','T','U','V',
                'W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']

def decode_text(filename):
    stem = Path(filename).stem; parts = stem.split('-')
    if len(parts) < 6: return None
    raw = parts[4].split('_')
    if len(raw) != 7: return None
    try: codes = [int(x) for x in raw]
    except: return None
    province = CCPD2019_PROVS[codes[0]] if 0 <= codes[0] < len(CCPD2019_PROVS) else '?'
    rest = ''
    for c in codes[1:]:
        rest += CCPD2019_ADS[c] if 0 <= c < len(CCPD2019_ADS) else '?'
    return province + rest

POSE_WEIGHT = ROOT / 'experiments/yolov8n-pos/weights/best.pt'
BASE_DIR = ROOT / 'datasets/CCPD2019/ccpd_base'
OUT_DIR = ROOT / 'datasets' / 'ccpd2019_base_posquads_20260509'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading pose model...", flush=True)
model = YOLO(str(POSE_WEIGHT))
print("Model loaded.", flush=True)

# Collect all samples
all_files = sorted(BASE_DIR.glob('*.jpg'))
all_samples = []
for img_path in all_files:
    text = decode_text(img_path.name)
    if text is None or len(text) != 7: continue
    gt_quad = parse_ccpd_quad_from_name(str(img_path))
    if gt_quad is None: continue
    all_samples.append({
        'img_path': str(img_path),
        'rel_path': str(img_path.relative_to(ROOT)),
        'text': text,
        'gt_quad': gt_quad,
    })

print(f"Total valid samples: {len(all_samples)}", flush=True)

# Province distribution
from collections import Counter
provs = Counter(s['text'][0] for s in all_samples)
print(f"Province distribution:")
for p, c in provs.most_common():
    print(f"  {p}: {c}", flush=True)

results, failures = [], []
batch_size = 128
t0 = time.time()

for i in range(0, len(all_samples), batch_size):
    batch = all_samples[i:i+batch_size]
    batch_paths = [s['img_path'] for s in batch]
    try:
        r_batch = model(batch_paths, imgsz=640, conf=0.25, iou=0.5, verbose=False)
    except Exception as e:
        for s in batch:
            failures.append({'img_path': s['img_path'], 'rel_path': s['rel_path'],
                             'text': s['text'], 'reason': f'inference_error: {e}'})
        continue
    for idx, s in enumerate(batch):
        r = r_batch[idx]
        success, conf, num_dets, quad = False, 0.0, 0, None
        if r.keypoints is not None and r.keypoints.xy is not None:
            kps = r.keypoints.xy.cpu().numpy(); num_dets = len(kps)
            if kps.ndim == 3 and kps.shape[0] > 0 and kps.shape[1] == 4:
                kp_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                det_conf = kp_conf.mean(axis=1) if kp_conf is not None and kp_conf.ndim > 1 else kp_conf
                best = int(np.argmax(det_conf)) if det_conf is not None else 0
                conf = float(det_conf[best]) if det_conf is not None else 1.0
                quad = order_quad_points(kps[best])
                with Image.open(s['img_path']) as im: w, h = im.size
                xs, ys = quad[:,0], quad[:,1]
                if not (xs.min() >= -10 and xs.max() <= w+10 and ys.min() >= -10 and ys.max() <= h+10):
                    failures.append({'img_path': s['img_path'], 'rel_path': s['rel_path'],
                        'text': s['text'], 'reason': f'out_of_bounds', 'gt_quad': s['gt_quad'].tolist()})
                    continue
                results.append({'img_path': s['img_path'], 'rel_path': s['rel_path'],
                    'text': s['text'], 'gt_quad': s['gt_quad'].tolist(),
                    'pose_quad': quad.tolist(), 'confidence': round(conf,4),
                    'num_detections': int(num_dets)})
            else:
                failures.append({'img_path': s['img_path'], 'rel_path': s['rel_path'],
                    'text': s['text'], 'reason': f'bad_kps_shape: {kps.shape}'})
        else:
            failures.append({'img_path': s['img_path'], 'rel_path': s['rel_path'],
                'text': s['text'], 'reason': 'no_keypoints'})
    if (i // batch_size + 1) % 20 == 0:
        elapsed = time.time()-t0; rate = (i+len(batch))/elapsed if elapsed > 0 else 0
        print(f"  {i+len(batch)}/{len(all_samples)} ok={len(results)} fail={len(failures)} {rate:.0f}/s", flush=True)

out = OUT_DIR / 'pose_quads.jsonl'
with open(out, 'w') as f:
    for r in results: f.write(json.dumps(r, ensure_ascii=False)+'\n')
fail_out = OUT_DIR / 'pose_quads_failures.jsonl'
with open(fail_out, 'w') as f:
    for r in failures: f.write(json.dumps(r, ensure_ascii=False)+'\n')

print(f"\nResults: {len(results)}/{len(all_samples)} ({len(results)/max(len(all_samples),1)*100:.1f}%)", flush=True)
print(f"Failures: {len(failures)}", flush=True)
if failures:
    reasons = Counter(f['reason'].split(':')[0] for f in failures)
    for r, c in reasons.most_common():
        print(f"  {r}: {c}", flush=True)
print(f"Elapsed: {(time.time()-t0)/60:.1f}min", flush=True)
print("Done.", flush=True)
