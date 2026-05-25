#!/usr/bin/env python3
"""Step 1: Run Pose inference on CCPD2019 tilt/db/challenge subsets.
Saves pose quad results and failures per subset."""

import csv, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import parse_ccpd_quad_from_name, order_quad_points

# ── CCPD2019 text decoding ──────────────────────────────────────────
CCPD2019_PROVINCES = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
]

CCPD2019_ADS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9',
]


def decode_ccpd2019_text(filename: str) -> str:
    """Decode CCPD2017/2019 plate text from filename."""
    stem = Path(filename).stem
    parts = stem.split('-')
    if len(parts) < 6:
        return None
    raw = parts[4].split('_')
    if len(raw) != 7:
        return None
    try:
        codes = [int(x) for x in raw]
    except ValueError:
        return None
    province = CCPD2019_PROVINCES[codes[0]] if 0 <= codes[0] < len(CCPD2019_PROVINCES) else '?'
    rest = ''
    for c in codes[1:]:
        if 0 <= c < len(CCPD2019_ADS):
            rest += CCPD2019_ADS[c]
        else:
            rest += '?'
    return province + rest


# ── Config ───────────────────────────────────────────────────────────
POSE_WEIGHT = ROOT / 'experiments/yolov8n-pos/weights/best.pt'
SUBSETS = {
    'ccpd_tilt':     ROOT / 'datasets/CCPD2019/ccpd_tilt',
    'ccpd_db':       ROOT / 'datasets/CCPD2019/ccpd_db',
    'ccpd_challenge': ROOT / 'datasets/CCPD2019/ccpd_challenge',
}
DATE_TAG = time.strftime('%Y%m%d')
OUT_DIR = ROOT / 'datasets' / f'ccpd2019_tilt_db_challenge_posquads_{DATE_TAG}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load pose model ─────────────────────────────────────────────────
print("Loading pose model...", flush=True)
model = YOLO(str(POSE_WEIGHT))
print("Model loaded.", flush=True)

# ── Collect samples ─────────────────────────────────────────────────
all_samples = []
for subset_name, subset_dir in SUBSETS.items():
    img_files = sorted(subset_dir.glob('*.jpg'))
    for img_path in img_files:
        text = decode_ccpd2019_text(img_path.name)
        if text is None or len(text) != 7:
            continue
        gt_quad = parse_ccpd_quad_from_name(str(img_path))
        if gt_quad is None:
            continue
        all_samples.append({
            'img_path': str(img_path),
            'rel_path': str(img_path.relative_to(ROOT)),
            'text': text,
            'subset': subset_name,
            'gt_quad': gt_quad,
        })
    print(f"  {subset_name}: {len(img_files)} files, "
          f"{len([s for s in all_samples if s['subset']==subset_name])} valid samples", flush=True)

print(f"\nTotal valid samples: {len(all_samples)}", flush=True)

# ── Run pose inference ──────────────────────────────────────────────
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
                'rel_path': s['rel_path'],
                'text': s['text'],
                'subset': s['subset'],
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
                pose_quad = order_quad_points(kps[best])  # [TL, TR, BR, BL]
                success = True
            else:
                reason = f'bad_kps_shape: {kps.shape}'
        else:
            reason = 'no_keypoints'

        if success:
            with Image.open(s['img_path']) as im:
                w, h = im.size
            xs = pose_quad[:, 0]
            ys = pose_quad[:, 1]
            in_bounds = (xs.min() >= -10 and xs.max() <= w + 10 and
                         ys.min() >= -10 and ys.max() <= h + 10)
            if not in_bounds:
                success = False
                reason = f'out_of_bounds: x[{xs.min():.0f},{xs.max():.0f}] y[{ys.min():.0f},{ys.max():.0f}] img={w}x{h}'
                failures.append({
                    'img_path': s['img_path'],
                    'rel_path': s['rel_path'],
                    'text': s['text'],
                    'subset': s['subset'],
                    'reason': reason,
                    'gt_quad': [p.tolist() for p in s['gt_quad']],
                })
            else:
                results.append({
                    'img_path': s['img_path'],
                    'rel_path': s['rel_path'],
                    'text': s['text'],
                    'subset': s['subset'],
                    'gt_quad': [p.tolist() for p in s['gt_quad']],
                    'pose_quad': pose_quad.tolist(),
                    'confidence': round(confidence, 4),
                    'num_detections': int(num_dets),
                })
        else:
            failures.append({
                'img_path': s['img_path'],
                'rel_path': s['rel_path'],
                'text': s['text'],
                'subset': s['subset'],
                'reason': reason,
                'gt_quad': [p.tolist() for p in s['gt_quad']],
            })

    if (i // batch_size + 1) % 10 == 0:
        pct = (i + len(batch)) / len(all_samples) * 100
        print(f"  {i + len(batch)}/{len(all_samples)} ({pct:.0f}%)  "
              f"ok={len(results)} fail={len(failures)}", flush=True)

# ── Save results ────────────────────────────────────────────────────
out_path = OUT_DIR / 'pose_quads.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"\nSaved: {out_path} ({len(results)} entries)", flush=True)

fail_path = OUT_DIR / 'pose_quads_failures.jsonl'
with open(fail_path, 'w', encoding='utf-8') as f:
    for r in failures:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"Saved: {fail_path} ({len(failures)} entries)", flush=True)

# ── Summary report ──────────────────────────────────────────────────
print(f"\n{'=' * 60}", flush=True)
print(f"POSE INFERENCE REPORT — CCPD2019 tilt/db/challenge", flush=True)
print(f"{'=' * 60}", flush=True)
total_ok = len(results)
total_all = len(all_samples)
print(f"  Total samples:    {total_all}", flush=True)
print(f"  Success:          {total_ok} ({total_ok/max(total_all,1)*100:.1f}%)", flush=True)
print(f"  Failures:         {len(failures)} ({len(failures)/max(total_all,1)*100:.1f}%)", flush=True)

if failures:
    print(f"\n  Failure reasons:", flush=True)
    reasons = {}
    for f in failures:
        r = f['reason'].split(':')[0]
        reasons[r] = reasons.get(r, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r}: {c}", flush=True)

for subset in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge']:
    total = sum(1 for s in all_samples if s['subset'] == subset)
    ok = sum(1 for r in results if r['subset'] == subset)
    fail = total - ok
    pct = ok / max(total, 1) * 100
    print(f"  {subset}: {ok}/{total} ({pct:.1f}%)", flush=True)

print(f"\nDone. Results in {OUT_DIR}", flush=True)
