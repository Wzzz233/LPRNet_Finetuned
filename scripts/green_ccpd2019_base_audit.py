#!/usr/bin/env python3
"""Quick audit for base generation."""
import csv, os
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
train = list(csv.DictReader(open(ROOT/'manifests_rebased/green_ccpd2019_base_cvreplace_posquad_v1_20260509/train_base_cvreplace_posquad.csv')))
val = list(csv.DictReader(open(ROOT/'manifests_rebased/green_ccpd2019_base_cvreplace_posquad_v1_20260509/val_base_cvreplace_posquad.csv')))

print(f'Train rows: {len(train)}, Val rows: {len(val)}')
r0 = train[0]
print(f'Fields: {list(r0.keys())}')
print(f'family={r0["family"]} has_quad={r0["has_quad"]} quad_source={r0["quad_source"]}')
print(f'preproc={r0["preprocess_group"]} crop={r0["ocr_crop_mode"]} preproc_mode={r0["ocr_preproc"]}')
print(f'channel={r0["ocr_channel_order"]} pad={r0["ocr_quad_pad_ratio"]}')

# Overlap
train_stems = {r['img_path'].split('/')[-1].split('_green_')[0] for r in train}
val_stems = {r['img_path'].split('/')[-1].split('_green_')[0] for r in val}
print(f'Source overlap: {len(train_stems & val_stems)} (must be 0)')

# Missing files
missing_train = sum(1 for r in train if not (ROOT/r['img_path']).exists())
missing_val = sum(1 for r in val if not (ROOT/r['img_path']).exists())
print(f'Missing images: train={missing_train} val={missing_val}')

# Province counts
from collections import Counter
tp = Counter(r['text'][0] for r in train)
vp = Counter(r['text'][0] for r in val)
print(f'Train provinces: min={min(tp.values())} max={max(tp.values())}')
print(f'Val provinces:   min={min(vp.values())} max={max(vp.values())}')

# Quad check
bad_q = 0
for r in train + val:
    for i in range(1,5):
        try:
            float(r[f'quad_{i}x']); float(r[f'quad_{i}y'])
        except: bad_q += 1
print(f'Bad quads: {bad_q}')
print('Audit: PASS' if (len(train_stems & val_stems)==0 and missing_train==0 and missing_val==0 and bad_q==0) else 'Audit: FAIL')
