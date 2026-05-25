#!/usr/bin/env python3
"""Quick audit for v4 dry-run."""
import csv, json
from pathlib import Path
from collections import Counter

train = list(csv.DictReader(open('manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4_20260508/train_cvreplace_v4.csv')))
val = list(csv.DictReader(open('manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4_20260508/val_cvreplace_v4.csv')))

print(f'Train: {len(train)}, Val: {len(val)}')

train_stems = set()
for r in train:
    stem = Path(r['img_path']).stem
    if '_green_' in stem:
        train_stems.add(stem.split('_green_')[0])
val_stems = set()
for r in val:
    stem = Path(r['img_path']).stem
    if '_green_' in stem:
        val_stems.add(stem.split('_green_')[0])

print(f'Train unique src: {len(train_stems)}')
print(f'Val unique src: {len(val_stems)}')
overlap = train_stems & val_stems
print(f'Train/val source overlap: {len(overlap)}')

train_paths = {r['img_path'] for r in train}
val_paths = {r['img_path'] for r in val}
print(f'Train/val img_path overlap: {len(train_paths & val_paths)}')

# Province balance
train_provs = Counter(r['text'][:1] for r in train if r['text'])
vals_provs = Counter(r['text'][:1] for r in val if r['text'])
print(f'Train provinces: min={min(train_provs.values())} max={max(train_provs.values())}')
print(f'Val provinces:   min={min(vals_provs.values())} max={max(vals_provs.values())}')

# Field check
required = ['img_path','text','family','source','split','preprocess_group',
    'has_quad','can_parse_ccpd_geom','can_perspective','quad_source','bbox_source',
    'quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y',
    'ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc',
    'ocr_channel_order','ocr_quad_pad_ratio']
print(f'All fields: {all(f in train[0] for f in required)}')

# Split, family
print(f'Train split: {Counter(r.get("split","") for r in train)}')
print(f'Val split:   {Counter(r.get("split","") for r in val)}')
print(f'Family:      {Counter(r.get("family","") for r in train)}')
print(f'Train subset dist: {Counter(r.get("source","") for r in train)}')

# Quad check
bad = 0
for r in train + val:
    for i in [1,2,3,4]:
        try:
            float(r[f'quad_{i}x']); float(r[f'quad_{i}y'])
        except:
            bad += 1
print(f'Bad quads: {bad}')

verdict = (len(overlap)==0 and len(train_paths & val_paths)==0 and all(f in train[0] for f in required) and bad==0)
msg = 'PASS' if verdict else 'FAIL'
print(f'Verdict: {msg}')
