#!/usr/bin/env python3
"""Step 3: Build combined manifest for green CCPD2019 CV-replace training.
Merges: existing green E12 data + province degrade data + new CV-replace data."""

import csv, sys
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))

DATE_TAG = '20260508'
MANIFEST_DIR = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_{DATE_TAG}'
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_MANIFESTS = [
    ('green_e12', ROOT / 'manifests_rebased/unified_manifest_green_e12_replace_pose_v3_append.csv'),
    ('province_degrade', ROOT / 'manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv'),
]

NEW_TRAIN = MANIFEST_DIR / 'train_cvreplace.csv'
NEW_VAL = MANIFEST_DIR / 'val_cvreplace.csv'

OUT_TRAIN = MANIFEST_DIR / 'train_combined.csv'
OUT_VAL = MANIFEST_DIR / 'val_cvreplace.csv'  # val_cvreplace stays as-is

# Fields for output (consistent with training code)
OUTPUT_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'preprocess_group', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio',
]


def load_and_normalize(path, label=None):
    """Load manifest, normalize fields to OUTPUT_FIELDS."""
    if not path.exists():
        print(f"  SKIP {label or path}: not found", flush=True)
        return []
    
    rows = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            nr = {}
            for k in OUTPUT_FIELDS:
                nr[k] = row.get(k, '')
            # Ensure quad fields are present
            for i in range(1, 5):
                qx = f'quad_{i}x'
                qy = f'quad_{i}y'
                # Try alternate field names
                alt_x = row.get(f'x{i}', row.get(f'quad_{i}_x', ''))
                alt_y = row.get(f'y{i}', row.get(f'quad_{i}_y', ''))
                if not nr[qx] and alt_x:
                    nr[qx] = alt_x
                if not nr[qy] and alt_y:
                    nr[qy] = alt_y
            rows.append(nr)
    print(f"  Loaded {label}: {len(rows)} rows", flush=True)
    return rows


print("Loading existing green manifests...", flush=True)
existing_rows = []
for label, path in EXISTING_MANIFESTS:
    existing_rows.extend(load_and_normalize(path, label))

print(f"\nLoading new CV-replace data...", flush=True)
new_train = load_and_normalize(NEW_TRAIN, 'cv_replace_train')
new_val = load_and_normalize(NEW_VAL, 'cv_replace_val')

# ── Build combined train ─────────────────────────────────────────
all_train = []
all_train.extend(existing_rows)  # All existing data
all_train.extend(new_train)      # All new cv-replace train

# Ensure split='train' for existing data
for r in existing_rows:
    r['split'] = 'train'

# Ensure split for new data
for r in new_train:
    r['split'] = 'train'

print(f"\nCombined train: {len(all_train)} rows", flush=True)

# Province distribution
prov_cnt = Counter(r.get('text', '')[:1] for r in all_train if r.get('text'))
print(f"\nCombined train province distribution (top 10):")
for p, c in prov_cnt.most_common(10):
    print(f"  {p}: {c} ({c/len(all_train)*100:.1f}%)", flush=True)

# Source distribution
src_cnt = Counter(r.get('source', '?') for r in all_train)
print(f"\nSource distribution:")
for s, c in src_cnt.most_common():
    print(f"  {s}: {c}", flush=True)

# ── Write manifests ─────────────────────────────────────────────
with open(OUT_TRAIN, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    w.writeheader()
    w.writerows(all_train)
print(f"\nWritten: {OUT_TRAIN} ({len(all_train)} rows)", flush=True)

# Val = new cv-replace val (kept separate for new data eval)
with open(OUT_VAL, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    w.writeheader()
    w.writerows(new_val)
print(f"Written: {OUT_VAL} ({len(new_val)} rows)", flush=True)

# ── Summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"MANIFEST SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Combined train:  {len(all_train)}", flush=True)
print(f"  CV-replace val:  {len(new_val)}", flush=True)
print(f"\n  Sources in combined train:", flush=True)
for s, c in src_cnt.most_common():
    print(f"    {s}: {c}", flush=True)
print(f"\n  Province distribution (all):", flush=True)
for p, c in prov_cnt.most_common():
    print(f"    {p}: {c} ({c/len(all_train)*100:.1f}%)", flush=True)
