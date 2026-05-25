#!/usr/bin/env python3
"""B2-D manifest: B1A base + B2-C extreme + B2-D additional extreme."""
import csv
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2D_paradigm3_progress'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. B1A base (no extreme)
b1a_path = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty/train_B1A.csv'
b1a_rows = []
with open(b1a_path, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        if 'green_edgefit_extreme' in row.get('source', ''):
            continue
        b1a_rows.append(row)
print(f"B1A base: {len(b1a_rows)}")

# 2. B2-C extreme rows
b2c_path = ROOT / 'manifests/ccpd2020_replace_extreme_v1/train_B2C_ccpd2020_replace_extreme.csv'
b2c_rows = []
with open(b2c_path, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        r = {k: row.get(k, '') for k in fields}
        b2c_rows.append(r)
print(f"B2-C extreme: {len(b2c_rows)}")

# 3. B2-D additional extreme rows
b2d_path = ROOT / 'manifests/ccpd2020_replace_extreme_v2_additional/train_B2D_additional.csv'
b2d_rows = []
with open(b2d_path, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        r = {k: row.get(k, '') for k in fields}
        b2d_rows.append(r)
print(f"B2-D additional: {len(b2d_rows)}")

# Write
train_path = OUT_DIR / 'train_B2D_paradigm3_progress.csv'
all_rows = b1a_rows + b2c_rows + b2d_rows
with open(train_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(all_rows)
print(f"\nCombined train: {len(all_rows)} rows -> {train_path}")

# Province dist of all extreme data
extreme = b2c_rows + b2d_rows
prov = Counter(r.get('text', '')[:1] for r in extreme)
print(f"\nTotal extreme province dist ({len(extreme)}):")
for p, c in sorted(prov.items()):
    print(f"  {p}: {c}")

# Copy val
import shutil
val_src = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty/val_B1A.csv'
val_dst = OUT_DIR / 'val_B2D_paradigm3_progress.csv'
with open(val_src, encoding='utf-8-sig') as f:
    val_rows = list(csv.DictReader(f))
with open(val_dst, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=val_rows[0].keys())
    w.writeheader()
    w.writerows(val_rows)
print(f"\nVal: {len(val_rows)} rows -> {val_dst}")
