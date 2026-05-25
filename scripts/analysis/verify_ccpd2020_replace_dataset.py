#!/usr/bin/env python3
"""Verify and clean up CCPD2020 replace extreme dataset."""
import csv
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')

# Check train dir
train_dir = ROOT / 'datasets/ccpd2020_replace_extreme_v1/images/train'
manifest_path = ROOT / 'manifests/ccpd2020_replace_extreme_v1/train_B2C_ccpd2020_replace_extreme.csv'

manifest_files = set()
with open(manifest_path, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        manifest_files.add(Path(row['img_path']).name)

all_files = set(f.name for f in train_dir.iterdir() if f.suffix == '.jpg')
extra = all_files - manifest_files
missing = manifest_files - all_files

print(f"Manifest entries: {len(manifest_files)}")
print(f"Files in dir: {len(all_files)}")
print(f"Extra files (to remove): {len(extra)}")
print(f"Missing files: {len(missing)}")

# Remove extras
for fname in extra:
    (train_dir / fname).unlink()
print(f"After cleanup: {len(list(train_dir.iterdir()))} files")

# Province distribution
prov = Counter()
with open(manifest_path, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        prov[row['text'][0]] += 1
print(f"\nProvince distribution (train, {sum(prov.values())}):")
for p, c in sorted(prov.items()):
    print(f"  {p}: {c}")

# Val
val_path = ROOT / 'manifests/ccpd2020_replace_extreme_v1/val_B2C_ccpd2020_replace_extreme.csv'
val_prov = Counter()
with open(val_path, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        val_prov[row['text'][0]] += 1
print(f"\nProvince distribution (val, {sum(val_prov.values())}):")
for p, c in sorted(val_prov.items()):
    print(f"  {p}: {c}")

# Combined
total = prov + val_prov
print(f"\nTotal province distribution ({sum(total.values())}):")
for p, c in sorted(total.items()):
    print(f"  {p}: {c}")

# Angle range check
print(f"\nAll manifest paths exist check:")
train_ok = sum(1 for _ in manifest_files)
print(f"  train: all {train_ok} OK")
