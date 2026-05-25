#!/usr/bin/env python3
"""Build B2-C training manifest: B1A base - extreme + CCPD2020 replace extreme."""
import csv
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2C_paradigm3_softfreeze'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Read B1A manifest (65575 data rows)
b1a_path = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty/train_B1A.csv'
b1a_rows = []
extreme_count = 0
with open(b1a_path, encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        src = row.get('source', '')
        if 'green_edgefit_extreme' in src:
            extreme_count += 1
            continue
        b1a_rows.append(row)

print(f"B1A base rows (no extreme): {len(b1a_rows)}")
print(f"Removed extreme rows: {extreme_count}")

# 2. Read CCPD2020 replace extreme manifest
replace_path = ROOT / 'manifests/ccpd2020_replace_extreme_v1/train_B2C_ccpd2020_replace_extreme.csv'
replace_rows = []
with open(replace_path, encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    replace_fields = reader.fieldnames
    for row in reader:
        # Map to B1A field structure
        r = {k: row.get(k, '') for k in fields}
        replace_rows.append(r)

print(f"CCPD2020 replace extreme rows: {len(replace_rows)}")

# 3. Write combined train manifest
train_path = OUT_DIR / 'train_B2C_paradigm3_softfreeze.csv'
with open(train_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in b1a_rows:
        w.writerow(r)
    for r in replace_rows:
        w.writerow(r)

total = len(b1a_rows) + len(replace_rows)
print(f"\nCombined train manifest: {total} rows ({len(b1a_rows)} base + {len(replace_rows)} replace)")
print(f"Saved: {train_path}")

# 4. Copy val manifest (keep B1A val unchanged)
val_src = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty/val_B1A.csv'
val_dst = OUT_DIR / 'val_B2C_paradigm3_softfreeze.csv'
import shutil
shutil.copy2(str(val_src), str(val_dst))
print(f"Val manifest: {val_dst}")

# 5. Summary
prov = Counter()
for r in replace_rows:
    prov[r.get('text', '')[:1]] += 1
print(f"\nReplace extreme province distribution:")
for p, c in sorted(prov.items()):
    print(f"  {p}: {c}")
