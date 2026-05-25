#!/usr/bin/env python3
"""Filter pose_quad val manifest to only CCPD2020 replacement data (real extreme proxy)."""
import csv
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
src = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2D_pose_quad' / 'val_pose_quad.csv'
out = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2D_pose_quad' / 'val_replace_pose_extreme.csv'

rows = []
with open(src, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row.get('source') == 'green_ccpd2020_replace_pose_v3':
            rows.append(row)

with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f'Filtered {len(rows)} replacement samples to {out}')
