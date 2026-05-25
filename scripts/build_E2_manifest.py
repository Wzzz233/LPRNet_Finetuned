#!/usr/bin/env python3
"""Build E2 manifest: extreme 35-40%, with real/bridge/blue recovery."""
import csv
from pathlib import Path
from collections import Counter
from copy import deepcopy

ROOT = Path('/home/wzzz/LPRNet')

POSE_MAN = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2D_pose_quad'
EXTREME_MAN = ROOT / 'manifests/ccpd2020_replace_extreme_v3'
OUT_DIR = ROOT / 'manifests' / 'curriculum_gray3_stageE_v2_balanced'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    for r in rows:
        for f in ['has_quad','can_parse_ccpd_geom','can_perspective','quad_source','bbox_source','ocr_quad_pad_ratio']:
            r.setdefault(f, '')
        r.setdefault('semantic_group', '')
        r.setdefault('domain_role', '')
        r.setdefault('source_family', '')
        r.setdefault('preprocess_group', 'ccpd_board')
    return rows

print("Loading manifests...")
base_train = load_csv(POSE_MAN / 'train_pose_quad.csv')
base_val = load_csv(POSE_MAN / 'val_pose_quad.csv')
extreme_train = load_csv(EXTREME_MAN / 'train_extreme_v3.csv')
extreme_val = load_csv(EXTREME_MAN / 'val_extreme_v3.csv')

print(f"Base train: {len(base_train)} fam={dict(Counter(r['family'] for r in base_train))}")
print(f"Extreme train: {len(extreme_train)}")
print(f"Extreme val:   {len(extreme_val)}")

# Remove old replacement sources (reversed text)
REMOVE_SOURCES = {
    'green_ccpd2020_replace_extreme',
    'green_ccpd2020_replace_extreme_v2',
    'green_ccpd2020_replace_extreme_v2_additional',
    'green_ccpd2020_replace_pose_v3',
}

def filter_rows(rows):
    return [r for r in rows if r.get('source','') not in REMOVE_SOURCES]

train_kept = filter_rows(base_train)
val_kept = filter_rows(base_val)

# Add extreme data with lower replication for E2 (target ~37% of green8)
EXTREME_REPLICATE = 10
for r in extreme_train:
    for _ in range(EXTREME_REPLICATE):
        train_kept.append(deepcopy(r))
val_kept.extend(deepcopy(extreme_val))

FINAL_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'semantic_group', 'domain_role', 'source_family', 'preprocess_group',
    'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source', 'bbox_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_quad_pad_ratio',
]

def ensure_fields(rows, split):
    out = []
    for r in rows:
        row = {f: r.get(f, '') for f in FINAL_FIELDS}
        row['split'] = split
        out.append(row)
    return out

train_final = ensure_fields(train_kept, 'train')
val_final = ensure_fields(val_kept, 'val')

# Stats
green_train = [r for r in train_final if r['family'] == 'green8']
extreme_in_green = sum(1 for r in green_train if 'extreme_v3' in r['source'])
print(f"\n=== E2 Manifest ===")
print(f"Train total: {len(train_final)}")
print(f"  Families: {dict(Counter(r['family'] for r in train_final))}")
print(f"  Extreme share of green8: {extreme_in_green}/{len(green_train)} = {extreme_in_green/max(len(green_train),1)*100:.1f}%")
print(f"Val total: {len(val_final)}")

# Write
for split, rows, name in [('train', train_final, 'train_E2.csv'), ('val', val_final, 'val_E2.csv')]:
    out_path = OUT_DIR / name
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {out_path} ({len(rows)} rows)")
