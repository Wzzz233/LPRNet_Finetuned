#!/usr/bin/env python3
"""Build E3 manifest (from E2 best) and E3-control manifest (from B2-D best).
Same data, different starting points. Both use extreme_v4 (top 35% sources)."""
import csv
from pathlib import Path
from collections import Counter
from copy import deepcopy

ROOT = Path('/home/wzzz/LPRNet')

POSE_MAN = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2D_pose_quad'
EXTREME_MAN = ROOT / 'manifests/ccpd2020_replace_extreme_v4'

def load_csv(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    for r in rows:
        for f in ['has_quad','can_parse_ccpd_geom','can_perspective','quad_source','bbox_source','ocr_quad_pad_ratio']:
            r.setdefault(f, '')
        r.setdefault('semantic_group', ''); r.setdefault('domain_role', '')
        r.setdefault('source_family', ''); r.setdefault('preprocess_group', 'ccpd_board')
    return rows

print("Loading manifests...")
base_train = load_csv(POSE_MAN / 'train_pose_quad.csv')
base_val = load_csv(POSE_MAN / 'val_pose_quad.csv')
extreme_train = load_csv(EXTREME_MAN / 'train_extreme_v3.csv')
extreme_val = load_csv(EXTREME_MAN / 'val_extreme_v3.csv')

print(f"Base train: {len(base_train)}")
print(f"Extreme v4 train: {len(extreme_train)}, val: {len(extreme_val)}")

# Remove old replacement sources
REMOVE = {'green_ccpd2020_replace_extreme', 'green_ccpd2020_replace_extreme_v2',
          'green_ccpd2020_replace_extreme_v2_additional', 'green_ccpd2020_replace_pose_v3',
          'green_ccpd2020_replace_extreme_v3'}  # also remove v3 (E1/E2 data)

def filt(rows):
    return [r for r in rows if r.get('source','') not in REMOVE]

train_kept = filt(base_train)
val_kept = filt(base_val)

# Add extreme v4 with replication (same as E2: 10x for ~40%)
EXTREME_REP = 10
for r in extreme_train:
    for _ in range(EXTREME_REP):
        train_kept.append(deepcopy(r))
val_kept.extend(deepcopy(extreme_val))

FINAL_FIELDS = [
    'img_path','text','family','source','split',
    'semantic_group','domain_role','source_family','preprocess_group',
    'has_quad','can_parse_ccpd_geom','can_perspective',
    'quad_source','bbox_source',
    'quad_1x','quad_1y','quad_2x','quad_2y',
    'quad_3x','quad_3y','quad_4x','quad_4y',
    'ocr_quad_pad_ratio',
]

def out_rows(rows, split):
    return [{**{f: r.get(f,'') for f in FINAL_FIELDS}, 'split': split} for r in rows]

train_final = out_rows(train_kept, 'train')
val_final = out_rows(val_kept, 'val')

# Fix val split to 'test' for training loader
for r in val_final:
    r['split'] = 'test'

# Stats
green_train = [r for r in train_final if r['family'] == 'green8']
extreme_count = sum(1 for r in green_train if 'extreme_v4' in r['source'])
print(f"\nTrain: {len(train_final)} total, green8={len(green_train)}")
print(f"  Extreme share of green8: {extreme_count}/{len(green_train)} = {extreme_count/len(green_train)*100:.1f}%")

# Write both manifest versions (same data, different dirs)
for suffix in ['e3_main', 'e3_control']:
    out_dir = ROOT / 'manifests' / f'curriculum_gray3_stageE_{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / f'train_{suffix}.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
        w.writeheader(); w.writerows(train_final)
    
    with open(out_dir / f'val_{suffix}.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
        w.writeheader(); w.writerows(val_final)
    
    print(f"Written: {out_dir} ({len(train_final)} train, {len(val_final)} val)")
