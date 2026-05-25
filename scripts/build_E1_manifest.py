#!/usr/bin/env python3
"""Build E1 manifest: extreme 60-70% of green8, + real + bridge + blue."""
import csv
from pathlib import Path
from collections import Counter
from copy import deepcopy

ROOT = Path('/home/wzzz/LPRNet')

POSE_MAN = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2D_pose_quad'
EXTREME_MAN = ROOT / 'manifests/ccpd2020_replace_extreme_v3'
OUT_DIR = ROOT / 'manifests' / 'curriculum_gray3_stageE_v1_extreme'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load sources ────────────────────────────────────────────────────
def load_csv(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    # Add missing fields for compatibility
    for r in rows:
        for f in ['has_quad','can_parse_ccpd_geom','can_perspective','quad_source','bbox_source','ocr_quad_pad_ratio']:
            r.setdefault(f, '')
        r.setdefault('semantic_group', '')
        r.setdefault('domain_role', '')
        r.setdefault('source_family', '')
        r.setdefault('preprocess_group', 'ccpd_board')
    return rows

print("Loading source manifests...")
base_train = load_csv(POSE_MAN / 'train_pose_quad.csv')
base_val = load_csv(POSE_MAN / 'val_pose_quad.csv')
extreme_train = load_csv(EXTREME_MAN / 'train_extreme_v3.csv')
extreme_val = load_csv(EXTREME_MAN / 'val_extreme_v3.csv')

print(f"Base train: {len(base_train)} ({Counter(r['family'] for r in base_train)})")
print(f"Base val:   {len(base_val)} ({Counter(r['family'] for r in base_val)})")
print(f"Extreme train: {len(extreme_train)}")
print(f"Extreme val:   {len(extreme_val)}")

# ── E1 data strategy ────────────────────────────────────────
# Extreme should be 60-70% of green8 in each batch.
# Since we can't use per-group ratios within green8, replicate
# extreme entries in the manifest to achieve ~65% batch presence.

# Remove old problematic replacement sources (reversed text)
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

# Add extreme data with replication to hit 60-70% of green8
EXTREME_REPLICATE = 30  # replicate extreme to achieve ~64% of green8 batch
for r in extreme_train:
    for _ in range(EXTREME_REPLICATE):
        train_kept.append(deepcopy(r))
for r in extreme_val:
    val_kept.append(deepcopy(r))

# Ensure all have required fields
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
        row = {}
        for f in FINAL_FIELDS:
            row[f] = r.get(f, '')
        row['split'] = split
        out.append(row)
    return out

train_final = ensure_fields(train_kept, 'train')
val_final = ensure_fields(val_kept, 'val')

# ── Stats ───────────────────────────────────────────────────────────
src_train = Counter(r['source'] for r in train_final)
src_val = Counter(r['source'] for r in val_final)
fam_train = Counter(r['family'] for r in train_final)
fam_val = Counter(r['family'] for r in val_final)
green_train = [r for r in train_final if r['family'] == 'green8']
green_val = [r for r in val_final if r['family'] == 'green8']

print(f"\n=== E1 Manifest ===")
print(f"Train: {len(train_final)}")
print(f"  Family: {dict(fam_train)}")
print(f"  Green8 sources:")
for s, c in sorted(src_train.items(), key=lambda x:-x[1]):
    if any(r['family']=='green8' and r['source']==s for r in train_final):
        print(f"    {s}: {c}")
print(f"  Extreme share of green8: {sum(1 for r in green_train if 'extreme_v3' in r['source'])}/{len(green_train)} = {sum(1 for r in green_train if 'extreme_v3' in r['source'])/max(len(green_train),1)*100:.1f}%")

print(f"\nVal: {len(val_final)}")
print(f"  Family: {dict(fam_val)}")
print(f"  Green8 sources:")
for s, c in sorted(src_val.items(), key=lambda x:-x[1]):
    if any(r['family']=='green8' and r['source']==s for r in val_final):
        print(f"    {s}: {c}")

# Province distribution (green8)
prov_train = Counter(r['text'][0] for r in green_train)
prov_val = Counter(r['text'][0] for r in green_val)
print(f"\nProvince (train, green8, {len(prov_train)} provinces):")
for p, c in prov_train.most_common(10):
    print(f"  {p}: {c}")

# ── Write ───────────────────────────────────────────────────────────
train_out = OUT_DIR / 'train_E1.csv'
val_out = OUT_DIR / 'val_E1.csv'

with open(train_out, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
    w.writeheader()
    w.writerows(train_final)
print(f"\nWritten: {train_out} ({len(train_final)} rows)")

with open(val_out, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
    w.writeheader()
    w.writerows(val_final)
print(f"Written: {val_out} ({len(val_final)} rows)")
