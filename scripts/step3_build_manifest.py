#!/usr/bin/env python3
"""Step 3: Build new training manifest for Pose replacement training.
- Copy B2-D manifest as base
- Replace CCPD2020 real quad fields with Pose quads
- Remove old green_ccpd2020_replace_extreme sources (v1, v2, v2_additional)
- Add new green_ccpd2020_replace_pose_v3
- Keep all other data unchanged
"""

import csv, json, os, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path('/home/wzzz/LPRNet')

# ── Paths ──────────────────────────────────────────────────────────
B2D_MANIFEST = ROOT / 'manifests/curriculum_gray3_stageb_v1_B2D_paradigm3_progress'
POSE_QUADS = ROOT / 'datasets/ccpd2020_pose_quads' / 'pose_quads.jsonl'
NEW_REPLACE_TRAIN = ROOT / 'manifests/ccpd2020_replace_pose_v3' / 'train_ccpd2020_replace_pose_v3.csv'
NEW_REPLACE_VAL = ROOT / 'manifests/ccpd2020_replace_pose_v3' / 'val_ccpd2020_replace_pose_v3.csv'
OUT_DIR = ROOT / 'manifests' / 'curriculum_gray3_stageb_v1_B2D_pose_quad'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sources to REMOVE from B2-D (old replacement with wrong quad) ──
REMOVE_SOURCES = {
    'green_ccpd2020_replace_extreme',
    'green_ccpd2020_replace_extreme_v2',
    'green_ccpd2020_replace_extreme_v2_additional',
}

# ── Load Pose quads for CCPD2020 ──────────────────────────────────
print("Loading Pose quad results...")
pose_lookup = {}
failures = []
for line in open(POSE_QUADS):
    r = json.loads(line)
    pose_lookup[r['img_path']] = r['pose_quad']
failures_path = ROOT / 'datasets/ccpd2020_pose_quads' / 'pose_quads_failures.jsonl'
if failures_path.exists():
    for line in open(failures_path):
        failures.append(json.loads(line)['img_path'])
print(f"  Pose quads: {len(pose_lookup)} entries, {len(failures)} failures")

# ── Read B2-D manifest ────────────────────────────────────────────
print("\nReading B2-D manifest...")
b2d_train = []
b2d_val = []
with open(B2D_MANIFEST / 'train_B2D_paradigm3_progress.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        b2d_train.append(row)
with open(B2D_MANIFEST / 'val_B2D_paradigm3_progress.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        b2d_val.append(row)
print(f"  B2-D train: {len(b2d_train)}, val: {len(b2d_val)}")

# ── Filter & modify ───────────────────────────────────────────────
def process_rows(rows, split_name):
    kept = []
    removed_old_replace = 0
    updated_ccpd2020 = 0
    pose_fallback = 0
    
    for row in rows:
        source = row.get('source', '')
        
        # Remove old replacement sources
        if source in REMOVE_SOURCES:
            removed_old_replace += 1
            continue
        
        # For CCPD2020 real data: replace quad with Pose quad
        if source == 'ccpd2020':
            img_path = row['img_path']
            if img_path in pose_lookup:
                pq = pose_lookup[img_path]
                row['quad_1x'] = f'{pq[0][0]:.1f}'
                row['quad_1y'] = f'{pq[0][1]:.1f}'
                row['quad_2x'] = f'{pq[1][0]:.1f}'
                row['quad_2y'] = f'{pq[1][1]:.1f}'
                row['quad_3x'] = f'{pq[2][0]:.1f}'
                row['quad_3y'] = f'{pq[2][1]:.1f}'
                row['quad_4x'] = f'{pq[3][0]:.1f}'
                row['quad_4y'] = f'{pq[3][1]:.1f}'
                row['quad_source'] = 'pose_v3'
                updated_ccpd2020 += 1
            elif img_path in failures:
                # Fallback: keep GT quad, count it
                row['quad_source'] = 'gt_fallback'
                pose_fallback += 1
            else:
                # Not in lookup — keep as-is
                pose_fallback += 1
        
        kept.append(row)
    
    return kept, removed_old_replace, updated_ccpd2020, pose_fallback

train_kept, tr_removed, tr_updated, tr_fallback = process_rows(b2d_train, 'train')
val_kept, vl_removed, vl_updated, vl_fallback = process_rows(b2d_val, 'val')

print(f"\nB2-D filtering results:")
print(f"  Train: kept {len(train_kept)}, removed old replace {tr_removed}, updated CCPD2020 {tr_updated}, fallback {tr_fallback}")
print(f"  Val:   kept {len(val_kept)}, removed old replace {vl_removed}, updated CCPD2020 {vl_updated}, fallback {vl_fallback}")

# ── Append new replacement data ───────────────────────────────────
print("\nLoading new replacement data...")
new_rows_train = []
new_rows_val = []
fields = None

for csv_path, target_list in [(NEW_REPLACE_TRAIN, new_rows_train), (NEW_REPLACE_VAL, new_rows_val)]:
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if fields is None:
            fields = reader.fieldnames
        for row in reader:
            # Convert to full manifest format with quad fields
            full_row = {
                'img_path': row['img_path'],
                'text': row['text'],
                'family': row['family'],
                'source': row['source'],
                'split': row['split'],
                'semantic_group': '',
                'domain_role': '',
                'source_family': '',
                'preprocess_group': row.get('preprocess_group', 'ccpd_board'),
                'has_quad': '1',
                'can_parse_ccpd_geom': '1',
                'can_perspective': '1',
                'quad_source': 'pose_v3',
                'bbox_source': 'pose_v3',
                # Quad fields left empty — training code will parse from manifest-quad fields
                # Actually, the training expects quad fields. Look up from pose_quads.
            }
            # Look up the quad from pose_quads using the source image
            # The source image is encoded in the replaced filename
            # But we already saved pose_quad in source_log.jsonl
            # For simplicity, add placeholder quad values — training code uses ocr_crop_mode=obb_warp
            # which reads quad fields from manifest. We need actual values.
            # The new replacement manifest doesn't have quad fields.
            # We need to add them.
            target_list.append(row)

# Actually, the new replacement CSV already has the right format for the manifest
# But it lacks quad_1x..quad_4y fields. Let me check what the replacement CSV has.
print(f"\nNew replacement train fields: {list(new_rows_train[0].keys()) if new_rows_train else 'N/A'}")

# Actually, we need to merge properly. The replacement manifest was generated with
# MANIFEST_FIELDS which don't include quad_1x..quad_4y. These need to be added.
# But the training code reads quad from the manifest's quad_1x..quad_4y fields.
# If they're missing, training will fail.
#
# The replacement record has img_path, text, family, source, split, has_quad, etc.
# but missing quad coordinates. We need to add quad coordinates from the pose_quad.
#
# Actually, looking at the replacement generation: it saved pose_quad in source_log.jsonl.
# But the manifest CSV doesn't have quad fields. We need to add them.

# Quick fix: for new replacement data, load the pose quad from source_log
source_log_path = ROOT / 'datasets/ccpd2020_replace_pose_v3' / 'source_log.jsonl'
new_replace_quads = {}
if source_log_path.exists():
    for line in open(source_log_path):
        s = json.loads(line)
        new_replace_quads[s['generated_img']] = s['pose_quad']
    print(f"  Loaded {len(new_replace_quads)} replacement quads from source_log")

# Now build the final manifest
FINAL_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'semantic_group', 'domain_role', 'source_family', 'preprocess_group',
    'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source', 'bbox_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y', 'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_quad_pad_ratio',
]

def finalize_row(row, split):
    """Ensure row has all required fields, filling missing with defaults."""
    result = {}
    for f in FINAL_FIELDS:
        result[f] = row.get(f, '')
    # Ensure defaults
    for f in ['has_quad', 'can_parse_ccpd_geom', 'can_perspective']:
        if not result[f]:
            result[f] = '1'
    if not result['preprocess_group']:
        result['preprocess_group'] = 'ccpd_board'
    result['split'] = split
    return result

# Combine: filtered B2-D + new replacement
final_train = [finalize_row(r, 'train') for r in train_kept]
final_val = [finalize_row(r, 'val') for r in val_kept]

# Add new replacement data with quad fields from source_log
for csv_path, target_list, split in [
    (NEW_REPLACE_TRAIN, final_train, 'train'),
    (NEW_REPLACE_VAL, final_val, 'val'),
]:
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            full = finalize_row(row, split)
            # Add quad from lookup
            img_path = row['img_path']
            if img_path in new_replace_quads:
                pq = new_replace_quads[img_path]
                full['quad_1x'] = f'{pq[0][0]:.1f}'
                full['quad_1y'] = f'{pq[0][1]:.1f}'
                full['quad_2x'] = f'{pq[1][0]:.1f}'
                full['quad_2y'] = f'{pq[1][1]:.1f}'
                full['quad_3x'] = f'{pq[2][0]:.1f}'
                full['quad_3y'] = f'{pq[2][1]:.1f}'
                full['quad_4x'] = f'{pq[3][0]:.1f}'
                full['quad_4y'] = f'{pq[3][1]:.1f}'
            target_list.append(full)

print(f"\nFinal manifest summary:")
print(f"  Train: {len(final_train)}")
print(f"  Val:   {len(final_val)}")

# ── Source distribution ───────────────────────────────────────────
src_train = defaultdict(int)
src_val = defaultdict(int)
for r in final_train:
    src_train[r['source']] += 1
for r in final_val:
    src_val[r['source']] += 1

print(f"\nSource distribution (train):")
for s, c in sorted(src_train.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c}")
print(f"\nSource distribution (val):")
for s, c in sorted(src_val.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c}")

# ── Province distribution ─────────────────────────────────────────
from collections import Counter
prov_train = Counter(r['text'][0] for r in final_train if r['family'] in ('green8', ''))
prov_val = Counter(r['text'][0] for r in final_val if r['family'] in ('green8', ''))
print(f"\nProvince distribution (green only, train):")
for p, c in prov_train.most_common():
    print(f"  {p}: {c}")
print(f"  Total green train: {sum(prov_train.values())}")

# ── Write ─────────────────────────────────────────────────────────
train_path = OUT_DIR / 'train_pose_quad.csv'
val_path = OUT_DIR / 'val_pose_quad.csv'

with open(train_path, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
    w.writeheader()
    w.writerows(final_train)
print(f"\nWritten: {train_path} ({len(final_train)} rows)")

with open(val_path, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
    w.writeheader()
    w.writerows(final_val)
print(f"Written: {val_path} ({len(final_val)} rows)")

# ── Summary report ────────────────────────────────────────────────
print(f"\n{'=' * 50}")
print("MANIFEST BUILD REPORT")
print(f"{'=' * 50}")
print(f"  Train total:  {len(final_train)}")
print(f"  Val total:    {len(final_val)}")
print(f"  Removed old replacement: {tr_removed + vl_removed}")
print(f"  Updated CCPD2020 to Pose quad: {tr_updated + vl_updated}")
print(f"  CCPD2020 fallback (GT quad kept): {tr_fallback + vl_fallback}")
print(f"  New replacement added: {len(new_rows_train) + len(new_rows_val)}")
print(f"\n  Output: {OUT_DIR}")
