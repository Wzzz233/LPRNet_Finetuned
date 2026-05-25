#!/usr/bin/env python3
"""Step 2: Build posquad manifest from pose inference results.
Splits by eval subset CSVs for test."""

import csv, json, sys, time
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))

DATE_TAG = time.strftime('%Y%m%d')
POSE_DIR = ROOT / 'datasets' / f'ccpd2019_tilt_db_challenge_posquads_{DATE_TAG}'

EVAL_SUBSETS = {
    'ccpd_tilt':     ROOT / 'eval_reports/front_preprocess_ablation/eval_subset_ccpd_tilt.csv',
    'ccpd_db':       ROOT / 'eval_reports/front_preprocess_ablation/eval_subset_ccpd_db.csv',
    'ccpd_challenge': ROOT / 'eval_reports/front_preprocess_ablation/eval_subset_ccpd_challenge.csv',
}

OUT_DIR = ROOT / 'manifests_rebased' / f'blue_ccpd2019_tilt_db_challenge_posquad_{DATE_TAG}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'preprocess_group', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio',
]

# ── Load eval subset filenames (held-out test) ─────────────────────
eval_subset_fnames = {}
for subset_name, csv_path in EVAL_SUBSETS.items():
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        names = set(row.get('filename', '') for row in reader)
        eval_subset_fnames[subset_name] = names
        print(f"  {subset_name}: {len(names)} eval filenames", flush=True)

all_eval_fnames = set()
for names in eval_subset_fnames.values():
    all_eval_fnames.update(names)
print(f"  Total unique eval filenames: {len(all_eval_fnames)}", flush=True)

# ── Load pose results ──────────────────────────────────────────────
print("\nLoading pose results...", flush=True)
pose_data = []
with open(POSE_DIR / 'pose_quads.jsonl', encoding='utf-8') as f:
    for line in f:
        pose_data.append(json.loads(line))
print(f"  {len(pose_data)} entries loaded", flush=True)

# ── Split → train/test ────────────────────────────────────────────
print("\nSplitting into train/test...", flush=True)
train_entries = []
test_entries = []

for entry in pose_data:
    fname = Path(entry['img_path']).name
    if fname in all_eval_fnames:
        test_entries.append(entry)
    else:
        train_entries.append(entry)

print(f"  Train: {len(train_entries)}, Test: {len(test_entries)}", flush=True)

def entry_to_row(entry, split):
    pq = entry['pose_quad']
    return {
        'img_path': entry['rel_path'],
        'text': entry['text'],
        'family': 'normal7',
        'source': f'ccpd2019_{entry["subset"]}',
        'split': split,
        'preprocess_group': 'ccpd_board',
        'has_quad': '1',
        'can_parse_ccpd_geom': '0',
        'can_perspective': '1',
        'quad_source': 'pose_yolov8n',
        'quad_1x': f'{pq[0][0]:.1f}',
        'quad_1y': f'{pq[0][1]:.1f}',
        'quad_2x': f'{pq[1][0]:.1f}',
        'quad_2y': f'{pq[1][1]:.1f}',
        'quad_3x': f'{pq[2][0]:.1f}',
        'quad_3y': f'{pq[2][1]:.1f}',
        'quad_4x': f'{pq[3][0]:.1f}',
        'quad_4y': f'{pq[3][1]:.1f}',
        'ocr_crop_mode': 'obb_warp',
        'ocr_resize_mode': 'letterbox',
        'ocr_resize_kernel': 'nn',
        'ocr_preproc': 'none',
        'ocr_channel_order': 'bgr',
        'ocr_quad_pad_ratio': '0.0',
    }

def write_csv(entries, split, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for entry in entries:
            w.writerow(entry_to_row(entry, split))

# ── Write files ────────────────────────────────────────────────────
train_path = OUT_DIR / 'train_posquad.csv'
write_csv(train_entries, 'train', train_path)
print(f"  Wrote train: {train_path} ({len(train_entries)} rows)", flush=True)

test_path = OUT_DIR / 'test_posquad.csv'
write_csv(test_entries, 'test', test_path)
print(f"  Wrote test: {test_path} ({len(test_entries)} rows)", flush=True)

all_entries = train_entries + test_entries
combined_path = OUT_DIR / 'all_posquad.csv'
write_csv(all_entries, '', combined_path)
print(f"  Wrote combined: {combined_path} ({len(all_entries)} rows)", flush=True)

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}", flush=True)
print(f"MANIFEST SUMMARY", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"  Total:  {len(all_entries)}", flush=True)
print(f"  Train:  {len(train_entries)} ({len(train_entries)/max(len(all_entries),1)*100:.1f}%)", flush=True)
print(f"  Test:   {len(test_entries)} ({len(test_entries)/max(len(all_entries),1)*100:.1f}%)", flush=True)

train_sources = Counter(e['subset'] for e in train_entries)
test_sources = Counter(e['subset'] for e in test_entries)
print(f"\n  Train by subset:", flush=True)
for k, v in sorted(train_sources.items()):
    print(f"    {k}: {v}", flush=True)
print(f"  Test by subset:", flush=True)
for k, v in sorted(test_sources.items()):
    print(f"    {k}: {v}", flush=True)

print(f"\nOutput dir: {OUT_DIR}", flush=True)
