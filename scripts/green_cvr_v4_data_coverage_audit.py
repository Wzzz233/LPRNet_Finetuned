#!/usr/bin/env python3
"""v4 Data Coverage Audit — analyze source coverage, repeats, error distribution."""

import csv, json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'experiments' / 'green_cvr_v4_data_coverage_audit_20260508'
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSE_JSONL = ROOT / 'datasets/ccpd2019_tilt_db_challenge_posquads_20260508/pose_quads.jsonl'
V3_TRAIN = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v3_20260508/train_v3_balanced.csv'
V3_VAL = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/val_cvreplace_v2.csv'
BLUE_TRAIN = ROOT / 'manifests_rebased/blue_ccpd2019_tilt_db_challenge_posquad_20260508/train_posquad.csv'
BLUE_TEST = ROOT / 'manifest_rebased/blue_ccpd2019_tilt_db_challenge_posquad_20260508/test_posquad.csv'

# ── 1. All available pose sources (90,258) ──────────────────────────
print("Loading pose sources...", flush=True)
pose_sources = {}  # rel_path -> {subset, text, confidence}
with open(POSE_JSONL) as f:
    for line in f:
        r = json.loads(line)
        pose_sources[r['rel_path']] = {
            'subset': r['subset'],
            'text': r['text'],
            'confidence': r['confidence'],
            'pose_quad': r['pose_quad'],
        }
print(f"  Total pose sources: {len(pose_sources)}", flush=True)

# Per-subset
pose_subsets = Counter(r['subset'] for r in pose_sources.values())
print(f"  By subset: {dict(pose_subsets)}", flush=True)

# ── 2. V3 training sources ─────────────────────────────────────────
print("\nLoading v3 train manifest...", flush=True)
v3_train_rows = list(csv.DictReader(open(V3_TRAIN)))
v3_val_rows = list(csv.DictReader(open(V3_VAL)))

# Extract unique source image paths from v3 train (the original CCPD2019 images used)
# The img_path in v3 train manifest points to the GENERATED green plate image
# The source image is encoded in the filename (the stem of the generated image
# contains the original CCPD2019 filename)
v3_train_sources = set()
for r in v3_train_rows:
    rp = r.get('img_path', '')
    # The generated file has format: {ccpdstem}_green_{text}.jpg
    # Extract the original CCPD2019 filename (strip _green_ suffix)
    stem = Path(rp).stem
    if '_green_' in stem:
        orig_stem = stem.split('_green_')[0]
    else:
        orig_stem = stem
    v3_train_sources.add(orig_stem)

# Also look at cvreplace-specific entries
cvr_train_sources = set()
for r in v3_train_rows:
    src = r.get('source', '')
    if 'cvreplace' in src:
        rp = r.get('img_path', '')
        stem = Path(rp).stem
        if '_green_' in stem:
            orig_stem = stem.split('_green_')[0]
            cvr_train_sources.add(orig_stem)

print(f"  Total v3 train sources (unique original CCPD2019 stems in cvreplace): {len(cvr_train_sources)}", flush=True)

# ── 3. Coverage analysis ───────────────────────────────────────────
# Match v3 cvreplace sources back to pose sources
cvr_source_paths = set()
for s in cvr_train_sources:
    # Find matching pose source by looking for the stem in pose source keys
    for ps in pose_sources:
        if s in ps or Path(ps).stem == s:
            cvr_source_paths.add(ps)
            break

print(f"  Matched to pose sources: {len(cvr_source_paths)}", flush=True)
print(f"  Coverage: {len(cvr_source_paths)} / {len(pose_sources)} = {len(cvr_source_paths)/max(len(pose_sources),1)*100:.1f}%", flush=True)

# Per-subset coverage
for subset in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge']:
    total = sum(1 for r in pose_sources.values() if r['subset'] == subset)
    covered = sum(1 for ps in cvr_source_paths if pose_sources[ps]['subset'] == subset)
    print(f"  {subset}: {covered} / {total} = {covered/max(total,1)*100:.1f}%", flush=True)

# ── 4. Repeat count analysis ────────────────────────────────────────
# Count how many generated images per source
source_repeats = defaultdict(int)
for r in v3_train_rows:
    src = r.get('source', '')
    if 'cvreplace' in src:
        rp = r.get('img_path', '')
        stem = Path(rp).stem
        if '_green_' in stem:
            orig_stem = stem.split('_green_')[0]
            source_repeats[orig_stem] += 1

if source_repeats:
    repeat_values = list(source_repeats.values())
    print(f"\n  Repeat counts per source: min={min(repeat_values)} max={max(repeat_values)} mean={np.mean(repeat_values):.1f} median={np.median(repeat_values):.0f}", flush=True)
    repeat_dist = Counter(repeat_values)
    for k in sorted(repeat_dist.keys())[:10]:
        print(f"    repeated {k:2d}x: {repeat_dist[k]:5d} sources", flush=True)

# ── 5. Province distribution in v3 training ─────────────────────────
train_provs = Counter()
for r in v3_train_rows:
    t = r.get('text', '')[:1]
    if t:
        train_provs[t] += 1
print(f"\n  V3 train province distribution (top 10):")
total_train = sum(train_provs.values())
for p, c in train_provs.most_common(10):
    print(f"    {p}: {c} ({c/total_train*100:.1f}%)", flush=True)

# ── 6. Full generation potential ────────────────────────────────────
# How many images could we generate if we used ALL 90,258 pose sources
# at 1-2 per source?
print(f"\n  Full generation potential:")
print(f"    Available pose sources: {len(pose_sources)}")
print(f"    At 2 per source: {len(pose_sources)*2}")
print(f"    At 1 per source (most diverse): {len(pose_sources)}")
print(f"    Current v3 train cvreplace: {len(cvr_train_sources)} unique sources")
print(f"    Capacity remaining: {len(pose_sources) - len(cvr_train_sources)} new sources")

# Province distribution in pose sources (original CCPD2019 text)
pose_provs = Counter()
for ps in pose_sources.values():
    t = ps['text'][:1]
    if t:
        pose_provs[t] += 1
print(f"\n  Original CCPD2019 province distribution:")
total_pose = sum(pose_provs.values())
for p, c in pose_provs.most_common():
    print(f"    {p}: {c} ({c/total_pose*100:.1f}%)", flush=True)

# ── Save ────────────────────────────────────────────────────────────
report = {
    'total_pose_sources': len(pose_sources),
    'pose_by_subset': dict(pose_subsets),
    'v3_cvreplace_unique_sources': len(cvr_train_sources),
    'v3_cvreplace_coverage': len(cvr_train_sources) / max(len(pose_sources), 1),
    'coverage_by_subset': {},
    'source_repeat_stats': {
        'min': min(repeat_values) if source_repeats else 0,
        'max': max(repeat_values) if source_repeats else 0,
        'mean': float(np.mean(repeat_values)) if source_repeats else 0,
    },
    'pose_province_distribution': {p: c for p, c in pose_provs.most_common()},
    'v3_train_province_distribution': {p: c for p, c in train_provs.most_common()},
    'full_generation_potential': len(pose_sources) * 2,
    'current_unique_sources': len(cvr_train_sources),
    'remaining_sources': len(pose_sources) - len(cvr_train_sources),
}

for subset in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge']:
    total = sum(1 for r in pose_sources.values() if r['subset'] == subset)
    covered = sum(1 for ps in cvr_source_paths if pose_sources[ps]['subset'] == subset)
    report['coverage_by_subset'][subset] = {
        'total': total,
        'covered': covered,
        'pct': covered / max(total, 1) * 100,
    }

json.dump(report, open(OUT_DIR / 'coverage_report.json', 'w'), ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT_DIR / 'coverage_report.json'}", flush=True)
print("Done.", flush=True)
