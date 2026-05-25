#!/usr/bin/env python3
"""Step 3 v3: Build source-balanced green training manifest.
Target ratios: cvreplace 35%, real CCPD2020 40%, province_degrade 15%, other 10%.
Province balance: max/avg ≤ 1.5, 皖 ≤ 8%."""

import csv, sys, random, math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
random.seed(20260508)

DATE_TAG = '20260508'
V3_DIR = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v3_{DATE_TAG}'
V3_DIR.mkdir(parents=True, exist_ok=True)

CVR2_TRAIN = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/train_cvreplace_v2.csv'
CVR2_VAL = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/val_cvreplace_v2.csv'
E12_BASE = ROOT / 'manifests_rebased/unified_manifest_green_e12_replace_pose_v3_append.csv'
PROV_DEGRADE = ROOT / 'manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv'

OUT_TRAIN = V3_DIR / 'train_v3_balanced.csv'
OUT_VAL = V3_DIR / 'val_v3.csv'

OUTPUT_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'preprocess_group', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio',
]

PROVINCES_SET = set('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新')

def get_prov(text):
    return text[0] if text and text[0] in PROVINCES_SET else '?'

def load_clean(path, label=''):
    if not path.exists():
        print(f"  SKIP {label}: not found", flush=True); return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            nr = {k: row.get(k, '') for k in OUTPUT_FIELDS}
            rows.append(nr)
    print(f"  Loaded {label}: {len(rows)}", flush=True)
    return rows

def cap_per_province(rows, cap):
    by_prov = defaultdict(list)
    for r in rows:
        by_prov[get_prov(r['text'])].append(r)
    capped = []
    for prov, samples in by_prov.items():
        take = min(cap, len(samples))
        capped.extend(random.sample(samples, take))
    return capped

print("Loading sources...", flush=True)
cv_replace = load_clean(CVR2_TRAIN, 'cvreplace_v2_train')
cv_val = load_clean(CVR2_VAL, 'cvreplace_v2_val')
e12_all = load_clean(E12_BASE, 'E12_base')
prov_degrade_all = load_clean(PROV_DEGRADE, 'province_degrade')

# ── Cvreplace: all retained ────────────────────────────────────
# Target 35%

# ── Real CCPD2020 green from E12: cap 2000/prov ─────────────
# E12 base contains sources: 'real', 'synthetic_exact_quad', etc.
# Filter for real data only
e12_real = [r for r in e12_all if r.get('source') in ('real', '') or 'real' in r.get('source', '')]
# Also capture any row with source starting with 'real'
e12_real = [r for r in e12_all if r.get('source', '').startswith('real') or r.get('source', '') == '']
print(f"  E12 real: {len(e12_real)}", flush=True)

# Cap at 2000 per province
e12_capped = cap_per_province(e12_real, 2000)
print(f"  E12 capped@2000/prov: {len(e12_capped)}", flush=True)

# ── Province_degrade: cap at 2000/prov ─────────────────────────
prov_capped = cap_per_province(prov_degrade_all, 2000)
print(f"  Province_degrade capped@2000/prov: {len(prov_capped)}", flush=True)

# ── Other (synthetic/edgefit): sample to 10% ──────────────────
# Classify E12 non-real as 'other'
e12_other = [r for r in e12_all if r.get('source', '').startswith('synthetic') 
             or r.get('source', '').startswith('v4_') 
             or r.get('source', '').startswith('e9_') 
             or r.get('source', '').startswith('e12_')
             or r.get('source', '').startswith('board_')]

# Also add any remaining E12 rows not classified
classified = set(id(r) for r in e12_real) | set(id(r) for r in e12_other)
remaining = [r for r in e12_all if id(r) not in classified]
print(f"  E12 other sources: {len(e12_other)}, remaining: {len(remaining)}", flush=True)

# ── Compute target sizes for ratio ────────────────────────────
n_cvr = len(cv_replace)
n_real = len(e12_capped)
n_prov = len(prov_capped)
n_other = len(e12_other)

total_no_ratio = n_cvr + n_real + n_prov + n_other
print(f"\nPre-ratio totals: cvr={n_cvr} real={n_real} prov={n_prov} other={n_other} total={total_no_ratio}", flush=True)

# Current ratios
print(f"  Current: cvr={n_cvr/total_no_ratio*100:.0f}% real={n_real/total_no_ratio*100:.0f}% "
      f"prov={n_prov/total_no_ratio*100:.0f}% other={n_other/total_no_ratio*100:.0f}%", flush=True)

# Target: 35/40/15/10. Adjust by sampling
# Use real as anchor: 40% → total = n_real / 0.4
target_total = int(n_real / 0.4)
target_cvr = int(target_total * 0.35)
target_prov = int(target_total * 0.15)
target_other = int(target_total * 0.10)

print(f"Target totals: total={target_total} cvr={target_cvr} real={n_real} prov={target_prov} other={target_other}", flush=True)

# Sample each category to target
random.shuffle(cv_replace); sampled_cvr = cv_replace[:min(target_cvr, len(cv_replace))]
random.shuffle(e12_capped); sampled_real = e12_capped  # keep all (already capped at 2K/prov)
random.shuffle(prov_capped); sampled_prov = prov_capped[:min(target_prov, len(prov_capped))]
random.shuffle(e12_other); sampled_other = e12_other[:min(target_other, len(e12_other))]

combined = sampled_cvr + sampled_real + sampled_prov + sampled_other
random.shuffle(combined)
print(f"Combined (pre-balance): {len(combined)}", flush=True)

# ── Province balance check ────────────────────────────────────
prov_cnt = Counter(get_prov(r['text']) for r in combined)
total = sum(prov_cnt.values())
avg_per_prov = total / max(len([p for p,c in prov_cnt.items() if p != '?']), 1)
max_cnt = max(prov_cnt.values()) if prov_cnt else 0
max_prov = max(prov_cnt, key=prov_cnt.get) if prov_cnt else '?'
wan_pct = prov_cnt.get('皖', 0) / max(total, 1) * 100
max_ratio = max_cnt / max(avg_per_prov, 1)

print(f"\nProvince balance: total={total} avg={avg_per_prov:.0f} max={max_prov}={max_cnt} "
      f"皖={wan_pct:.1f}% max/avg={max_ratio:.2f}", flush=True)

# If imbalance, reduce 皖 and any province with >1.5x avg
if max_ratio > 1.5 or wan_pct > 8.0:
    print("  Adjusting for province balance...", flush=True)
    # Target max per province = avg * 1.3
    target_max = avg_per_prov * 1.3
    adjusted = []
    for r in combined:
        p = get_prov(r['text'])
        if prov_cnt[p] > target_max:
            # Skip this sample randomly with probability proportional to excess
            keep_prob = target_max / prov_cnt[p]
            if random.random() > keep_prob:
                prov_cnt[p] -= 1
                continue
        adjusted.append(r)
    combined = adjusted
    random.shuffle(combined)
    
    # Re-check
    prov_cnt = Counter(get_prov(r['text']) for r in combined)
    total = sum(prov_cnt.values())
    wan_pct = prov_cnt.get('皖', 0) / max(total, 1) * 100
    avg_per_prov = total / max(len([p for p,c in prov_cnt.items() if p != '?']), 1)
    max_cnt = max(prov_cnt.values())
    max_ratio = max_cnt / max(avg_per_prov, 1)
    print(f"  After: total={total} 皖={wan_pct:.1f}% max/avg={max_ratio:.2f}", flush=True)

# Final source ratios
src_cnt = Counter(r.get('source', '?') for r in combined)
src_groups = defaultdict(int)
for s, c in src_cnt.items():
    if 'cvreplace_v2' in s: src_groups['cvreplace'] += c
    elif s.startswith('real') or s == '': src_groups['real'] += c
    elif s.startswith('province_degrade'): src_groups['province_degrade'] += c
    else: src_groups['other'] += c
total_src = sum(src_groups.values())
print(f"\nFinal source ratios:")
for g, c in sorted(src_groups.items()):
    print(f"  {g}: {c} ({c/total_src*100:.0f}%)", flush=True)

# ── Write ─────────────────────────────────────────────────────
for r in combined: r['split'] = 'train'
with open(OUT_TRAIN, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    w.writeheader(); w.writerows(combined)
print(f"\nTrain: {OUT_TRAIN} ({len(combined)} rows)", flush=True)

# Val = cvreplace v2 val (ensure split=test)
for r in cv_val: r['split'] = 'test'
with open(OUT_VAL, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    w.writeheader(); w.writerows(cv_val)
print(f"Val: {OUT_VAL} ({len(cv_val)} rows)", flush=True)

# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"V3 MANIFEST SUMMARY")
print(f"{'='*60}")
print(f"Train: {len(combined)}, Val: {len(cv_val)}")
print(f"\nProvince distribution (top):")
for p, c in prov_cnt.most_common():
    print(f"  {p}: {c} ({c/total*100:.1f}%)")
print(f"\nSource distribution (top 10):")
for s, c in src_cnt.most_common(10):
    print(f"  {s}: {c}")
