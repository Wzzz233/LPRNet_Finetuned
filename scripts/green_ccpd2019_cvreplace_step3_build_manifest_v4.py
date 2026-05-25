#!/usr/bin/env python3
"""Stage 4: Build v4 balanced manifest.
Mix: cvreplace_v4 ~55%, real CCPD2020 ~30%, province_degrade ~10%, other ~5%."""

import csv, sys, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
random.seed(20260508)

DATE_TAG = '20260508'
V4_DIR = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v4_{DATE_TAG}'
V4_DIR.mkdir(parents=True, exist_ok=True)

CVR4_TRAIN = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v4_{DATE_TAG}' / 'train_cvreplace_v4.csv'
CVR4_VAL = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v4_{DATE_TAG}' / 'val_cvreplace_v4.csv'
E12_BASE = ROOT / 'manifests_rebased/unified_manifest_green_e12_replace_pose_v3_append.csv'
PROV_DEGRADE = ROOT / 'manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv'

OUT_TRAIN = V4_DIR / 'train_v4_balanced.csv'

OUTPUT_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'preprocess_group', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio',
]

PROVS = set('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新')
def get_prov(t): return t[0] if t and t[0] in PROVS else '?'

def load_clean(path, label=''):
    if not path.exists(): print(f"  SKIP {label}"); return []
    rows = [{k: r.get(k,'') for k in OUTPUT_FIELDS} for r in csv.DictReader(open(path))]
    print(f"  {label}: {len(rows)}"); return rows

def cap_per_province(rows, cap):
    by_prov = defaultdict(list)
    for r in rows: by_prov[get_prov(r['text'])].append(r)
    capped = []
    for prov, samples in by_prov.items():
        capped.extend(random.sample(samples, min(cap, len(samples))))
    return capped

print("Loading sources...", flush=True)
cvr4 = load_clean(CVR4_TRAIN, 'cvreplace_v4_train')
cvr4_val = load_clean(CVR4_VAL, 'cvreplace_v4_val')
e12_all = load_clean(E12_BASE, 'E12_base')
prov_all = load_clean(PROV_DEGRADE, 'province_degrade')

# Separate E12 into real and other by source field
e12_real = [r for r in e12_all if r.get('source','').startswith('real') or r.get('source','') == '']
e12_other = [r for r in e12_all if not (r.get('source','').startswith('real') or r.get('source','') == '')]

print(f"  E12 real: {len(e12_real)}, other: {len(e12_other)}", flush=True)

# ── Compute target sizes ────────────────────────────────────────────
n_cvr = len(cvr4)  # 91,998
# Target ratios: cvr 55%, real 30%, prov 10%, other 5%
# cvr is fixed at 92K, so total = n_cvr / 0.55
target_total = int(n_cvr / 0.55)
target_real = int(target_total * 0.30)
target_prov = int(target_total * 0.10)
target_other = int(target_total * 0.05)

print(f"Target total: {target_total}", flush=True)
print(f"Target: cvr={n_cvr} real={target_real} prov={target_prov} other={target_other}", flush=True)

# Cap real at 2000/province (enough for diversity without dominating)
real_capped = cap_per_province(e12_real, 2000)
if len(real_capped) > target_real:
    random.shuffle(real_capped)
    real_capped = real_capped[:target_real]

# Cap prov_deg at 1000/province
prov_capped = cap_per_province(prov_all, 1000)
if len(prov_capped) > target_prov:
    random.shuffle(prov_capped)
    prov_capped = prov_capped[:target_prov]

# Cap other at 500/province
other_capped = cap_per_province(e12_other, 500)
if len(other_capped) > target_other:
    random.shuffle(other_capped)
    other_capped = other_capped[:target_other]

# Combine
combined = []
combined.extend(cvr4)
combined.extend(real_capped)
combined.extend(prov_capped)
combined.extend(other_capped)
random.shuffle(combined)

print(f"\nCombined: {len(combined)}", flush=True)

# Province balance check
prov_cnt = Counter(get_prov(r['text']) for r in combined)
total = sum(prov_cnt.values())
avg = total / max(len([p for p,c in prov_cnt.items() if p != '?']), 1)
wan_pct = prov_cnt.get('皖', 0) / max(total, 1) * 100
max_ratio = max(prov_cnt.values()) / max(avg, 1)
print(f"Province: avg={avg:.0f} 皖={wan_pct:.1f}% max/avg={max_ratio:.2f}", flush=True)

# Source ratios
src_cnt = Counter(r.get('source','?') for r in combined)
src_groups = defaultdict(int)
for s, c in src_cnt.items():
    if 'cvreplace_v4' in s: src_groups['cvreplace_v4'] += c
    elif s.startswith('real') or s == '': src_groups['real'] += c
    elif 'province_degrade' in s: src_groups['province_degrade'] += c
    else: src_groups['other'] += c
total_s = sum(src_groups.values())
print(f"\nSource ratios:")
for g, c in sorted(src_groups.items()):
    print(f"  {g}: {c} ({c/total_s*100:.0f}%)", flush=True)

# Write
for r in combined: r['split'] = 'train'
with open(OUT_TRAIN, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    w.writeheader(); w.writerows(combined)
print(f"\nTrain: {OUT_TRAIN} ({len(combined)} rows)", flush=True)

# Val stays as cvr4 val (already split=test)
for r in cvr4_val: r['split'] = 'test'
out_val = V4_DIR / 'val_cvreplace_v4.csv'
with open(out_val, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    w.writeheader(); w.writerows(cvr4_val)
print(f"Val: {out_val} ({len(cvr4_val)} rows)", flush=True)
