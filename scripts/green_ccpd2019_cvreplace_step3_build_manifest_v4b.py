#!/usr/bin/env python3
"""v4b rebalanced manifest: cvr45%, real40%, prov10%, other5%.
Reuses v4 images, only resamples existing data."""

import csv, sys, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
random.seed(20260509)

V4B_DIR = ROOT / 'manifests_rebased' / 'green_ccpd2019_tilt_db_challenge_cvreplace_v4b_20260509'
V4B_DIR.mkdir(parents=True, exist_ok=True)

CVR4_TRAIN = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4_20260508/train_cvreplace_v4.csv'
CVR4_VAL = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/val_cvreplace_v2.csv'
E12_BASE = ROOT / 'manifests_rebased/unified_manifest_green_e12_replace_pose_v3_append.csv'
PROV_DEGRADE = ROOT / 'manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv'

OUT_FIELDS = [
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
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({k: r.get(k,'') for k in OUT_FIELDS})
    print(f"  {label}: {len(rows)}", flush=True)
    return rows

def cap_per_province(rows, cap):
    by_prov = defaultdict(list)
    for r in rows: by_prov[get_prov(r['text'])].append(r)
    capped = []
    for prov, samples in by_prov.items():
        capped.extend(random.sample(samples, min(cap, len(samples))))
    return capped

print("Loading...", flush=True)
cvr4 = load_clean(CVR4_TRAIN, 'cvreplace_v4')
cvr4_val = load_clean(CVR4_VAL, 'cvreplace_v4_val')
e12_all = load_clean(E12_BASE, 'E12_base')
prov_all = load_clean(PROV_DEGRADE, 'province_degrade')

# Split E12 by source
e12_real = [r for r in e12_all if r.get('source','').startswith('real') or r.get('source','') == '']
e12_other = [r for r in e12_all if not (r.get('source','').startswith('real') or r.get('source','') == '')]

# ── Target: cvr=45%, real=40%, prov=10%, other=5% ──────────────
n_cvr = len(cvr4)  # 91,998
target_total = int(n_cvr / 0.45)
target_real = int(target_total * 0.40)
target_prov = int(target_total * 0.10)
target_other = int(target_total * 0.05)
print(f"Target: total={target_total} cvr={n_cvr} real={target_real} prov={target_prov} other={target_other}", flush=True)

# Sample each source to target
real_capped = cap_per_province(e12_real, 3000)
if len(real_capped) > target_real:
    random.shuffle(real_capped)
    real_capped = real_capped[:target_real]
print(f"  Real sampled: {len(real_capped)}", flush=True)

prov_capped = cap_per_province(prov_all, 800)
if len(prov_capped) > target_prov:
    random.shuffle(prov_capped)
    prov_capped = prov_capped[:target_prov]
print(f"  Prov sampled: {len(prov_capped)}", flush=True)

other_capped = cap_per_province(e12_other, 400)
if len(other_capped) > target_other:
    random.shuffle(other_capped)
    other_capped = other_capped[:target_other]
print(f"  Other sampled: {len(other_capped)}", flush=True)

# Combine
combined = cvr4 + real_capped + prov_capped + other_capped
random.shuffle(combined)
print(f"\nCombined: {len(combined)}", flush=True)

# Actual ratios
src_groups = defaultdict(int)
for r in combined:
    s = r.get('source', '')
    if 'cvreplace_v4' in s: src_groups['cvreplace_v4'] += 1
    elif s.startswith('real') or s == '': src_groups['real'] += 1
    elif 'province_degrade' in s: src_groups['province_degrade'] += 1
    else: src_groups['other'] += 1
total_s = sum(src_groups.values())
print(f"\nSource ratios:")
for g, c in sorted(src_groups.items()):
    print(f"  {g}: {c} ({c/total_s*100:.0f}%)", flush=True)

# Province balance
prov_cnt = Counter(get_prov(r['text']) for r in combined)
total_p = sum(prov_cnt.values())
avg = total_p / max(len([p for p,c in prov_cnt.items() if p != '?']), 1)
wan_pct = prov_cnt.get('皖',0)/max(total_p,1)*100
max_ratio = max(prov_cnt.values())/max(avg,1)
print(f"\nProvince: avg={avg:.0f} 皖={wan_pct:.1f}% max/avg={max_ratio:.2f}", flush=True)

# Write
for r in combined: r['split'] = 'train'
out = V4B_DIR / 'train_v4b_rebalanced.csv'
with open(out, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    w.writeheader(); w.writerows(combined)
print(f"\nTrain: {out} ({len(combined)} rows)", flush=True)
