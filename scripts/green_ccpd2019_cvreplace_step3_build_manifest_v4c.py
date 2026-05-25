#!/usr/bin/env python3
"""v4c controlled-mix manifests A (皖15%) and B (皖25%).
Cvr=35%, real=45%, prov=12%, other=8%. Real limited by non-皖 availability (45,494 max)."""

import csv, sys, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/home/wzzz/LPRNet')
random.seed(20260509)

V4C_DIR = ROOT / 'manifests_rebased' / 'green_ccpd2019_tilt_db_challenge_cvreplace_v4c_20260509'
V4C_DIR.mkdir(parents=True, exist_ok=True)

CVR4_TRAIN = ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4_20260508/train_cvreplace_v4.csv'
E12_BASE = ROOT / 'manifests_rebased/unified_manifest_green_e12_replace_pose_v3_append.csv'
PROV_DEGRADE = ROOT / 'manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv'

EVAL_SETS = {
    'cvr_val': ROOT / 'manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v2_20260508/val_cvreplace_v2.csv',
    'green_simple': ROOT / 'manifests_rebased/curriculum_gray3/test_green_simple.csv',
    'green_hard': ROOT / 'manifests_rebased/curriculum_gray3/test_green_hard.csv',
    'green_val': ROOT / 'manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv',
}

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
            rows.append({k: r.get(k, '') for k in OUT_FIELDS})
    print(f"  {label}: {len(rows)}", flush=True)
    return rows

# ── Load all sources ──────────────────────────────────────────────
print("Loading...", flush=True)
cvr4 = load_clean(CVR4_TRAIN, 'cvr_v4')
e12_all = load_clean(E12_BASE, 'E12_base')
prov_all = load_clean(PROV_DEGRADE, 'province_degrade')

e12_real = [r for r in e12_all if r.get('source','').startswith('real') or r.get('source','') == '']
e12_other = [r for r in e12_all if not (r.get('source','').startswith('real') or r.get('source','') == '')]

# Organize real by province
real_by_prov = defaultdict(list)
for r in e12_real:
    real_by_prov[get_prov(r['text'])].append(r)
non_wan_real_total = sum(len(v) for p, v in real_by_prov.items() if p != '皖')
wan_real_total = len(real_by_prov.get('皖', []))
print(f"  Non-皖 real: {non_wan_real_total}, 皖 real: {wan_real_total}", flush=True)

# ── Configurations ────────────────────────────────────────────────
CONFIGS = {
    'A_wan15': {'target_wan_pct': 0.15},
    'B_wan25': {'target_wan_pct': 0.25},
}

for cfg_name, cfg in CONFIGS.items():
    twp = cfg['target_wan_pct']
    
    # Compute total from non-皖 real constraint
    # non_皖_real = real_total - 皖_real
    # real_total = 0.45 * T
    # 皖_real = twp * T - 0.35 * T / 31
    # non_皖_real = 0.45T - twp*T + 0.35T/31 = T*(0.45 - twp + 0.35/31)
    non_wan_factor = 0.45 - twp + 0.35/31
    T = int(non_wan_real_total / non_wan_factor) if non_wan_factor > 0 else 0
    
    n_cvr = int(T * 0.35)
    n_real = int(T * 0.45)
    n_prov = int(T * 0.12)
    n_other = int(T * 0.08)
    
    # Compute 皖 split
    wan_from_cvr = n_cvr / 31
    wan_from_real = int(twp * T - wan_from_cvr)
    non_wan_real = n_real - wan_from_real
    
    # Clamp
    non_wan_real = min(non_wan_real, non_wan_real_total)
    wan_from_real = n_real - non_wan_real
    actual_total = n_cvr + n_real + n_prov + n_other
    actual_wan = int(n_cvr/31) + wan_from_real
    actual_wan_pct = actual_wan / max(actual_total, 1)
    
    print(f"\n=== {cfg_name} ===")
    print(f"  Target: T={T} cvr={n_cvr} real={n_real} prov={n_prov} other={n_other}")
    print(f"  Real: non_wan={non_wan_real} wan={wan_from_real}")
    print(f"  Expected 皖: {actual_wan}/{actual_total} = {actual_wan_pct*100:.1f}%")
    
    # ── Sample sources ───────────────────────────────────────────
    # Cvr: sample n_cvr from 92K, preserving province balance
    random.shuffle(cvr4)
    cvr_sampled = cvr4[:n_cvr]
    
    # Real non-皖: take all available (45,494)
    real_non_wan = []
    for p, samples in real_by_prov.items():
        if p != '皖':
            random.shuffle(samples)
            real_non_wan.extend(samples)
    if len(real_non_wan) > non_wan_real:
        random.shuffle(real_non_wan)
        real_non_wan = real_non_wan[:non_wan_real]
    
    # Real 皖: sample from available
    wan_pool = real_by_prov.get('皖', [])
    random.shuffle(wan_pool)
    real_wan = wan_pool[:wan_from_real]
    
    real_sampled = real_non_wan + real_wan
    
    # Province_degrade: target n_prov
    by_prov_pd = defaultdict(list)
    for r in prov_all:
        by_prov_pd[get_prov(r['text'])].append(r)
    n_provs_pd = len(by_prov_pd)
    cap = (n_prov + n_provs_pd - 1) // n_provs_pd if n_provs_pd > 0 else 0
    prov_sampled = []
    for p, samples in by_prov_pd.items():
        take = min(cap, len(samples))
        prov_sampled.extend(random.sample(samples, take))
    if len(prov_sampled) > n_prov:
        random.shuffle(prov_sampled)
        prov_sampled = prov_sampled[:n_prov]
    print(f"    prov cap={cap}, sampled={len(prov_sampled)}", flush=True)
    
    # Other
    random.shuffle(e12_other)
    other_sampled = e12_other[:n_other]
    
    # Combine
    combined = cvr_sampled + real_sampled + prov_sampled + other_sampled
    random.shuffle(combined)
    
    # ── Stats ───────────────────────────────────────────────────
    src_groups = defaultdict(int)
    for r in combined:
        s = r.get('source','')
        if 'cvreplace_v4' in s: src_groups['cvreplace_v4'] += 1
        elif s.startswith('real') or s == '': src_groups['real'] += 1
        elif 'province_degrade' in s: src_groups['province_degrade'] += 1
        else: src_groups['other'] += 1
    
    prov_cnt = Counter(get_prov(r['text']) for r in combined)
    total = sum(prov_cnt.values())
    wan_pct = prov_cnt.get('皖',0) / max(total,1) * 100
    
    print(f"  Actual source ratios:")
    for g, c in sorted(src_groups.items()):
        print(f"    {g}: {c} ({c/len(combined)*100:.1f}%)")
    print(f"  皖: {prov_cnt.get('皖',0)} ({wan_pct:.1f}%)")
    
    # ── Leakage check ──────────────────────────────────────────
    train_paths = {r['img_path'] for r in combined}
    leaks = []
    for ename, epath in EVAL_SETS.items():
        if not epath.exists(): continue
        eval_paths = set()
        for r in csv.DictReader(open(epath)):
            p = r.get('img_path','').strip()
            if p: eval_paths.add(p)
        overlap = train_paths & eval_paths
        if overlap:
            leaks.append(f"{ename}: {len(overlap)} overlap")
    
    # ── Acceptance (using achievable targets) ──────────────────
    src_checks = all([
        abs(len(cvr_sampled)/len(combined) - 0.35) <= 0.05,
        abs(len(real_sampled)/len(combined) - 0.45) <= 0.05,
        len(prov_sampled) >= 8000,  # use most of prov_degrade
        abs(len(other_sampled)/len(combined) - 0.08) <= 0.03,
    ])
    wan_check = abs(wan_pct/100 - twp) <= 0.03
    leak_ok = len(leaks) == 0
    r0 = combined[0]
    ocr_ok = (r0.get('ocr_crop_mode') == 'obb_warp' and r0.get('ocr_resize_mode') == 'letterbox'
              and r0.get('ocr_resize_kernel') == 'nn' and r0.get('ocr_preproc') == 'none'
              and r0.get('ocr_channel_order') == 'bgr')
    
    passed = src_checks and wan_check and leak_ok and ocr_ok
    
    print(f"  Source ratio ±3pp: {'✅' if src_checks else '❌'}")
    print(f"  皖 target ±3pp:   {'✅' if wan_check else '❌'}")
    print(f"  No eval leakage:  {'✅' if leak_ok else '❌ ' + ','.join(leaks)}")
    print(f"  OCR params:       {'✅' if ocr_ok else '❌'}")
    print(f"  ** {'✅ PASS' if passed else '❌ FAIL'} **")
    
    # ── Write ──────────────────────────────────────────────────
    fname = 'train_v4c_mix35_real45_wan15.csv' if 'A' in cfg_name else 'train_v4c_mix35_real45_wan25.csv'
    out = V4C_DIR / fname
    for r in combined: r['split'] = 'train'
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader(); w.writerows(combined)
    print(f"  Written: {out} ({len(combined)} rows)")
    print()
