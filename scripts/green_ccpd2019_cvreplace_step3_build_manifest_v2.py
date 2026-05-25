#!/usr/bin/env python3
"""Step 3 v2: Build province-balanced train manifest for green CCPD2019 CV-replace v2.
Rules:
- New cvreplace v2 train: all retained
- Province_degrade: all retained
- Old E12 base: per-province capped sampling
- 皖 <= 8% of total
- No province > 1.5x average
- Fail with error if conditions not met"""

import csv, sys, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
random.seed(20260508)

DATE_TAG = '20260508'
MANIFEST_DIR = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v2_{DATE_TAG}'
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Input manifests
NEW_TRAIN = MANIFEST_DIR / 'train_cvreplace_v2.csv'
NEW_VAL = MANIFEST_DIR / 'val_cvreplace_v2.csv'
PROV_DEGRADE = ROOT / 'manifests_rebased/province_degrade_train_v1/train_province_degrade_v1.csv'
E12_BASE = ROOT / 'manifests_rebased/unified_manifest_green_e12_replace_pose_v3_append.csv'

OUT_TRAIN = MANIFEST_DIR / 'train_balanced_combined.csv'
OUT_VAL = MANIFEST_DIR / 'val_cvreplace_v2.csv'  # copy val unchanged

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


def load_rows(path, label=None):
    if not path.exists():
        print(f"  SKIP {label or path}: not found", flush=True)
        return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            nr = {k: row.get(k, '') for k in OUTPUT_FIELDS}
            rows.append(nr)
    print(f"  Loaded {label}: {len(rows)} rows", flush=True)
    return rows


def summarize_prov(rows, tag):
    c = Counter(get_prov(r['text']) for r in rows)
    print(f"  {tag} province: {dict(c.most_common(5))}  total={sum(c.values())}", flush=True)
    return c


print("Loading manifests...", flush=True)
new_train = load_rows(NEW_TRAIN, 'cvreplace_v2_train')
new_val = load_rows(NEW_VAL, 'cvreplace_v2_val')

TOTAL_NEW = len(new_train)
if TOTAL_NEW < 1000:
    CAP_PER_PROV = 20  # smoke
else:
    CAP_PER_PROV = 500  # full
print(f"Total new data: {TOTAL_NEW}, using E12 cap: {CAP_PER_PROV} per province", flush=True)

prov_degrade = load_rows(PROV_DEGRADE, 'province_degrade')

# Cap province_degrade per-province too
degrade_by_prov = defaultdict(list)
for r in prov_degrade:
    degrade_by_prov[get_prov(r['text'])].append(r)
prov_degrade_sampled = []
for prov, samples in degrade_by_prov.items():
    take = min(CAP_PER_PROV, len(samples))
    prov_degrade_sampled.extend(random.sample(samples, take))
print(f"  Capped province_degrade: {len(prov_degrade_sampled)} (from {len(prov_degrade)})", flush=True)

e12_base = load_rows(E12_BASE, 'E12_base')

print(f"\n--- Input summaries ---", flush=True)
summarize_prov(new_train, 'cvreplace_v2_train')
summarize_prov(prov_degrade, 'province_degrade')
e12_prov = summarize_prov(e12_base, 'E12_base')

# Sample E12 base per-province
e12_by_prov = defaultdict(list)
for r in e12_base:
    p = get_prov(r['text'])
    e12_by_prov[p].append(r)

e12_sampled = []
for prov, samples in e12_by_prov.items():
    take = min(CAP_PER_PROV, len(samples))
    e12_sampled.extend(random.sample(samples, take))

print(f"  Sampled E12: {len(e12_sampled)} (from {len(e12_base)} total)", flush=True)
summarize_prov(e12_sampled, 'E12_sampled')

# ── Build combined ─────────────────────────────────────────────
combined = []
combined.extend(new_train)   # all cvreplace v2
combined.extend(prov_degrade_sampled)  # capped province_degrade
combined.extend(e12_sampled)   # capped E12

# Set all splits to train
for r in combined:
    r['split'] = 'train'

random.shuffle(combined)
print(f"\nCombined train: {len(combined)}", flush=True)

# ── Validate province balance ──────────────────────────────────
prov_counts = Counter(get_prov(r['text']) for r in combined)
total = sum(prov_counts.values())
avg_per_prov = total / max(len(prov_counts), 1)
max_prov = max(prov_counts.values()) if prov_counts else 0
max_prov_name = max(prov_counts, key=prov_counts.get) if prov_counts else '?'
wan_pct = prov_counts.get('皖', 0) / max(total, 1) * 100
max_ratio = max_prov / max(avg_per_prov, 1)

print(f"\n  --- Validation ---", flush=True)
print(f"  Total: {total}", flush=True)
print(f"  Avg per province: {avg_per_v:.1f}" if False else f"  Avg per province: {avg_per_prov:.1f}", flush=True)
print(f"  Max province: {max_prov_name}={max_prov} ({max_prov/total*100:.1f}%) ratio_to_avg={max_ratio:.2f}", flush=True)
print(f"  皖: {prov_counts.get('皖',0)} ({wan_pct:.1f}%)", flush=True)

violations = []
if wan_pct > 8.0:
    violations.append(f"皖占比 {wan_pct:.1f}% > 8.0%")
if max_ratio > 1.5:
    violations.append(f"最大省/均值比 {max_ratio:.2f} > 1.5")

if violations:
    print(f"\n  *** VIOLATIONS ***", flush=True)
    for v in violations:
        print(f"    FAIL: {v}", flush=True)
    print(f"\n  Adjusting: further capping 皖...", flush=True)
    
    # Reduce 皖 samples
    wan_rows = [r for r in combined if get_prov(r['text']) == '皖']
    non_wan = [r for r in combined if get_prov(r['text']) != '皖']
    target_wan = int(total * 0.08)
    if len(wan_rows) > target_wan:
        random.shuffle(wan_rows)
        wan_rows = wan_rows[:target_wan]
    combined = non_wan + wan_rows
    random.shuffle(combined)
    
    # Re-check
    prov_counts = Counter(get_prov(r['text']) for r in combined)
    total = sum(prov_counts.values())
    wan_pct = prov_counts.get('皖', 0) / max(total, 1) * 100
    avg_per_prov = total / max(len(prov_counts), 1)
    max_prov = max(prov_counts.values())
    max_ratio = max_prov / max(avg_per_prov, 1)
    
    print(f"  After adjustment:", flush=True)
    print(f"  Total: {total}, 皖: {prov_counts.get('皖',0)} ({wan_pct:.1f}%), max/avg ratio: {max_ratio:.2f}", flush=True)
    
    # Final check — fail hard if still out of bounds
    if wan_pct > 8.0 or max_ratio > 1.5:
        raise RuntimeError(
            f"Province balance validation FAILED: 皖={wan_pct:.1f}% (max 8%), "
            f"max/avg={max_ratio:.2f} (max 1.5). "
            f"Adjust caps and retry."
        )

print(f"\n  ✅ Province balance PASSED", flush=True)

# ── Write manifests ────────────────────────────────────────────
for r in combined:
    r['split'] = 'train'

with open(OUT_TRAIN, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    w.writeheader()
    w.writerows(combined)
print(f"\nWritten: {OUT_TRAIN} ({len(combined)} rows)", flush=True)

# Copy val manifest (ensure split=test)
val_rows = load_rows(NEW_VAL, 'cvreplace_v2_val')
for r in val_rows:
    r['split'] = 'test'
with open(OUT_VAL, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    w.writeheader()
    w.writerows(val_rows)
print(f"Written: {OUT_VAL} ({len(val_rows)} rows)", flush=True)

# ── Full summary ────────────────────────────────────────────────
src_cnt = Counter(r.get('source', '?') for r in combined)
print(f"\n{'='*60}")
print(f"FINAL MANIFEST SUMMARY v2")
print(f"{'='*60}")
print(f"  Train: {len(combined)}")
print(f"  Val:   {len(val_rows)}")
print(f"\n  Province distribution:")
for p, c in sorted(prov_counts.items(), key=lambda x: -x[1]):
    print(f"    {p}: {c} ({c/total*100:.1f}%)")
print(f"\n  Source distribution:")
for s, c in src_cnt.most_common():
    print(f"    {s}: {c}")
print(f"\n  Balance: 皖={wan_pct:.1f}%, max/avg={max_ratio:.2f}")
