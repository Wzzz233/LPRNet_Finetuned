#!/usr/bin/env python3
"""Phase 0: Audit baseline manifest - pose bins, quality bins, real/replace ratio.
Outputs report and tagged manifest for next-stage experiments.

Usage: python tools/routeA_nextstage/audit_manifest.py
"""
import csv, json, math, cv2, numpy as np
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/home/wzzz/LPRNet')
MANIFEST = ROOT / 'manifests_rebased/routeA_prime_quadwarp_20260512/train_real_replace_bal31_v1.csv'
OUT_DIR = ROOT / 'experiments/routeA_nextstage_20260512'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROVINCES = ['京','津','冀','晋','蒙','辽','吉','黑','沪','苏','浙','皖','闽','赣','鲁','豫',
             '鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新','渝']

def parse_quad(row):
    try:
        q = np.array([[float(row[f'quad_{i}x']), float(row[f'quad_{i}y'])] for i in range(1,5)])
        return q
    except: return None

def compute_tilt(quad):
    """Compute tilt angle from quad top edge."""
    top_edge = quad[1] - quad[0]
    angle = math.degrees(math.atan2(abs(top_edge[1]), abs(top_edge[0])))
    return angle

def compute_blur(img_path):
    """Laplacian variance for blur estimation."""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except: return None

def compute_contrast(img_path):
    """RMS contrast."""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        return img.std()
    except: return None

# Load manifest
rows = []
q_missing = 0
total = 0
with open(MANIFEST, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        rows.append(r)
total = len(rows)
print(f'Total rows: {total}')

# Province + source stats
prov_counter = Counter()
source_counter = Counter()
real_count = 0
replace_count = 0
tilt_bins = Counter()
quality_bins = Counter()
prov_source = defaultdict(lambda: Counter())
sample_rows = []  # for spot-check

for idx, r in enumerate(rows):
    text = r.get('text','').strip()
    src = r.get('source','').strip()
    prov = text[0] if text else '?'
    prov_counter[prov] += 1
    source_counter[src] += 1
    prov_source[prov][src] += 1
    
    if 'ccpd2020' in src.lower() or 'real' in src.lower():
        real_count += 1
    else:
        replace_count += 1
    
    # Tilt from quad
    quad = parse_quad(r)
    if quad is not None:
        angle = compute_tilt(quad)
        if angle < 5: tilt_bins['fronto-parallel'] += 1
        elif angle < 15: tilt_bins['light'] += 1
        elif angle < 30: tilt_bins['medium'] += 1
        else: tilt_bins['strong'] += 1
    else:
        q_missing += 1

    # Sample first 200 rows for quality analysis
    if idx < 200:
        img_path = r.get('img_path','').strip()
        full_path = img_path if img_path.startswith('/') else str(ROOT / img_path)
        sample_rows.append((full_path, prov, src, angle if quad is not None else None))

# Print report
print(f'\n=== AUDIT REPORT ===')
print(f'Total: {total}')
print(f'Real: {real_count} ({real_count/total*100:.1f}%)')
print(f'Replace: {replace_count} ({replace_count/total*100:.1f}%)')
print(f'Missing quad: {q_missing}')
print(f'\nSource distribution:')
for s, c in source_counter.most_common():
    print(f'  {s}: {c} ({c/total*100:.1f}%)')
print(f'\nProvince distribution:')
for p in PROVINCES:
    if p in prov_counter:
        c = prov_counter[p]
        real_p = prov_source[p].get('ccpd2020', 0)
        print(f'  {p}: {c} (real={real_p})')
print(f'\nTilt bins:')
for b, c in tilt_bins.most_common():
    print(f'  {b}: {c} ({c/total*100:.1f}%)')

# Quality analysis on sample
print(f'\nQuality sample (first 200 rows):')
blurs = []
contrasts = []
for path, prov, src, angle in sample_rows:
    blur = compute_blur(path)
    contrast = compute_contrast(path)
    if blur is not None: blurs.append(blur)
    if contrast is not None: contrasts.append(contrast)

if blurs:
    blurs_p25, blurs_p50, blurs_p75 = np.percentile(blurs, [25,50,75])
    print(f'  Laplacian variance (blur): p25={blurs_p25:.0f} p50={blurs_p50:.0f} p75={blurs_p75:.0f}')
    print(f'  Blur categories:')
    print(f'    Sharp (>500): {sum(1 for b in blurs if b>500)}')
    print(f'    Normal (100-500): {sum(1 for b in blurs if 100<=b<=500)}')
    print(f'    Blurry (<100): {sum(1 for b in blurs if b<100)}')
if contrasts:
    print(f'  Contrast (std): mean={np.mean(contrasts):.0f} p25={np.percentile(contrasts,25):.0f} p50={np.percentile(contrasts,50):.0f} p75={np.percentile(contrasts,75):.0f}')

# Count real samples per province (from CCD2020)
print(f'\nReal (CCPD2020) samples per province:')
real_prov = Counter()
for r in rows:
    if 'ccpd2020' in r.get('source','').lower():
        prov = r.get('text','').strip()[0] if r.get('text','').strip() else '?'
        real_prov[prov] += 1
for p in PROVINCES:
    if p in real_prov:
        print(f'  {p}: {real_prov[p]}')
print(f'  Total real in balanced set: {sum(real_prov.values())}')

# Generate tagged manifest
print(f'\nGenerating tagged manifest...')
tagged_fields = fieldnames + ['tilt_bin']
tagged_csv = OUT_DIR / 'train_bal31_tagged.csv'
tagged_rows = []
for idx, r in enumerate(rows):
    quad = parse_quad(r)
    if quad is not None:
        angle = compute_tilt(quad)
        if angle < 5: tb = 'fronto'
        elif angle < 15: tb = 'light'
        elif angle < 30: tb = 'medium'
        else: tb = 'strong'
    else:
        tb = 'unknown'
    new_r = dict(r)
    new_r['tilt_bin'] = tb
    tagged_rows.append(new_r)

with open(tagged_csv, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=tagged_fields)
    w.writeheader()
    w.writerows(tagged_rows)
print(f'  Saved: {tagged_csv} ({len(tagged_rows)} rows)')

# Summary JSON
summary = {
    'total': total,
    'real_count': real_count,
    'replace_count': replace_count,
    'real_pct': round(real_count/total*100, 1),
    'province_counts': {p: prov_counter.get(p,0) for p in PROVINCES},
    'real_per_province': {p: real_prov.get(p,0) for p in PROVINCES},
    'source_counts': dict(source_counter.most_common()),
    'tilt_bins': dict(tilt_bins.most_common()),
    'quality_sample': {
        'blur_p50': float(np.percentile(blurs, 50)) if blurs else None,
        'contrast_p50': float(np.percentile(contrasts, 50)) if contrasts else None,
    }
}
(OUT_DIR / 'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f'\nAudit: {OUT_DIR / "audit_summary.json"}')
