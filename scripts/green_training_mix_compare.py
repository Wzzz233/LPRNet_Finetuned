#!/usr/bin/env python3
"""Compare blue v2 vs green v4/v4b training mix."""
import csv
from collections import Counter

blue_v2 = list(csv.DictReader(open('manifests_rebased/blue_ccpd2019_posquad_v2_hardmine_20260508/train_hardmine.csv')))
green_v4 = list(csv.DictReader(open('manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4_20260508/train_cvreplace_v4.csv')))
green_v4b = list(csv.DictReader(open('manifests_rebased/green_ccpd2019_tilt_db_challenge_cvreplace_v4b_20260509/train_v4b_rebalanced.csv')))

def prov_dist(rows, label):
    provs = Counter(r.get('text','')[:1] for r in rows if r.get('text'))
    wan = provs.get('皖',0)
    print(f'{label}: total={len(rows)} 皖={wan} ({wan/len(rows)*100:.1f}%) provinces={len(provs)} min={min(provs.values())} max={max(provs.values())}')

def src_dist(rows, label):
    src = Counter(r.get('source','') for r in rows)
    print(f'{label} sources:')
    for s, c in src.most_common(10):
        print(f'  {s}: {c} ({c/len(rows)*100:.1f}%)')

print('=== BLUE v2 (hardmine + balanced) ===')
prov_dist(blue_v2, 'Blue v2')
src_dist(blue_v2, 'Blue v2')
print()

print('=== GREEN v4 (cvreplace only, 63% of balanced) ===')
prov_dist(green_v4, 'Green v4')
src_dist(green_v4, 'Green v4')
print()

print('=== GREEN v4b (rebalanced, cvr 60% real 27%) ===')
prov_dist(green_v4b, 'Green v4b')
v4b_cat = Counter()
for r in green_v4b:
    s = r.get('source','')
    if 'cvreplace_v4' in s: v4b_cat['cvreplace'] += 1
    elif s.startswith('real') or s == '': v4b_cat['real'] += 1
    elif 'province_degrade' in s: v4b_cat['prov_degrade'] += 1
    else: v4b_cat['other'] += 1
for s, c in v4b_cat.most_common():
    print(f'  {s}: {c} ({c/len(green_v4b)*100:.1f}%)')
print()

print('=== KEY DIFFERENCES ===')
print(f'Blue v2 has hard mining (2x error weight): YES')
print(f'Green v4 has hard mining: NO')
print(f'Blue v2 real blue mix: blue_simple/val/hard')
print(f'Green v4b real green mix: E12 real (capped 3K/prov)')
print(f'Blue v2 cvr-style share: 100% (all tilt/db/challenge)')
print(f'Green v4b cvr-style share: 60%')
print(f'Blue v2 province: perfectly balanced (hardmine)')
print(f'Green v4b province: 皖={blue_v2.count}/{len(blue_v2)*100:.1f}%' if False else '')
