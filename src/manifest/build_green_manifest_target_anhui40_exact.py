#!/usr/bin/env python3
import csv
import json
from collections import Counter

SRC = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv'
OUT = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_exact.csv'
TARGET_RATIO = 0.40

with open(SRC, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

train = [r for r in rows if r.get('split') == 'train']
others = [r for r in rows if r.get('split') != 'train']
used_keys = set((r['img_path'], r.get('text','')) for r in train)

def pick(source):
    pool = [r.copy() for r in rows if r.get('family') == 'green8' and r.get('source') == source and (r.get('img_path'), r.get('text','')) not in used_keys and not (r.get('text') or '').startswith('皖')]
    pool.sort(key=lambda r: ((r.get('text') or '')[:1], r.get('img_path') or ''))
    return pool

syn_pool = pick('synthetic_exact_quad')
pseudo_pool = pick('pseudo_geom')
real_pool = pick('real')

base_total = len(train)
anhui_count = sum(1 for r in train if (r.get('text') or '').startswith('皖'))
needed = 0
while anhui_count / (base_total + needed) > TARGET_RATIO:
    needed += 1

selected = []
remaining = needed
for pool in (syn_pool, pseudo_pool, real_pool):
    take = min(len(pool), remaining)
    for row in pool[:take]:
        row['split'] = 'train'
        selected.append(row)
    remaining -= take
    if remaining <= 0:
        break

if remaining > 0:
    raise RuntimeError(f'Not enough unique extra non-Anhui rows even after syn+pseudo+real, still need {remaining}')

new_train = list(train) + selected
with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(new_train)
    w.writerows(others)

prov = Counter((r.get('text') or '')[:1] or '__empty__' for r in new_train)
src = Counter(r.get('source') or 'unknown' for r in new_train)
report = {
    'src': SRC,
    'out': OUT,
    'target_anhui_ratio': TARGET_RATIO,
    'original_train_total': len(train),
    'original_train_anhui': anhui_count,
    'needed_extra_non_anhui': needed,
    'picked_counts': {
        'synthetic_exact_quad': min(len(syn_pool), needed),
        'pseudo_geom': min(len(pseudo_pool), max(0, needed - len(syn_pool))),
        'real': max(0, needed - len(syn_pool) - len(pseudo_pool)),
    },
    'new_train_total': len(new_train),
    'new_train_anhui': sum(1 for r in new_train if (r.get('text') or '').startswith('皖')),
    'new_train_anhui_ratio': sum(1 for r in new_train if (r.get('text') or '').startswith('皖')) / len(new_train),
    'source_breakdown': dict(src),
    'top_provinces': prov.most_common(15),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
