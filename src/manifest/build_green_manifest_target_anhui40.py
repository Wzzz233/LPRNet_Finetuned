#!/usr/bin/env python3
import csv
import json
from collections import Counter

SRC = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv'
OUT = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40.csv'
TARGET_RATIO = 0.40
EPS = 0.0005

with open(SRC, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

train = [r for r in rows if r.get('split') == 'train']
others = [r for r in rows if r.get('split') != 'train']
used_paths = set(r['img_path'] for r in train)
all_syn = [r for r in rows if r.get('source') == 'synthetic_exact_quad']
extra_syn_pool = [r.copy() for r in all_syn if r.get('img_path') not in used_paths and not (r.get('text') or '').startswith('皖')]
extra_syn_pool.sort(key=lambda r: ((r.get('text') or '')[:1], r.get('img_path') or ''))

base_total = len(train)
anhui_count = sum(1 for r in train if (r.get('text') or '').startswith('皖'))
needed = 0
while anhui_count / (base_total + needed) > (TARGET_RATIO + EPS):
    needed += 1
selected_extra = extra_syn_pool[:needed]
for r in selected_extra:
    r['split'] = 'train'
new_train = list(train) + selected_extra

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
    'eps': EPS,
    'original_train_total': len(train),
    'original_train_anhui': anhui_count,
    'new_train_total': len(new_train),
    'new_train_anhui': sum(1 for r in new_train if (r.get('text') or '').startswith('皖')),
    'new_train_anhui_ratio': sum(1 for r in new_train if (r.get('text') or '').startswith('皖')) / len(new_train),
    'added_extra_non_anhui_synthetic': len(selected_extra),
    'source_breakdown': dict(src),
    'top_provinces': prov.most_common(15),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
