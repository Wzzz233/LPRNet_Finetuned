#!/usr/bin/env python3
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_h30a_targeted_tail.csv'
BASE = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv'
RAW_VAL = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/val_synthetic_labels.txt')
RAW_TEST = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/test_synthetic_labels.txt')
TARGETS = {'苏','沪','闽','浙'}
ROOT = '/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/'


def load_raw(path):
    out=set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        rel, text = line.strip().split(maxsplit=1)
        out.add((ROOT + rel, text.strip().upper()))
    return out

raw_val = load_raw(RAW_VAL)
raw_test = load_raw(RAW_TEST)

with open(MANIFEST, 'r', encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
with open(BASE, 'r', encoding='utf-8', newline='') as f:
    base_rows = list(csv.DictReader(f))

train_rows = [r for r in rows if r.get('split') == 'train']
base_train_rows = [r for r in base_rows if r.get('split') == 'train']
base_keys = Counter((r['img_path'], (r.get('text') or '').upper()) for r in base_train_rows)
train_keys = Counter((r['img_path'], (r.get('text') or '').upper()) for r in train_rows)
repeated = {k:c for k,c in train_keys.items() if c > base_keys.get(k, 0)}
inter_val = sum(1 for k in train_keys if k in raw_val)
inter_test = sum(1 for k in train_keys if k in raw_test)
prov_all = Counter((r.get('text') or '')[:1] for r in train_rows)
prov_repeat = Counter(k[1][:1] for k in repeated)
report = {
    'manifest': MANIFEST,
    'train_total': len(train_rows),
    'anhui_ratio': sum(1 for r in train_rows if (r.get('text') or '').startswith('皖')) / len(train_rows),
    'target_counts_all_sources': {p: prov_all[p] for p in ['苏','沪','闽','浙','皖']},
    'audit': {
        'train_intersects_raw_synthetic_val': inter_val,
        'train_intersects_raw_synthetic_test': inter_test,
        'repeat_key_groups_added_or_upsampled': len(repeated),
        'repeat_rows_extra_vs_base': sum(train_keys[k] - base_keys.get(k, 0) for k in repeated),
        'repeat_target_breakdown': {p: prov_repeat[p] for p in ['苏','沪','闽','浙']},
        'non_target_repeat_groups': sum(v for k,v in prov_repeat.items() if k not in TARGETS),
    }
}
print(json.dumps(report, ensure_ascii=False, indent=2))
