#!/usr/bin/env python3
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv'
OUT = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_h30a_targeted_tail.csv'
RAW_TRAIN = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/train_synthetic_labels.txt')
RAW_VAL = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/val_synthetic_labels.txt')
RAW_TEST = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/test_synthetic_labels.txt')
ROOT = '/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/'
TARGET_BOOST = {
    '苏': 1.20,
    '沪': 1.50,
    '闽': 1.20,
    '浙': 0.80,
}
TARGET_ORDER = ['苏', '沪', '闽', '浙']


def load_label_file(path):
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        rel, text = line.strip().split(maxsplit=1)
        text = text.strip().upper()
        img = ROOT + rel
        out.append((img, rel, text))
    return out

raw_train = load_label_file(RAW_TRAIN)
raw_val = set((img, text) for img, _, text in load_label_file(RAW_VAL))
raw_test = set((img, text) for img, _, text in load_label_file(RAW_TEST))

with open(BASE, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

train_rows = [r for r in rows if r.get('split') == 'train']
other_rows = [r for r in rows if r.get('split') != 'train']
train_key_count = Counter((r['img_path'], (r.get('text') or '').upper()) for r in train_rows)

# template row for synthetic metadata
synthetic_template = None
for r in train_rows:
    if r.get('family') == 'green8' and r.get('source') == 'synthetic_exact_quad':
        synthetic_template = r
        break
if synthetic_template is None:
    raise RuntimeError('No synthetic_exact_quad template row found')

# build available synthetic pool from raw synthetic train only
raw_train_records = []
for img, rel, text in raw_train:
    if text.startswith('皖'):
        continue
    row = dict(synthetic_template)
    row['img_path'] = img
    row['img_rel_path'] = 'green_exact_quad_synthetic_v1/' + rel
    row['dataset_name'] = 'green_exact_quad_synthetic_v1'
    row['split'] = 'train'
    row['text'] = text
    row['plate_len'] = str(len(text))
    row['family'] = 'green8'
    row['sub_type'] = 'green_small'
    row['source'] = 'synthetic_exact_quad'
    row['is_real'] = '0'
    row['need_tilt_aug'] = '1'
    row['preprocess_group'] = 'ccpd_board'
    row['has_bbox'] = '1'
    row['has_quad'] = '1'
    row['can_parse_ccpd_geom'] = '1'
    row['can_perspective'] = '1'
    row['bbox_source'] = 'synthetic_exact_quad'
    row['quad_source'] = 'synthetic_exact_quad'
    row['ocr_channel_order'] = 'bgr'
    row['ocr_crop_mode'] = 'obb_warp'
    row['ocr_resize_mode'] = 'letterbox'
    row['ocr_resize_kernel'] = 'nn'
    row['ocr_preproc'] = 'none'
    row['ocr_min_occ_ratio'] = '0.9'
    row['ocr_quad_pad_ratio'] = '0.0'
    raw_train_records.append(row)

# current targeted synthetic counts in H29B train
current_target_counts = Counter()
current_target_rows = defaultdict(list)
for r in train_rows:
    prov = (r.get('text') or '')[:1]
    if prov in TARGET_BOOST and r.get('source') == 'synthetic_exact_quad':
        current_target_counts[prov] += 1
        current_target_rows[prov].append(r)

# raw train rows not yet present in H29B train
unused_by_target = defaultdict(list)
for r in raw_train_records:
    key = (r['img_path'], (r.get('text') or '').upper())
    prov = (r.get('text') or '')[:1]
    if prov in TARGET_BOOST and train_key_count[key] == 0:
        unused_by_target[prov].append(r)
for prov in TARGET_ORDER:
    unused_by_target[prov].sort(key=lambda r: r['img_path'])
    current_target_rows[prov].sort(key=lambda r: r['img_path'])

added_rows = []
# stage 1: add truly unused raw train rows where available
used_extra_count = Counter()
for prov in TARGET_ORDER:
    base_count = current_target_counts[prov]
    target_add = int(round(base_count * TARGET_BOOST[prov]))
    take = min(target_add, len(unused_by_target[prov]))
    for row in unused_by_target[prov][:take]:
        added_rows.append(dict(row))
    used_extra_count[prov] += take

# stage 2: if still short, duplicate existing train synthetic rows of target provinces
# duplication is intentional re-sampling, so keep same img_path/text and append extra rows unchanged
for prov in TARGET_ORDER:
    base_count = current_target_counts[prov]
    target_add = int(round(base_count * TARGET_BOOST[prov]))
    remain = target_add - used_extra_count[prov]
    if remain <= 0:
        continue
    pool = current_target_rows[prov]
    if not pool:
        continue
    idx = 0
    while remain > 0:
        added_rows.append(dict(pool[idx % len(pool)]))
        idx += 1
        remain -= 1

new_train_rows = list(train_rows) + added_rows

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(new_train_rows)
    w.writerows(other_rows)

# audits
new_train_keys = Counter((r['img_path'], (r.get('text') or '').upper()) for r in new_train_rows)
duplicate_key_count = sum(1 for _, c in new_train_keys.items() if c > 1)
added_keys = Counter((r['img_path'], (r.get('text') or '').upper()) for r in added_rows)
added_dup_repeat = sum(c - 1 for _, c in added_keys.items() if c > 1)
train_vs_val = sum(1 for k in new_train_keys if k in raw_val)
train_vs_test = sum(1 for k in new_train_keys if k in raw_test)
prov_counts = Counter((r.get('text') or '')[:1] or '__empty__' for r in new_train_rows)
src_counts = Counter(r.get('source') or 'unknown' for r in new_train_rows)
added_target_breakdown = Counter((r.get('text') or '')[:1] or '__empty__' for r in added_rows)
added_source_breakdown = Counter(r.get('source') or 'unknown' for r in added_rows)

report = {
    'base_manifest': BASE,
    'out_manifest': OUT,
    'target_boost': TARGET_BOOST,
    'current_target_synth_counts': dict(current_target_counts),
    'unused_target_raw_train_counts': {k: len(v) for k, v in unused_by_target.items()},
    'added_total_rows': len(added_rows),
    'added_target_breakdown': dict(added_target_breakdown),
    'added_source_breakdown': dict(added_source_breakdown),
    'added_repeated_rows_beyond_first_copy': added_dup_repeat,
    'new_train_total': len(new_train_rows),
    'new_train_anhui_ratio': sum(1 for r in new_train_rows if (r.get('text') or '').startswith('皖')) / len(new_train_rows),
    'new_train_source_breakdown': dict(src_counts),
    'new_train_target_counts_all_sources': {p: prov_counts[p] for p in TARGET_ORDER + ['皖']},
    'audit': {
        'train_intersects_raw_synthetic_val': train_vs_val,
        'train_intersects_raw_synthetic_test': train_vs_test,
        'duplicate_train_keys_gt1': duplicate_key_count,
    },
}
print(json.dumps(report, ensure_ascii=False, indent=2))
