#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path

BASE = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b.csv'
OUT = '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv'
RAW_TRAIN = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/train_synthetic_labels.txt')
TARGET_RATIO = 0.40
ROOT = '/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/'

with open(BASE, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

base_train = [r for r in rows if r.get('split') == 'train']
others = [r for r in rows if r.get('split') != 'train']
used_train_keys = set((r['img_path'], (r.get('text') or '').upper()) for r in base_train)

# use one existing synthetic train row as template for static metadata
synthetic_template = None
for r in base_train:
    if r.get('family') == 'green8' and r.get('source') == 'synthetic_exact_quad':
        synthetic_template = r
        break
if synthetic_template is None:
    raise RuntimeError('No synthetic_exact_quad template row found in base train manifest')

extra_pool = []
for line in RAW_TRAIN.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    rel, text = line.strip().split(maxsplit=1)
    text = text.strip().upper()
    img_path = ROOT + rel
    key = (img_path, text)
    if key in used_train_keys:
        continue
    if text.startswith('皖'):
        continue
    row = dict(synthetic_template)
    row['img_path'] = img_path
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
    extra_pool.append(row)

extra_pool.sort(key=lambda r: ((r.get('text') or '')[:1], r.get('img_path') or ''))
base_total = len(base_train)
anhui_count = sum(1 for r in base_train if (r.get('text') or '').startswith('皖'))
needed = 0
while anhui_count / (base_total + needed) > TARGET_RATIO:
    needed += 1
selected = extra_pool[:needed]
new_train = list(base_train) + selected

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(new_train)
    w.writerows(others)

prov = Counter((r.get('text') or '')[:1] or '__empty__' for r in new_train)
src = Counter(r.get('source') or 'unknown' for r in new_train)
report = {
    'src': BASE,
    'out': OUT,
    'target_anhui_ratio': TARGET_RATIO,
    'original_train_total': len(base_train),
    'original_train_anhui': anhui_count,
    'raw_train_extra_pool': len(extra_pool),
    'needed_extra_non_anhui': needed,
    'picked_extra_non_anhui': len(selected),
    'new_train_total': len(new_train),
    'new_train_anhui': sum(1 for r in new_train if (r.get('text') or '').startswith('皖')),
    'new_train_anhui_ratio': sum(1 for r in new_train if (r.get('text') or '').startswith('皖')) / len(new_train),
    'source_breakdown': dict(src),
    'top_provinces': prov.most_common(15),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
