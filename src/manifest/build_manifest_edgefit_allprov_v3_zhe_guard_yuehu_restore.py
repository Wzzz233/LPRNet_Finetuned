#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path('/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv')
EDGEFIT_ROOT = Path('/home/wzzz/LPRNet/green_edgefit_allprov_v3_zhe_guard_yuehu_restore')
OUT = Path('/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_allprov_v3_zhe_guard_yuehu_restore.csv')

with BASE.open('r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

base_train = [r for r in rows if r.get('split') == 'train']
base_others = [r for r in rows if r.get('split') != 'train']

synthetic_template = None
for r in base_train:
    if r.get('family') == 'green8' and r.get('source') == 'synthetic_exact_quad':
        synthetic_template = r
        break
if synthetic_template is None:
    raise RuntimeError('No synthetic_exact_quad template row found in base train manifest')

edgefit_rows = []
for split in ['train', 'val', 'test']:
    txt = EDGEFIT_ROOT / 'manifests' / f'{split}_labels.txt'
    for line in txt.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        rel, text = line.strip().split(maxsplit=1)
        row = dict(synthetic_template)
        row['img_path'] = str(EDGEFIT_ROOT / rel)
        row['img_rel_path'] = f'green_edgefit_allprov_v3_zhe_guard_yuehu_restore/{rel}'
        row['dataset_name'] = 'green_edgefit_allprov_v3_zhe_guard_yuehu_restore'
        row['split'] = split
        row['text'] = text.strip().upper()
        row['plate_len'] = str(len(row['text']))
        row['family'] = 'green8'
        row['sub_type'] = 'green_small'
        row['source'] = 'synthetic_exact_quad_edgefit_v3_zhe_guard_yuehu_restore'
        row['is_real'] = '0'
        row['need_tilt_aug'] = '1'
        row['preprocess_group'] = 'ccpd_board'
        row['has_bbox'] = '1'
        row['has_quad'] = '1'
        row['can_parse_ccpd_geom'] = '1'
        row['can_perspective'] = '1'
        row['bbox_source'] = 'synthetic_exact_quad_edgefit_v3_zhe_guard_yuehu_restore'
        row['quad_source'] = 'synthetic_exact_quad_edgefit_v3_zhe_guard_yuehu_restore'
        row['ocr_channel_order'] = 'bgr'
        row['ocr_crop_mode'] = 'obb_warp'
        row['ocr_resize_mode'] = 'letterbox'
        row['ocr_resize_kernel'] = 'nn'
        row['ocr_preproc'] = 'none'
        row['ocr_min_occ_ratio'] = '0.9'
        row['ocr_quad_pad_ratio'] = '0.0'
        edgefit_rows.append(row)

new_rows = base_train + [r for r in edgefit_rows if r['split']=='train'] + base_others + [r for r in edgefit_rows if r['split']!='train']
with OUT.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(new_rows)

all_train = [r for r in new_rows if r.get('split')=='train']
prov = Counter((r.get('text') or '')[:1] for r in all_train)
src = Counter(r.get('source') or 'unknown' for r in all_train)
report = {
    'base_manifest': str(BASE),
    'edgefit_root': str(EDGEFIT_ROOT),
    'out_manifest': str(OUT),
    'added_edgefit_total': len(edgefit_rows),
    'added_edgefit_split_counts': dict(Counter(r['split'] for r in edgefit_rows)),
    'train_total_after_merge': len(all_train),
    'train_anhui_ratio_after_merge': sum(1 for r in all_train if (r.get('text') or '').startswith('皖')) / len(all_train),
    'train_zhe_count_after_merge': sum(1 for r in all_train if (r.get('text') or '').startswith('浙')),
    'train_yue_count_after_merge': sum(1 for r in all_train if (r.get('text') or '').startswith('粤')),
    'train_hu_count_after_merge': sum(1 for r in all_train if (r.get('text') or '').startswith('沪')),
    'train_source_breakdown': dict(src),
    'train_top_provinces': prov.most_common(15),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
