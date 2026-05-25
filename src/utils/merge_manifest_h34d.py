#!/usr/bin/env python3
"""
合并 unified manifest：
1. 基础 CCPD green8 (unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv)
2. synthetic_exact_quad (green_exact_quad_synthetic_v1)
3. edgefit v3 (green_edgefit_v3_allprov)
"""
import csv
from pathlib import Path
from collections import Counter

manifests_to_merge = [
    # 1. 基础 real + CCPD synthetic
    '/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv',
    # 2. Exact quad synthetic
    '/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/train_synthetic_labels_without_su_hu_holdout.txt',
    '/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/val_synthetic_labels.txt',
    '/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/test_synthetic_labels.txt',
    # 3. Edgefit v3
    '/home/wzzz/LPRNet/manifests/unified_manifest_green_edgefit_v3_allprov.csv',
]

out_path = Path('/home/wzzz/LPRNet/manifests/unified_manifest_green_h34d_v3_three_tiers.csv')

all_rows = []
seen_texts_split = {}  # (text, split) -> True

for mpath in manifests_to_merge:
    p = Path(mpath)
    if not p.exists():
        print(f"Warning: {mpath} not found, skipping")
        continue
    
    if p.suffix == '.csv':
        with open(p, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('text', '').strip()
                split = row.get('split', 'train')
                key = (text, split)
                if key not in seen_texts_split:
                    seen_texts_split[key] = True
                    all_rows.append(row)
    else:
        # txt format: "path text"
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)
                if len(parts) != 2:
                    continue
                img_rel, text = parts
                # Determine split from filename
                if 'train' in p.name:
                    split = 'train'
                elif 'val' in p.name:
                    split = 'val'
                elif 'test' in p.name:
                    split = 'test'
                else:
                    split = 'train'
                
                key = (text, split)
                if key not in seen_texts_split:
                    seen_texts_split[key] = True
                    # Build full path
                    if 'green_exact_quad' in str(p):
                        base = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1')
                        img_path = str(base / img_rel)
                    else:
                        img_path = img_rel
                    
                    all_rows.append({
                        'img_path': img_path,
                        'text': text,
                        'bbox': '',
                        'quad': '',
                        'has_quad': '0',
                        'can_parse_ccpd_geom': '0',
                        'source': 'synthetic_exact_quad' if 'exact_quad' in str(p) else 'unknown',
                        'province': text[0] if text else '',
                        'split': split,
                        'is_green': '1',
                        'family': 'green',
                        'comment': 'exact_quad_synthetic',
                    })

# 标准化所有行
fieldnames = ['img_path', 'text', 'bbox', 'quad', 'has_quad', 'can_parse_ccpd_geom', 
              'source', 'province', 'split', 'is_green', 'family', 'comment']

standardized_rows = []
for row in all_rows:
    std_row = {k: row.get(k, '') for k in fieldnames}
    standardized_rows.append(std_row)

# 写入
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(standardized_rows)

all_rows = standardized_rows  # 用于后续统计

print(f"Created: {out_path}")
print(f"  Total: {len(all_rows)} rows")

# 统计
split_counts = Counter(r['split'] for r in all_rows)
source_counts = Counter(r.get('source', 'unknown') for r in all_rows)

print(f"\nBy split:")
for split, count in sorted(split_counts.items()):
    print(f"  {split}: {count}")

print(f"\nBy source:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count}")

# 按省份统计 train
print(f"\n关键省份 Train 统计:")
train_rows = [r for r in all_rows if r['split'] == 'train']
for prov in ['皖', '沪', '苏', '粤', '浙', '湘']:
    prov_rows = [r for r in train_rows if r.get('province') == prov]
    real = len([r for r in prov_rows if r.get('source') == 'real'])
    exact = len([r for r in prov_rows if 'exact_quad' in r.get('source', '')])
    edgefit_simple = len([r for r in prov_rows if 'edgefit_v3_simple' in r.get('source', '')])
    edgefit_hard = len([r for r in prov_rows if 'edgefit_v3_hard' in r.get('source', '')])
    edgefit_extreme = len([r for r in prov_rows if 'edgefit_v3_extreme' in r.get('source', '')])
    total = len(prov_rows)
    print(f"  {prov}: total={total}, real={real}, exact={exact}, edgefit(s/h/e)={edgefit_simple}/{edgefit_hard}/{edgefit_extreme}")
