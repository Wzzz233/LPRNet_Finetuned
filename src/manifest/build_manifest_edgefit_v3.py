#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 green_edgefit_v3_allprov 创建 unified manifest
"""
import csv
import json
from pathlib import Path
from datetime import datetime

edgefit_root = Path('/home/wzzz/LPRNet/green_edgefit_v3_allprov')
out_manifest = Path('/home/wzzz/LPRNet/manifests/unified_manifest_green_edgefit_v3_allprov.csv')

tsv_path = edgefit_root / 'details' / 'accepted.tsv'

rows = []
with open(tsv_path, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for r in reader:
        quad = json.loads(r['quad'])
        rows.append({
            'img_path': str(edgefit_root / r['rel_path']),
            'text': r['text'],
            'bbox': '',
            'quad': r['quad'],
            'has_quad': '1',
            'can_parse_ccpd_geom': '1',
            'source': f"synthetic_edgefit_v3_{r['difficulty']}",
            'province': r['province'],
            'split': r['split'],
            'is_green': '1',
            'family': 'green',
            'comment': f"edgefit_v3_{r['difficulty']}",
        })

# 写入 manifest
fieldnames = ['img_path', 'text', 'bbox', 'quad', 'has_quad', 'can_parse_ccpd_geom', 
              'source', 'province', 'split', 'is_green', 'family', 'comment']

with open(out_manifest, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Created: {out_manifest}")
print(f"  Total: {len(rows)} rows")

# 统计
from collections import Counter
split_counts = Counter(r['split'] for r in rows)
source_counts = Counter(r['source'] for r in rows)

print(f"\nBy split:")
for split, count in sorted(split_counts.items()):
    print(f"  {split}: {count}")

print(f"\nBy source:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count}")
