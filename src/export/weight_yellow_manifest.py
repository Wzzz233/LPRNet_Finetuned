#!/usr/bin/env python3
"""Create weighted training manifest: oversample 学/挂 3x."""
import csv, os
from pathlib import Path

manifest = Path('/home/wzzz/LPRNet/manifests/yellow_single_train.csv')
out_path = Path('/home/wzzz/LPRNet/manifests/yellow_single_train_weighted.csv')

rows = []
xue_rows = []
gua_rows = []
other_rows = []

with open(manifest, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        text = row['text']
        if '学' in text:
            xue_rows.append(dict(row))
        elif '挂' in text:
            gua_rows.append(dict(row))
        else:
            other_rows.append(dict(row))

# Duplicate 学 and 挂 3x for weighted sampling
weighted_rows = other_rows + xue_rows * 3 + gua_rows * 3
print(f'Original: {len(xue_rows)} 学, {len(gua_rows)} 挂, {len(other_rows)} other = {len(xue_rows)+len(gua_rows)+len(other_rows)}')
print(f'Weighted: {len(xue_rows)*3} 学, {len(gua_rows)*3} 挂, {len(other_rows)} other = {len(weighted_rows)}')

with open(out_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(weighted_rows)

print(f'Saved to {out_path}')
