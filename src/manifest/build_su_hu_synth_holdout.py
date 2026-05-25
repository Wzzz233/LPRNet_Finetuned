#!/usr/bin/env python3
from pathlib import Path

BASE = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests')
TRAIN = BASE / 'train_synthetic_labels.txt'
OUT = BASE / 'su_hu_holdout_synthetic_labels.txt'
OUT_REMAIN = BASE / 'train_synthetic_labels_without_su_hu_holdout.txt'
TARGETS = {'苏': 80, '沪': 80}

lines = [line for line in TRAIN.read_text(encoding='utf-8').splitlines() if line.strip()]
selected = []
remain = []
counts = {'苏': 0, '沪': 0}
for line in lines:
    rel, text = line.strip().split(maxsplit=1)
    prov = text.strip()[0]
    if prov in TARGETS and counts[prov] < TARGETS[prov]:
        selected.append(line)
        counts[prov] += 1
    else:
        remain.append(line)
OUT.write_text('\n'.join(selected) + '\n', encoding='utf-8')
OUT_REMAIN.write_text('\n'.join(remain) + '\n', encoding='utf-8')
print({'selected_total': len(selected), 'counts': counts, 'remain_total': len(remain), 'out': str(OUT), 'out_remain': str(OUT_REMAIN)})
