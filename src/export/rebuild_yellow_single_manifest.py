#!/usr/bin/env python3
"""Rebuild yellow manifest: single-layer + 挂-suffix double-layer only."""
import csv, os
from pathlib import Path
from collections import defaultdict

SRC = Path('/home/wzzz/LPRNet')
CBLPRD_ROOT = SRC / 'datasets' / 'CBLPRD-330k_v1'
CBLPRD_TRAIN_TXT = CBLPRD_ROOT / 'train.txt'
CBLPRD_VAL_TXT = CBLPRD_ROOT / 'val.txt'
OUT_DIR = SRC / 'manifests'

def build():
    rows = []
    for txt_path, split in [(CBLPRD_TRAIN_TXT, 'train'), (CBLPRD_VAL_TXT, 'test')]:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(maxsplit=2)
                if len(parts) != 3: continue
                rel, text, ptype = parts
                ptype = ptype.strip()
                
                # Single-layer: include ALL
                if ptype == '单层黄牌':
                    pass
                # Double-layer: include ONLY those with 挂 suffix
                elif ptype == '双层黄牌' and '挂' in text:
                    pass
                else:
                    continue
                    
                img_path = str(CBLPRD_ROOT / rel)
                if not os.path.exists(img_path): continue
                rows.append({
                    'img_path': img_path, 'text': text, 'split': split,
                    'preprocess_group': 'plain_plate', 'source': 'cblprd_yellow_single',
                })
    
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    
    # Separate val into regular + hard (学/挂)
    hard_val = [r for r in test if any(c in r['text'] for c in '学挂')]
    regular_val = [r for r in test if r not in hard_val]
    
    for data, name in [(train, 'yellow_single_train'),
                        (regular_val, 'yellow_single_val'),
                        (hard_val, 'yellow_single_hard_val')]:
        path = OUT_DIR / f'{name}.csv'
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['img_path','text','split','preprocess_group','source'])
            w.writeheader()
            w.writerows(data)
        
        xue = sum(1 for r in data if '学' in r['text'])
        gua = sum(1 for r in data if '挂' in r['text'])
        print(f'{path.name:45s}: {len(data):5d} (学={xue}, 挂={gua})')

if __name__ == '__main__':
    build()
