#!/usr/bin/env python3
"""Rebuild yellow manifest: CBLPRD + CRPD yellow with quad in train."""
import csv, os, re
from pathlib import Path
from collections import defaultdict

SRC = Path('/home/wzzz/LPRNet')
CBLPRD_ROOT = SRC / 'datasets' / 'CBLPRD-330k_v1'
CBLPRD_TRAIN_TXT = CBLPRD_ROOT / 'train.txt'
CBLPRD_VAL_TXT = CBLPRD_ROOT / 'val.txt'
OUT_DIR = SRC / 'manifests'

YELLOW_PLATE_TYPES = {'单层黄牌', '双层黄牌'}

def parse_crpd_label_path(img_path):
    try:
        real = os.readlink(img_path)
    except OSError:
        return None
    m = re.match(r'.*?CRPD_all/(CRPD_\w+)/(train|test|val)/images/(.+)\.jpg', real)
    if not m:
        return None
    subdir = m.group(1); split = m.group(2); stem = m.group(3)
    return f'/home/wzzz/LPRNet/datasets/CRPD_all/{subdir}/{split}/labels/{stem}.txt'

def read_crpd_quad(label_path, gt_text):
    if not label_path or not os.path.exists(label_path):
        return None
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 10 and parts[9] == gt_text:
                return {'quad_1x': parts[0], 'quad_1y': parts[1],
                        'quad_2x': parts[2], 'quad_2y': parts[3],
                        'quad_3x': parts[4], 'quad_3y': parts[5],
                        'quad_4x': parts[6], 'quad_4y': parts[7]}
    return None

def build():
    rows = []
    
    # 1. CBLPRD yellow - ALL train + val as train (for max data), use subset of val as test
    for txt_path, split_tag in [(CBLPRD_TRAIN_TXT, 'train'), (CBLPRD_VAL_TXT, 'test')]:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(maxsplit=2)
                if len(parts) != 3: continue
                rel, text, ptype = parts
                ptype = ptype.strip()
                if ptype not in YELLOW_PLATE_TYPES: continue
                img_path = str(CBLPRD_ROOT / rel)
                if not os.path.exists(img_path): continue
                rows.append({
                    'img_path': img_path, 'text': text, 'split': split_tag,
                    'preprocess_group': 'plain_plate', 'source': 'cblprd_yellow',
                })
    
    # 2. CRPD yellow - take ALL as train (adds real-world quad-warp samples)
    crpd_manifest = SRC / 'manifests' / 'crpd_all_raw_board_v1_supported.csv'
    with open(crpd_manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            st = row.get('sub_type', '').strip()
            if st not in ('yellow_single', 'yellow_double'): continue
            img_path = row['img_path']
            if not os.path.exists(img_path): continue
            
            label_path = parse_crpd_label_path(img_path)
            quad = read_crpd_quad(label_path, row['text'])
            
            r = {
                'img_path': img_path, 'text': row['text'], 'split': 'train',
                'preprocess_group': 'ccpd_board',
                'source': f'crpd_{st}',
                'img_rel_path': row.get('img_rel_path', img_path),
            }
            if quad:
                r.update(quad)
            else:
                # fallback to plain_plate if no quad
                r['preprocess_group'] = 'plain_plate'
            rows.append(r)
    
    # Write
    train_rows = [r for r in rows if r['split'] == 'train']
    test_rows = [r for r in rows if r['split'] == 'test']
    
    fieldnames = ['img_path', 'text', 'split', 'preprocess_group', 'source',
                  'img_rel_path', 'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
                  'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y']
    
    for subset_rows, suffix in [(train_rows, 'train'), (test_rows, 'test')]:
        out_path = OUT_DIR / f'yellow_{suffix}.csv'
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset_rows)
        src_counts = defaultdict(int)
        pg_counts = defaultdict(int)
        for r in subset_rows:
            src_counts[r['source']] += 1
            pg_counts[r.get('preprocess_group', '?')] += 1
        print(f'{out_path}: {len(subset_rows)} samples')
        print(f'  Sources: {dict(src_counts)}')
        print(f'  Preprocess: {dict(pg_counts)}')

if __name__ == '__main__':
    build()
