#!/usr/bin/env python3
"""
Build separate train/val manifest CSVs for yellow and special LPRNet.
"""
import csv
import os
import re
from pathlib import Path
from collections import Counter

SRC = Path(__file__).resolve().parent.parent.parent
CBLPRD_ROOT = SRC / 'datasets' / 'CBLPRD-330k_v1'
CBLPRD_TRAIN_TXT = CBLPRD_ROOT / 'train.txt'
CBLPRD_VAL_TXT = CBLPRD_ROOT / 'val.txt'
GIT_PLATE_TRAIN = SRC / 'datasets' / 'git_plate' / 'train'
GIT_PLATE_VAL = SRC / 'datasets' / 'git_plate' / 'val' / 'val_verify'
CRPD_MANIFEST = SRC / 'manifests' / 'crpd_all_raw_board_v1_supported.csv'
OUT_DIR = SRC / 'manifests'
OUT_DIR.mkdir(exist_ok=True)

YELLOW_PLATE_TYPES = {'单层黄牌', '双层黄牌'}
SPECIAL_PLATE_TYPES = {'黑色车牌'}
SPECIAL_CHARS = set('警使领澳港')


def parse_git_plate_text(fname):
    stem = fname.rsplit('.', 1)[0]
    # Remove trailing _index, _distortN, _stretchN
    stem = re.sub(r'_(distort|stretch)?\d+$', '', stem, flags=re.IGNORECASE)
    if '_' in stem:
        stem = stem.split('_')[0]
    return stem


def add_plain_row(rows, img_path, text, split, source):
    if not os.path.exists(img_path):
        return
    rows.append({
        'img_path': img_path,
        'text': text,
        'split': split,
        'preprocess_group': 'plain_plate',
        'source': source,
    })


def add_ccpd_row(rows, crpd_row, split):
    img_path = crpd_row['img_path']
    if not os.path.exists(img_path):
        return
    rows.append({
        'img_path': img_path,
        'text': crpd_row['text'],
        'split': split,
        'preprocess_group': 'ccpd_board',
        'source': 'crpd_' + crpd_row.get('sub_type', 'special'),
        'img_rel_path': crpd_row.get('img_rel_path', img_path),
        'quad_1x': crpd_row.get('quad_1x', ''),
        'quad_1y': crpd_row.get('quad_1y', ''),
        'quad_2x': crpd_row.get('quad_2x', ''),
        'quad_2y': crpd_row.get('quad_2y', ''),
        'quad_3x': crpd_row.get('quad_3x', ''),
        'quad_3y': crpd_row.get('quad_3y', ''),
        'quad_4x': crpd_row.get('quad_4x', ''),
        'quad_4y': crpd_row.get('quad_4y', ''),
    })


def load_crpd_rows():
    crpd = {'yellow_single': [], 'yellow_double': [], 'special': []}
    if not CRPD_MANIFEST.exists():
        print(f'[WARN] CRPD manifest not found: {CRPD_MANIFEST}')
        return crpd
    with open(CRPD_MANIFEST, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            st = row.get('sub_type', '').strip()
            if st in crpd:
                crpd[st].append(row)
    print(f'  CRPD loaded: yellow_single={len(crpd["yellow_single"])} '
          f'yellow_double={len(crpd["yellow_double"])} '
          f'special={len(crpd["special"])}')
    return crpd


def build_yellow(crpd):
    rows = []
    # CBLPRD yellow: train.txt -> train, val.txt -> test
    for txt_path, split in [(CBLPRD_TRAIN_TXT, 'train'), (CBLPRD_VAL_TXT, 'test')]:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                if len(parts) != 3:
                    continue
                rel, text, ptype = parts
                ptype = ptype.strip()
                if ptype not in YELLOW_PLATE_TYPES:
                    continue
                add_plain_row(rows, str(CBLPRD_ROOT / rel), text, split, 'cblprd_yellow')

    # CRPD yellow: all as test (eval holdout from real scenes)
    for st in ('yellow_single', 'yellow_double'):
        for r in crpd.get(st, []):
            add_ccpd_row(rows, r, 'test')

    print(f'  Yellow total: {len(rows)}')
    return rows


def build_special(crpd):
    rows = []
    # CBLPRD 黑色车牌
    for txt_path, split in [(CBLPRD_TRAIN_TXT, 'train'), (CBLPRD_VAL_TXT, 'test')]:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                if len(parts) != 3:
                    continue
                rel, text, ptype = parts
                ptype = ptype.strip()
                if ptype not in SPECIAL_PLATE_TYPES:
                    continue
                add_plain_row(rows, str(CBLPRD_ROOT / rel), text, split, 'cblprd_special')

    # git_plate only 警使领澳港
    for img_dir, split in [(GIT_PLATE_TRAIN, 'train'), (GIT_PLATE_VAL, 'test')]:
        if not img_dir.exists():
            continue
        for fname in os.listdir(str(img_dir)):
            if not fname.endswith('.jpg'):
                continue
            if not any(c in fname for c in SPECIAL_CHARS):
                continue
            text = parse_git_plate_text(fname)
            if not text or '民航' in text:
                continue
            add_plain_row(rows, str(img_dir / fname), text, split, 'git_plate_special')

    # CRPD special
    for r in crpd.get('special', []):
        add_ccpd_row(rows, r, 'test')

    print(f'  Special total: {len(rows)}')
    return rows


def write_manifest(rows, name):
    # Write train split
    train_rows = [r for r in rows if r['split'] == 'train']
    test_rows = [r for r in rows if r['split'] == 'test']

    fieldnames = ['img_path', 'text', 'split', 'preprocess_group', 'source',
                  'img_rel_path', 'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
                  'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y']

    for subset_rows, suffix in [(train_rows, 'train'), (test_rows, 'test')]:
        out_path = OUT_DIR / f'{name}_{suffix}.csv'
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset_rows)
        print(f'  Written: {out_path} ({len(subset_rows)} samples)')

    src_counts = Counter(r['source'] for r in rows)
    print(f'  Source breakdown: {dict(src_counts)}')
    return train_rows, test_rows


if __name__ == '__main__':
    crpd = load_crpd_rows()

    print('\n=== Yellow LPRNet Manifest ===')
    yellow_rows = build_yellow(crpd)
    write_manifest(yellow_rows, 'yellow')

    print('\n=== Special LPRNet Manifest ===')
    special_rows = build_special(crpd)
    write_manifest(special_rows, 'special')

    print('\nDone.')
