#!/usr/bin/env python3
"""
为 curriculum_gray3_stagea_redesign manifests 补充几何字段，
让 CCPD/CRPD 原始大图在训练时走 ccpd_board / quad warp，
避免被 plain_plate 整图缩放到 94x24。
"""

import csv
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path('/home/wzzz/LPRNet/manifests/curriculum_gray3_stagea_redesign')


def read_crpd_quad(img_path: str):
    p = Path(img_path)
    label_dir = p.parent.parent / 'labels'
    label_file = label_dir / (p.stem + '.txt')
    if not label_file.exists():
        return None
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            return tuple(parts[0:8])
    return None


def fix_manifest(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    fieldnames = [
        'img_path', 'text', 'family', 'source', 'split',
        'preprocess_group',
        'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
        'quad_source', 'bbox_source',
        'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
        'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
        'ocr_quad_pad_ratio',
    ]

    out_rows = []
    stats = defaultdict(int)
    for r in rows:
        source = r['source']
        out = {k: r.get(k, '') for k in ['img_path', 'text', 'family', 'source', 'split']}
        if source in ('ccpd2019', 'ccpd2020'):
            out.update({
                'preprocess_group': 'ccpd_board',
                'has_quad': '1',
                'can_parse_ccpd_geom': '1',
                'can_perspective': '1',
                'quad_source': 'ccpd_filename',
                'bbox_source': 'ccpd_filename',
                'ocr_quad_pad_ratio': '0.0',
                'quad_1x': '', 'quad_1y': '', 'quad_2x': '', 'quad_2y': '',
                'quad_3x': '', 'quad_3y': '', 'quad_4x': '', 'quad_4y': '',
            })
            stats['ccpd_board'] += 1
        elif source.startswith('crpd'):
            quad = read_crpd_quad(r['img_path'])
            if quad is not None:
                out.update({
                    'preprocess_group': 'ccpd_board',
                    'has_quad': '1',
                    'can_parse_ccpd_geom': '1',
                    'can_perspective': '1',
                    'quad_source': 'crpd_label',
                    'bbox_source': 'quad_derived',
                    'ocr_quad_pad_ratio': '0.0',
                    'quad_1x': quad[0], 'quad_1y': quad[1], 'quad_2x': quad[2], 'quad_2y': quad[3],
                    'quad_3x': quad[4], 'quad_3y': quad[5], 'quad_4x': quad[6], 'quad_4y': quad[7],
                })
                stats['crpd_board'] += 1
            else:
                out.update({
                    'preprocess_group': 'plain_plate',
                    'has_quad': '0',
                    'can_parse_ccpd_geom': '0',
                    'can_perspective': '0',
                    'quad_source': 'none',
                    'bbox_source': 'none',
                    'ocr_quad_pad_ratio': '0.0',
                    'quad_1x': '', 'quad_1y': '', 'quad_2x': '', 'quad_2y': '',
                    'quad_3x': '', 'quad_3y': '', 'quad_4x': '', 'quad_4y': '',
                })
                stats['crpd_plain'] += 1
        else:
            out.update({
                'preprocess_group': 'plain_plate',
                'has_quad': '0',
                'can_parse_ccpd_geom': '0',
                'can_perspective': '0',
                'quad_source': 'none',
                'bbox_source': 'none',
                'ocr_quad_pad_ratio': '0.0',
                'quad_1x': '', 'quad_1y': '', 'quad_2x': '', 'quad_2y': '',
                'quad_3x': '', 'quad_3y': '', 'quad_4x': '', 'quad_4y': '',
            })
            stats['plain_plate'] += 1
        out_rows.append(out)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(path)
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}')


def main():
    names = [
        'train_stageA.csv',
        'val.csv',
        'proxy_stageA_blue_simple.csv',
        'proxy_stageA_green_simple.csv',
        'proxy_stageA_mixed_foundation.csv',
    ]
    for name in names:
        p = OUT_DIR / name
        if p.exists():
            fix_manifest(p)


if __name__ == '__main__':
    main()
