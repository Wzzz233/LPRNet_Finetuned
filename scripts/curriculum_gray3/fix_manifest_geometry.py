#!/usr/bin/env python3
"""
为现有 curriculum_gray3 manifest 补充几何字段，
使 CCPD/CRPD 原始大图数据能被正确透视裁剪（而非整张 resize）。
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("manifests/curriculum_gray3")


def read_crpd_quad(img_path: str) -> tuple:
    """根据 CRPD 图像路径找到对应 label txt，读取 quad 坐标。"""
    p = Path(img_path)
    # path: .../CRPD_single/train/images/48_1053.jpg
    # label: .../CRPD_single/train/labels/48_1053.txt
    label_dir = p.parent.parent / "labels"
    label_file = label_dir / (p.stem + ".txt")
    if not label_file.exists():
        return None
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            # parts[0:8] = x1 y1 x2 y2 x3 y3 x4 y4
            return tuple(parts[0:8])
    return None


def fix_manifest(in_path: Path, out_path: Path):
    print(f"\nFixing {in_path.name} -> {out_path.name}")
    
    with open(in_path) as f:
        rows = list(csv.DictReader(f))
    
    # 新的字段列表
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
        out = dict(r)
        
        if source in ('ccpd2019', 'ccpd2020'):
            # CCPD: quad 在文件名中，运行时解析
            out['preprocess_group'] = 'ccpd_board'
            out['has_quad'] = '1'
            out['can_parse_ccpd_geom'] = '1'
            out['can_perspective'] = '1'
            out['quad_source'] = 'ccpd_filename'
            out['bbox_source'] = 'ccpd_filename'
            out['ocr_quad_pad_ratio'] = '0.0'
            stats['ccpd_board'] += 1
            
        elif source.startswith('crpd'):
            # CRPD: 从 label txt 读取 quad
            quad = read_crpd_quad(r['img_path'])
            if quad is not None:
                out['preprocess_group'] = 'ccpd_board'
                out['has_quad'] = '1'
                out['can_parse_ccpd_geom'] = '1'
                out['can_perspective'] = '1'
                out['quad_source'] = 'crpd_label'
                out['bbox_source'] = 'quad_derived'
                out['ocr_quad_pad_ratio'] = '0.0'
                out['quad_1x'] = quad[0]
                out['quad_1y'] = quad[1]
                out['quad_2x'] = quad[2]
                out['quad_2y'] = quad[3]
                out['quad_3x'] = quad[4]
                out['quad_3y'] = quad[5]
                out['quad_4x'] = quad[6]
                out['quad_4y'] = quad[7]
                stats['crpd_board'] += 1
            else:
                # 找不到 label，fallback 到 plain_plate
                out['preprocess_group'] = 'plain_plate'
                out['has_quad'] = '0'
                out['can_parse_ccpd_geom'] = '0'
                out['can_perspective'] = '0'
                out['quad_source'] = 'none'
                out['bbox_source'] = 'none'
                out['ocr_quad_pad_ratio'] = '0.0'
                stats['crpd_plain'] += 1
                
        else:
            # cblprd, green_exact_quad, green_edgefit_* 已经是裁剪好的 patch
            out['preprocess_group'] = 'plain_plate'
            out['has_quad'] = '0'
            out['can_parse_ccpd_geom'] = '0'
            out['can_perspective'] = '0'
            out['quad_source'] = 'none'
            out['bbox_source'] = 'none'
            out['ocr_quad_pad_ratio'] = '0.0'
            stats['plain_plate'] += 1
        
        out_rows.append(out)
    
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    
    print(f"  Wrote {len(out_rows)} rows")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")


def main():
    manifests = [
        'train_stageA.csv',
        'val.csv',
        'test_blue_simple.csv',
        'test_blue_hard.csv',
        'test_green_real.csv',
        'test_green_simple.csv',
        'test_green_hard.csv',
        'test_green_extreme.csv',
    ]
    
    for name in manifests:
        in_path = OUT_DIR / name
        out_path = OUT_DIR / name
        if not in_path.exists():
            print(f"  Skip {name} (not found)")
            continue
        fix_manifest(in_path, out_path)
    
    print("\n=== Manifest geometry fix complete ===")


if __name__ == '__main__':
    main()
