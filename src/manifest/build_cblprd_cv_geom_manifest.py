#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cblprd_cv_geom_manifest.py

对 CBLPRD-330k 使用灰度方差定位车牌区域，生成 CCPD 格式文件名的 symlink，
输出与 unified_manifest_v4 格式兼容的 CSV manifest。

无需 OBB 神经网络检测器，对 GAN 生成图更稳健。
"""

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# --- 复用 auto_label_cblprd_obb.py 的常量和辅助函数 ---

PLATE_TYPE_META = {
    '普通蓝牌': ('normal7', 'blue'),
    '新能源小型车': ('green8', 'green_small'),
    '新能源大型车': ('green8', 'green_large'),
    '单层黄牌': ('special', 'yellow_single'),
    '双层黄牌': ('special', 'yellow_double'),
    '拖拉机绿牌': ('special', 'tractor_green'),
    '黑色车牌': ('special', 'black'),
}

# LPRNet CHARS 中允许的字符（不含 I O - 等 CBLPRD 特有字符）
LPRNET_CHARS = set([
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '学', '警', '挂', '港', '澳',
])

MANIFEST_FIELDNAMES = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text',
    'family', 'sub_type', 'source', 'pos0_data_source',
    'has_bbox', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'preprocess_group',
    'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode',
    'ocr_resize_kernel', 'ocr_preproc',
    'ocr_min_occ_ratio', 'ocr_quad_pad_ratio',
]


@dataclass
class Sample:
    image_path: Path
    rel_path: str
    text: str
    plate_type: str
    split: str
    family: str
    sub_type: str


def safe_token(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or '\u4e00' <= ch <= '\u9fff' or ch in '._-':
            out.append(ch)
        else:
            out.append('_')
    token = ''.join(out).strip('_')
    return token or 'x'


def text_valid_for_lprnet(text: str) -> bool:
    """只保留所有字符都在 LPRNet CHARS 中的样本"""
    return all(ch in LPRNET_CHARS for ch in text)


def box_from_quad(quad) -> Tuple[int, int, int, int]:
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def quad_to_ccpd_text(quad) -> str:
    return '_'.join(f'{int(round(x))}&{int(round(y))}' for x, y in quad)


def make_pseudo_ccpd_name(sample: Sample, quad, suffix='jpg') -> str:
    x1, y1, x2, y2 = box_from_quad(quad)
    bbox = f'{x1}&{y1}_{x2}&{y2}'
    quad_text = quad_to_ccpd_text(quad)
    text_tag = safe_token(sample.text)
    origin_tag = safe_token(Path(sample.rel_path).with_suffix('').as_posix())
    ptype_tag = safe_token(sample.plate_type)
    return f'cvcrop-0-{bbox}-{quad_text}-{text_tag}-{ptype_tag}-0-{origin_tag}.{suffix}'


def locate_plate_variance(img_path: Path) -> Tuple[Tuple[int, int, int, int], str]:
    """
    用灰度方差定位车牌活跃区域。
    返回 (x1, y1, x2, y2) 和方法标记 ('variance' | 'fallback')。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 'read_error'
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

    col_var = np.var(gray, axis=0)
    row_var = np.var(gray, axis=1)

    col_thresh = max(float(np.max(col_var)) * 0.1, 5.0)
    row_thresh = max(float(np.max(row_var)) * 0.1, 5.0)

    active_cols = np.where(col_var > col_thresh)[0]
    active_rows = np.where(row_var > row_thresh)[0]

    if len(active_cols) < 5 or len(active_rows) < 3:
        return (0, 0, w - 1, h - 1), 'fallback'

    x1, x2 = int(active_cols[0]), int(active_cols[-1])
    y1, y2 = int(active_rows[0]), int(active_rows[-1])
    return (x1, y1, x2, y2), 'variance'


def parse_data_txt(txt_path: Path, root: Path) -> Tuple[List[Sample], List[dict]]:
    """解析 CBLPRD-330k 的 data.txt / train.txt / val.txt"""
    samples: List[Sample] = []
    skipped: List[dict] = []
    with txt_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                skipped.append({'line_no': line_no, 'reason': 'bad_line', 'raw': line})
                continue
            rel_path, text, plate_type = parts
            image_path = root / rel_path
            if not image_path.exists():
                skipped.append({'line_no': line_no, 'reason': 'missing_image', 'rel_path': rel_path})
                continue
            if not text_valid_for_lprnet(text):
                bad = ''.join(sorted({ch for ch in text if ch not in LPRNET_CHARS}))
                skipped.append({'line_no': line_no, 'reason': 'invalid_chars', 'rel_path': rel_path, 'bad': bad})
                continue
            family, sub_type = PLATE_TYPE_META.get(plate_type, ('special', safe_token(plate_type)))
            samples.append(Sample(
                image_path=image_path,
                rel_path=rel_path.replace('\\', '/'),
                text=text,
                plate_type=plate_type,
                split='',  # 由调用方指定
                family=family,
                sub_type=sub_type,
            ))
    return samples, skipped


def make_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except FileExistsError:
        pass


def process_split(
    samples: List[Sample],
    split: str,
    out_root: Path,
    rows: list,
    stats: Counter,
    lpr_root: Path,
):
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        sample.split = split
        bbox, method = locate_plate_variance(sample.image_path)
        if bbox is None:
            stats['read_error'] += 1
            continue

        x1, y1, x2, y2 = bbox
        quad = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        ccpd_name = make_pseudo_ccpd_name(sample, quad)
        dst = split_dir / ccpd_name
        make_symlink(sample.image_path, dst)

        # img_rel_path 相对于 LPRNet 项目根
        img_rel = dst.resolve().relative_to(lpr_root).as_posix()

        rows.append({
            'img_path': str(dst),
            'img_rel_path': img_rel,
            'dataset_name': 'cblprd_330k',
            'split': split,
            'text': sample.text,
            'family': sample.family,
            'sub_type': sample.sub_type,
            'source': 'cblprd_cv_geom',
            'pos0_data_source': 'cblprd',
            'has_bbox': '1',
            'has_quad': '1',
            'can_parse_ccpd_geom': '1',
            'can_perspective': '1',
            'preprocess_group': 'ccpd_board',
            'ocr_channel_order': 'bgr',
            'ocr_crop_mode': 'obb_warp',
            'ocr_resize_mode': 'letterbox',
            'ocr_resize_kernel': 'nn',
            'ocr_preproc': 'none',
            'ocr_min_occ_ratio': '0.90',
            'ocr_quad_pad_ratio': '0.0',
        })
        stats[f'{split}_ok'] += 1
        stats[f'method_{method}'] += 1
        stats[f'family_{sample.family}'] += 1
        stats[f'ptype_{sample.plate_type}'] += 1


def main():
    parser = argparse.ArgumentParser(description='Build CBLPRD-330k CV-geom manifest')
    parser.add_argument('--cblprd_root', default='CBLPRD-330k_v1', help='CBLPRD-330k 根目录')
    parser.add_argument('--lpr_root', default='.', help='LPRNet 项目根目录（用于计算相对路径）')
    parser.add_argument('--out_dir', default='CBLPRD-330k_v1/cblprd_cv_geom', help='symlink 输出目录')
    parser.add_argument('--out_manifest', default='manifests/cblprd_cv_geom_manifest.csv', help='输出 manifest CSV')
    parser.add_argument('--out_summary', default='manifests/cblprd_cv_geom_summary.json', help='输出统计 JSON')
    parser.add_argument('--use_data_txt', action='store_true', help='使用 data.txt 而非 train.txt+val.txt（不区分 split）')
    parser.add_argument('--max_samples', default=0, type=int, help='最多处理样本数（0=全部，用于测试）')
    args = parser.parse_args()

    cblprd_root = Path(args.cblprd_root)
    lpr_root = Path(args.lpr_root).resolve()
    out_root = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)
    out_summary = Path(args.out_summary)
    img_root = cblprd_root  # 图片路径相对于 cblprd_root

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    stats: Counter = Counter()

    if args.use_data_txt:
        txt_path = cblprd_root / 'data.txt'
        print(f'[Parse] Reading {txt_path}')
        samples, skipped = parse_data_txt(txt_path, img_root)
        if args.max_samples > 0:
            samples = samples[:args.max_samples]
        stats['skipped_parse'] += len(skipped)
        print(f'[Parse] {len(samples)} valid, {len(skipped)} skipped')
        process_split(samples, 'train', out_root, rows, stats, lpr_root)
    else:
        for split_name, txt_name in [('train', 'train.txt'), ('val', 'val.txt')]:
            txt_path = cblprd_root / txt_name
            if not txt_path.exists():
                print(f'[Skip] {txt_path} not found')
                continue
            print(f'[Parse] Reading {txt_path}')
            samples, skipped = parse_data_txt(txt_path, img_root)
            if args.max_samples > 0 and split_name == 'train':
                samples = samples[:args.max_samples]
            stats['skipped_parse'] += len(skipped)
            print(f'[Parse] split={split_name} {len(samples)} valid, {len(skipped)} skipped')
            process_split(samples, split_name, out_root, rows, stats, lpr_root)

    # 写 manifest
    with out_manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f'[Output] manifest: {out_manifest} ({len(rows)} rows)')

    # 省份分布统计
    province_counts: Counter = Counter()
    for row in rows:
        if row['text']:
            province_counts[row['text'][0]] += 1
    top_provinces = province_counts.most_common(35)

    # 写 summary
    summary = {
        'total_rows': len(rows),
        'stats': dict(stats),
        'province_distribution': {k: v for k, v in top_provinces},
    }
    with out_summary.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'[Output] summary: {out_summary}')

    print('\n--- 省份分布 top 35 ---')
    for prov, cnt in top_provinces:
        bar = '█' * (cnt // 500)
        print(f'  {prov}: {cnt:6d} {bar}')

    print('\n--- Stats ---')
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
