#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FIELDS = [
    'img_path','img_rel_path','dataset_name','split','text','plate_len','family','sub_type','source','is_real','need_tilt_aug',
    'preprocess_group','has_bbox','has_quad','can_parse_ccpd_geom','can_perspective','bbox_source','quad_source',
    'ocr_channel_order','ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc','ocr_min_occ_ratio','ocr_quad_pad_ratio'
]


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def infer_cblprd_plain_meta(plate_type: str, text: str):
    mapping = {
        '普通蓝牌': ('normal7', 'blue'),
        '新能源小型车': ('green8', 'green_small'),
        '新能源大型车': ('green8', 'green_large'),
        '单层黄牌': ('special', 'yellow_single'),
        '双层黄牌': ('special', 'yellow_double'),
        '拖拉机绿牌': ('special', 'tractor_green'),
        '黑色车牌': ('special', 'black'),
    }
    family, subtype = mapping.get(plate_type, ('special', 'unknown'))
    if family == 'normal7' and len(text) != 7:
        family = 'special'
    if family == 'green8' and len(text) != 8:
        family = 'special'
    return family, subtype


def rows_from_cblprd_fail(fail_csv: Path, root: Path):
    rows = []
    for r in read_rows(fail_csv):
        family, subtype = infer_cblprd_plain_meta(r['plate_type'], r['text'])
        need_tilt_aug = 0 if family == 'normal7' else 1
        img_path = Path(r['image_path']).resolve()
        rows.append({
            'img_path': str(img_path),
            'img_rel_path': str(img_path.relative_to(root)).replace('\\', '/'),
            'dataset_name': 'cblprd_plain',
            'split': r['split'],
            'text': r['text'],
            'plate_len': len(r['text']),
            'family': family,
            'sub_type': subtype,
            'source': 'real',
            'is_real': 1,
            'need_tilt_aug': need_tilt_aug,
            'preprocess_group': 'plain_plate',
            'has_bbox': 0,
            'has_quad': 0,
            'can_parse_ccpd_geom': 0,
            'can_perspective': 0,
            'bbox_source': 'none',
            'quad_source': 'none',
            'ocr_channel_order': 'bgr',
            'ocr_crop_mode': 'plain_plate',
            'ocr_resize_mode': 'letterbox',
            'ocr_resize_kernel': 'nn',
            'ocr_preproc': 'none',
            'ocr_min_occ_ratio': 1.0,
            'ocr_quad_pad_ratio': 0.0,
        })
    return rows


def rows_from_cblprd_success(success_csv: Path, root: Path):
    rows = []
    for r in read_rows(success_csv):
        img_path = Path(r['output_image']).absolute()
        rows.append({
            'img_path': str(img_path),
            'img_rel_path': str(img_path.relative_to(root)).replace('\\', '/'),
            'dataset_name': 'cblprd_pseudo_geom',
            'split': r['split'],
            'text': r['text'],
            'plate_len': len(r['text']),
            'family': r['family'],
            'sub_type': r['sub_type'],
            'source': 'pseudo_geom',
            'is_real': 1,
            'need_tilt_aug': 0 if r['family'] == 'normal7' else 1,
            'preprocess_group': 'ccpd_board',
            'has_bbox': 1,
            'has_quad': 1,
            'can_parse_ccpd_geom': 1,
            'can_perspective': 1,
            'bbox_source': 'detector_obb',
            'quad_source': 'detector_obb',
            'ocr_channel_order': 'bgr',
            'ocr_crop_mode': 'obb_warp',
            'ocr_resize_mode': 'letterbox',
            'ocr_resize_kernel': 'nn',
            'ocr_preproc': 'none',
            'ocr_min_occ_ratio': 0.90,
            'ocr_quad_pad_ratio': 0.0,
        })
    return rows


def family_for_crpd(text: str):
    return ('normal7', 'blue') if len(text) == 7 else ('special', 'unknown')


def rows_from_crpd_mapping(mapping_csv: Path, root: Path):
    rows = []
    with mapping_csv.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = r['plate_text'].strip().upper()
            family, subtype = family_for_crpd(text)
            new_image = Path(r['new_image'].replace('\\', '/'))
            rel_tail = Path(*new_image.parts[-3:])
            img_path = (root / 'CRPD_CCPD_STRICT_YOLO_v2' / 'plate_crops' / rel_tail).resolve()
            split = 'train'
            for token in ['train', 'val', 'test']:
                if f'/{token}/' in r['new_image'].replace('\\', '/'):
                    split = token
                    break
            rows.append({
                'img_path': str(img_path),
                'img_rel_path': str(img_path.relative_to(root)).replace('\\', '/'),
                'dataset_name': 'crpd_ccpd_strict_yolo_v2',
                'split': split,
                'text': text,
                'plate_len': len(text),
                'family': family,
                'sub_type': subtype,
                'source': 'real',
                'is_real': 1,
                'need_tilt_aug': 1,
                'preprocess_group': 'ccpd_board',
                'has_bbox': 1,
                'has_quad': 1,
                'can_parse_ccpd_geom': 1,
                'can_perspective': 1,
                'bbox_source': 'ccpd_filename',
                'quad_source': 'ccpd_filename',
                'ocr_channel_order': 'bgr',
                'ocr_crop_mode': 'obb_warp',
                'ocr_resize_mode': 'letterbox',
                'ocr_resize_kernel': 'nn',
                'ocr_preproc': 'none',
                'ocr_min_occ_ratio': 0.90,
                'ocr_quad_pad_ratio': 0.0,
            })
    return rows


def summarize(rows):
    return {
        'count': len(rows),
        'datasets': dict(sorted(Counter(r['dataset_name'] for r in rows).items())),
        'families': dict(sorted(Counter(r['family'] for r in rows).items())),
        'sub_types': dict(sorted(Counter(r['sub_type'] for r in rows).items())),
        'sources': dict(sorted(Counter(r['source'] for r in rows).items())),
        'splits': dict(sorted(Counter(r['split'] for r in rows).items())),
        'preprocess_groups': dict(sorted(Counter(r['preprocess_group'] for r in rows).items())),
    }


def main():
    ap = argparse.ArgumentParser(description='Build manifest v3 by extending manifest v2 with CRPD and CBLPRD plain/pseudo data.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--cblprd-success-csv', required=True)
    ap.add_argument('--cblprd-fail-csv', required=True)
    ap.add_argument('--crpd-mapping-csv', required=True)
    ap.add_argument('--root', default='/home/wzzz/LPRNet')
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    base_rows = read_rows(Path(args.base_manifest))
    cblprd_success = rows_from_cblprd_success(Path(args.cblprd_success_csv), root)
    cblprd_fail = rows_from_cblprd_fail(Path(args.cblprd_fail_csv), root)
    crpd_rows = rows_from_crpd_mapping(Path(args.crpd_mapping_csv), root)
    merged = base_rows + crpd_rows + cblprd_success + cblprd_fail

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged)

    summary = {
        'base_count': len(base_rows),
        'crpd_added': len(crpd_rows),
        'cblprd_pseudo_added': len(cblprd_success),
        'cblprd_plain_added': len(cblprd_fail),
        'merged_count': len(merged),
        'crpd_summary': summarize(crpd_rows),
        'cblprd_pseudo_summary': summarize(cblprd_success),
        'cblprd_plain_summary': summarize(cblprd_fail),
        'merged_summary': summarize(merged),
    }
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
