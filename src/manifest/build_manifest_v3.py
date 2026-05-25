#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from lpr_pipeline_policy import apply_board_params

FIELDS = [
    'img_path','img_rel_path','dataset_name','split','text','plate_len','family','sub_type','source','is_real','need_tilt_aug',
    'preprocess_group','has_bbox','has_quad','can_parse_ccpd_geom','can_perspective','bbox_source','quad_source',
    'ocr_channel_order','ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc','ocr_min_occ_ratio','ocr_quad_pad_ratio'
]


def read_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def row_from_success(r, root: Path, dataset_name: str):
    img = Path(r['output_image']).resolve()
    family = r['family']
    row = {
        'img_path': str(img),
        'img_rel_path': str(img.relative_to(root)).replace('\\', '/'),
        'dataset_name': dataset_name,
        'split': r['split'],
        'text': r['text'],
        'plate_len': len(r['text']),
        'family': family,
        'sub_type': r['sub_type'],
        'source': 'pseudo_geom',
        'is_real': 0,
        'need_tilt_aug': 1 if family in {'normal7','green8'} else 0,
        'preprocess_group': 'ccpd_board',
        'has_bbox': 1,
        'has_quad': 1,
        'can_parse_ccpd_geom': 1,
        'can_perspective': 1,
        'bbox_source': 'detector_obb',
        'quad_source': 'detector_obb',
    }
    return apply_board_params(row)


def row_from_failed_plain(r, root: Path, dataset_name: str):
    img = Path(r['output_image']).resolve()
    family = r['family']
    return {
        'img_path': str(img),
        'img_rel_path': str(img.relative_to(root)).replace('\\', '/'),
        'dataset_name': dataset_name,
        'split': r['split'],
        'text': r['text'],
        'plate_len': len(r['text']),
        'family': family,
        'sub_type': r['sub_type'],
        'source': 'real',
        'is_real': 1,
        'need_tilt_aug': 0,
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
    }


def summarize(rows):
    out = {}
    for key in ['dataset_name','family','source','preprocess_group','split']:
        c = Counter(r[key] for r in rows)
        out[key + 's'] = dict(sorted(c.items()))
    out['sample_count'] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--nonccpd-success', required=True)
    ap.add_argument('--cblprd-success', required=True)
    ap.add_argument('--cblprd-failed', required=True)
    ap.add_argument('--crpd-mapping', required=True)
    ap.add_argument('--crpd-root', required=True)
    ap.add_argument('--root', default='/home/wzzz/LPRNet')
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = read_rows(args.base_manifest)

    rows += [row_from_success(r, root, 'targeted_green_missing_18_pseudo_geom' if r['dataset_name']=='targeted_green_missing_18' else 'git_plate_pseudo_geom')
             for r in read_rows(args.nonccpd_success)]

    rows += [row_from_success(r, root, 'CBLPRD_pseudo_geom') for r in read_rows(args.cblprd_success)]

    failed_rows = read_rows(args.cblprd_failed)
    rows += [row_from_failed_plain(r, root, 'CBLPRD_plain_fallback')
             for r in failed_rows if r['family'] == 'normal7' and r['plate_type'] == '普通蓝牌']

    crpd_root = Path(args.crpd_root)
    with open(args.crpd_mapping, 'r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            plate_text = (r['plate_text'] or '').strip().upper()
            if not plate_text:
                continue
            if '挂' in plate_text or '使' in plate_text or '澳' in plate_text or '港' in plate_text or len(plate_text) not in (7, 8):
                family = 'special'; sub_type = 'special'; need_tilt = 0
            elif len(plate_text) == 8:
                family = 'green8'; sub_type = 'green'; need_tilt = 1
            else:
                family = 'normal7'; sub_type = 'blue'; need_tilt = 1
            new_image = Path(r['new_image'])
            plate_crop = crpd_root / 'plate_crops' / '/'.join(new_image.parts[-3:])
            if not plate_crop.exists():
                continue
            row = {
                'img_path': str(plate_crop.resolve()),
                'img_rel_path': str(plate_crop.resolve().relative_to(root)).replace('\\', '/'),
                'dataset_name': 'CRPD_CCPD_STRICT_YOLO_v2',
                'split': plate_crop.parts[-2],
                'text': plate_text,
                'plate_len': len(plate_text),
                'family': family,
                'sub_type': sub_type,
                'source': 'real',
                'is_real': 1,
                'need_tilt_aug': need_tilt,
                'preprocess_group': 'ccpd_board',
                'has_bbox': 1,
                'has_quad': 1,
                'can_parse_ccpd_geom': 1,
                'can_perspective': 1,
                'bbox_source': 'mapping_csv',
                'quad_source': 'mapping_csv',
            }
            rows.append(apply_board_params(row))

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    summary = summarize(rows)
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
