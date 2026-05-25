#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from lpr_pipeline_policy import apply_board_params, BOARD_PARAM_EXPECTED
from load_data import CHARS_DICT

FIELDS = [
    'img_path','img_rel_path','dataset_name','split','text','plate_len','family','sub_type','source','is_real','need_tilt_aug',
    'preprocess_group','has_bbox','has_quad','can_parse_ccpd_geom','can_perspective','bbox_source','quad_source',
    'ocr_channel_order','ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc','ocr_min_occ_ratio','ocr_quad_pad_ratio'
]

ALLOWED_SPLITS = {'train', 'val', 'test', 'eval'}
SUPPORTED_CHARS = set(CHARS_DICT.keys())


def read_label_txt(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, text = line.split(maxsplit=1)
            rows.append((rel.replace('\\', '/'), text.strip().upper()))
    return rows


def read_csv_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def text_supported(text: str) -> bool:
    return all(ch in SUPPORTED_CHARS for ch in text)


def add_row(rows, row):
    rows.append(row)


def build_ccpd_rows(root: Path, dataset_name: str, img_root: Path, txt_map: dict, family: str, sub_type: str):
    rows = []
    skipped = []
    for split, txt_path in txt_map.items():
        for rel, text in read_label_txt(txt_path):
            if not text_supported(text):
                skipped.append({'dataset_name': dataset_name, 'split': split, 'text': text, 'reason': 'unsupported_chars'})
                continue
            img_path = (img_root / rel).resolve()
            row = {
                'img_path': str(img_path),
                'img_rel_path': str(img_path.relative_to(root)).replace('\\', '/'),
                'dataset_name': dataset_name,
                'split': split,
                'text': text,
                'plate_len': len(text),
                'family': family,
                'sub_type': sub_type,
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
            }
            rows.append(apply_board_params(row))
    return rows, skipped


def row_from_nonccpd_success(r: dict, root: Path) -> dict:
    dataset_map = {
        'targeted_green_missing_18': 'targeted_green_missing_18_pseudo_geom',
        'git_plate': 'git_plate_pseudo_geom',
        'CBLPRD': 'cblprd_pseudo_geom',
        'CBLPRD-330k_v1': 'cblprd_pseudo_geom',
    }
    source_name = r.get('dataset_name') or r.get('source_name') or ''
    if not source_name:
        image_path = str(r.get('image_path') or '')
        rel_path = str(r.get('rel_path') or '')
        output_image = str(r.get('output_image') or '')
        hay = ' '.join([image_path, rel_path, output_image])
        if 'CBLPRD-330k' in hay or 'cblprd_obb_autolabel' in output_image.lower():
            source_name = 'CBLPRD'
    out_img = Path(r['output_image']).absolute()
    if source_name == 'CBLPRD' and not out_img.exists():
        base_dir = root / 'cblprd_obb_autolabel_v1' / 'success' / r['family'] / r['split']
        plate_type = (r.get('plate_type') or '').strip()
        candidates = [base_dir / out_img.name]
        if plate_type:
            candidates.insert(0, base_dir / plate_type / out_img.name)
        for candidate in candidates:
            if candidate.exists():
                out_img = candidate
                break
    row = {
        'img_path': str(out_img),
        'img_rel_path': str(out_img.relative_to(root)).replace('\\', '/'),
        'dataset_name': dataset_map.get(source_name, f'{source_name}_pseudo_geom'),
        'split': r['split'],
        'text': r['text'].strip().upper(),
        'plate_len': len(r['text'].strip().upper()),
        'family': r['family'],
        'sub_type': r['sub_type'],
        'source': 'pseudo_geom',
        'is_real': 1 if source_name in {'CBLPRD', 'CBLPRD-330k_v1'} else 0,
        'need_tilt_aug': 1 if r['family'] in {'normal7', 'green8'} else 0,
        'preprocess_group': 'ccpd_board',
        'has_bbox': 1,
        'has_quad': 1,
        'can_parse_ccpd_geom': 1,
        'can_perspective': 1,
        'bbox_source': 'detector_obb',
        'quad_source': 'detector_obb',
    }
    return apply_board_params(row)


def row_from_cblprd_failed(r: dict, root: Path) -> dict:
    img_path = Path(r['image_path']).resolve()
    return {
        'img_path': str(img_path),
        'img_rel_path': str(img_path.relative_to(root)).replace('\\', '/'),
        'dataset_name': 'cblprd_plain',
        'split': r['split'],
        'text': r['text'].strip().upper(),
        'plate_len': len(r['text'].strip().upper()),
        'family': r['family'],
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
        'ocr_channel_order': BOARD_PARAM_EXPECTED['ocr_channel_order'],
        'ocr_crop_mode': 'plain_plate',
        'ocr_resize_mode': BOARD_PARAM_EXPECTED['ocr_resize_mode'],
        'ocr_resize_kernel': BOARD_PARAM_EXPECTED['ocr_resize_kernel'],
        'ocr_preproc': BOARD_PARAM_EXPECTED['ocr_preproc'],
        'ocr_min_occ_ratio': 1.0,
        'ocr_quad_pad_ratio': 0.0,
    }


def rows_from_crpd_mapping(mapping_csv: Path, root: Path):
    rows = []
    skipped = []
    with mapping_csv.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            text = (r['plate_text'] or '').strip().upper()
            if not text or not text_supported(text):
                skipped.append({'dataset_name': 'CRPD_CCPD_STRICT_YOLO_v2', 'split': 'unknown', 'text': text, 'reason': 'unsupported_or_empty'})
                continue
            if len(text) == 8:
                family, sub_type = 'green8', 'green'
            elif len(text) == 7:
                family, sub_type = 'normal7', 'blue'
            else:
                family, sub_type = 'special', 'special'
            new_image = Path(r['new_image'].replace('\\', '/'))
            rel_tail = Path(*new_image.parts[-3:])
            plate_crop = root / 'CRPD_CCPD_STRICT_YOLO_v2' / 'plate_crops' / rel_tail
            if not plate_crop.exists():
                skipped.append({'dataset_name': 'CRPD_CCPD_STRICT_YOLO_v2', 'split': 'unknown', 'text': text, 'reason': 'missing_plate_crop'})
                continue
            split = 'train'
            for token in ('train', 'val', 'test'):
                if f'/{token}/' in r['new_image'].replace('\\', '/'):
                    split = token
                    break
            row = {
                'img_path': str(plate_crop.resolve()),
                'img_rel_path': str(plate_crop.resolve().relative_to(root)).replace('\\', '/'),
                'dataset_name': 'crpd_ccpd_strict_yolo_v2',
                'split': split,
                'text': text,
                'plate_len': len(text),
                'family': family,
                'sub_type': sub_type,
                'source': 'real',
                'is_real': 1,
                'need_tilt_aug': 1 if family in {'normal7', 'green8'} else 0,
                'preprocess_group': 'ccpd_board',
                'has_bbox': 1,
                'has_quad': 1,
                'can_parse_ccpd_geom': 1,
                'can_perspective': 1,
                'bbox_source': 'mapping_csv',
                'quad_source': 'mapping_csv',
            }
            rows.append(apply_board_params(row))
    return rows, skipped


def summarize(rows):
    return {
        'sample_count': len(rows),
        'datasets': dict(sorted(Counter(r['dataset_name'] for r in rows).items())),
        'families': dict(sorted(Counter(r['family'] for r in rows).items())),
        'sub_types': dict(sorted(Counter(r['sub_type'] for r in rows).items())),
        'splits': dict(sorted(Counter(r['split'] for r in rows).items())),
        'sources': dict(sorted(Counter(r['source'] for r in rows).items())),
        'preprocess_groups': dict(sorted(Counter(r['preprocess_group'] for r in rows).items())),
    }


def main():
    ap = argparse.ArgumentParser(description='Rebuild board-aligned manifest v4 excluding generated targeted/git datasets and keeping real+pseudo data with strict params.')
    ap.add_argument('--root', default='/home/wzzz/LPRNet')
    ap.add_argument('--ccpd2019-train', default='/home/wzzz/LPRNet/prepared_labels/ccpd2019/train_labels.txt')
    ap.add_argument('--ccpd2019-val', default='/home/wzzz/LPRNet/prepared_labels/ccpd2019/val_labels.txt')
    ap.add_argument('--ccpd2019-test', default='/home/wzzz/LPRNet/prepared_labels/ccpd2019/test_labels.txt')
    ap.add_argument('--ccpd2019-hard-train', default='/home/wzzz/LPRNet/prepared_labels/ccpd2019_hard_tilt/train_labels.txt')
    ap.add_argument('--ccpd2019-hard-val', default='/home/wzzz/LPRNet/prepared_labels/ccpd2019_hard_tilt/val_labels.txt')
    ap.add_argument('--ccpd2019-hard-test', default='/home/wzzz/LPRNet/prepared_labels/ccpd2019_hard_tilt/test_labels.txt')
    ap.add_argument('--ccpd2020-green-train', default='/home/wzzz/LPRNet/prepared_labels/ccpd2020_green/train_labels.txt')
    ap.add_argument('--ccpd2020-green-val', default='/home/wzzz/LPRNet/prepared_labels/ccpd2020_green/val_labels.txt')
    ap.add_argument('--ccpd2020-green-test', default='/home/wzzz/LPRNet/prepared_labels/ccpd2020_green/test_labels.txt')
    ap.add_argument('--nonccpd-success-csv', default='/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/success_records.csv')
    ap.add_argument('--cblprd-success-csv', default='/home/wzzz/LPRNet/cblprd_obb_autolabel_v1/success_records.csv')
    ap.add_argument('--cblprd-failed-csv', default='/home/wzzz/LPRNet/cblprd_obb_autolabel_v1/failed_records.csv')
    ap.add_argument('--crpd-mapping-csv', default='/home/wzzz/LPRNet/CRPD_CCPD_STRICT_YOLO_v2/mapping.csv')
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = []
    skipped = []

    ccpd2019_rows, ccpd2019_skipped = build_ccpd_rows(root, 'ccpd2019', root / 'CCPD2019', {
        'train': Path(args.ccpd2019_train), 'val': Path(args.ccpd2019_val), 'test': Path(args.ccpd2019_test)
    }, 'normal7', 'blue')
    rows += ccpd2019_rows; skipped += ccpd2019_skipped

    hard_rows, hard_skipped = build_ccpd_rows(root, 'ccpd2019_hard_tilt', root / 'CCPD2019', {
        'train': Path(args.ccpd2019_hard_train), 'val': Path(args.ccpd2019_hard_val), 'test': Path(args.ccpd2019_hard_test)
    }, 'normal7', 'blue')
    rows += hard_rows; skipped += hard_skipped

    green_rows, green_skipped = build_ccpd_rows(root, 'ccpd2020_green', root / 'CCPD2020' / 'ccpd_green', {
        'train': Path(args.ccpd2020_green_train), 'val': Path(args.ccpd2020_green_val), 'test': Path(args.ccpd2020_green_test)
    }, 'green8', 'green')
    rows += green_rows; skipped += green_skipped

    for r in read_csv_rows(Path(args.nonccpd_success_csv)):
        if r.get('dataset_name') in {'targeted_green_missing_18', 'git_plate'}:
            skipped.append({'dataset_name': r.get('dataset_name'), 'split': r.get('split'), 'text': r.get('text'), 'reason': 'excluded_generated_dataset'})
            continue
        if not text_supported((r.get('text') or '').strip().upper()):
            skipped.append({'dataset_name': r.get('dataset_name'), 'split': r.get('split'), 'text': r.get('text'), 'reason': 'unsupported_chars'})
            continue
        rows.append(row_from_nonccpd_success(r, root))

    for r in read_csv_rows(Path(args.cblprd_success_csv)):
        if not text_supported((r.get('text') or '').strip().upper()):
            skipped.append({'dataset_name': 'CBLPRD', 'split': r.get('split'), 'text': r.get('text'), 'reason': 'unsupported_chars'})
            continue
        rows.append(row_from_nonccpd_success(r, root))

    for r in read_csv_rows(Path(args.cblprd_failed_csv)):
        text = (r.get('text') or '').strip().upper()
        if not text_supported(text):
            skipped.append({'dataset_name': 'CBLPRD', 'split': r.get('split'), 'text': text, 'reason': 'unsupported_chars'})
            continue
        rows.append(row_from_cblprd_failed(r, root))

    crpd_rows, crpd_skipped = rows_from_crpd_mapping(Path(args.crpd_mapping_csv), root)
    rows += crpd_rows; skipped += crpd_skipped

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    summary = {
        'strategy': 'board_aligned_v4_real_only_generated_excluded',
        'root': str(root),
        'excluded_generated_datasets': ['targeted_green_missing_18', 'git_plate'],
        'skipped_count': len(skipped),
        'skipped_by_reason': dict(sorted(Counter(r['reason'] for r in skipped).items())),
        'skipped_by_dataset': dict(sorted(Counter(r['dataset_name'] for r in skipped).items())),
        'manifest_summary': summarize(rows),
    }
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
