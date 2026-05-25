#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from pathlib import Path

from lpr_pipeline_policy import validate_manifest_row, summarize_issues, ensure_existing_file, BOARD_PARAM_EXPECTED
from load_data import UnifiedManifestDataset, parse_ccpd_quad_from_name, parse_ccpd_bbox_from_name


def parse_args():
    ap = argparse.ArgumentParser(description='Project-level strict checker for bbox/quad and board-aligned preprocessing rules.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--sample-count', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260328)
    ap.add_argument('--out-json', default='')
    return ap.parse_args()


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def sample_indices(total, k, seed):
    random.seed(seed)
    idxs = list(range(total))
    random.shuffle(idxs)
    return sorted(idxs[:min(k, total)])


def quad_signature(row):
    rel = row.get('img_rel_path') or row.get('img_path') or ''
    quad = parse_ccpd_quad_from_name(rel)
    bbox = parse_ccpd_bbox_from_name(rel)
    bbox_list = None
    if bbox is not None:
        bbox_list = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
    return {
        'quad': quad.tolist() if quad is not None else None,
        'bbox': bbox_list,
    }


def main():
    args = parse_args()
    manifest = Path(args.manifest)
    rows = read_rows(manifest)
    issues = []
    missing_files = []
    board_rows = []
    pseudo_geom_rows = []
    style_hint_rows = []

    for i, row in enumerate(rows, 2):
        issues.extend(validate_manifest_row(row, row_index=i))
        path = row.get('img_path')
        if path and not ensure_existing_file(path):
            missing_files.append({'line': i, 'img_path': path})
        if (row.get('preprocess_group') or '').strip() == 'ccpd_board':
            board_rows.append((i, row))
        if (row.get('source') or '').strip() == 'pseudo_geom':
            pseudo_geom_rows.append((i, row))
        hay = ' '.join([str(row.get('dataset_name') or ''), str(row.get('img_path') or ''), str(row.get('img_rel_path') or '')]).lower()
        if any(tok in hay for tok in ('style_transfer', 'stylized', 'translated', 'cycle', 'realmix', 'fastcut')):
            style_hint_rows.append((i, row))

    board_sample = []
    board_idxs = sample_indices(len(board_rows), args.sample_count, args.seed)
    for local_idx in board_idxs:
        line_no, row = board_rows[local_idx]
        sig = quad_signature(row)
        board_sample.append({
            'line': line_no,
            'img_path': row.get('img_path'),
            'family': row.get('family'),
            'bbox_source': row.get('bbox_source'),
            'quad_source': row.get('quad_source'),
            'geom': sig,
        })

    unique_geom = set()
    for item in board_sample:
        unique_geom.add(json.dumps(item['geom'], ensure_ascii=False, sort_keys=True))

    loader_sample_report = None
    if board_rows:
        ds = UnifiedManifestDataset(
            manifest_path=str(manifest),
            img_size=[94, 24],
            lpr_max_len=8,
            split_filter='test',
            ocr_channel_order=BOARD_PARAM_EXPECTED['ocr_channel_order'],
            ocr_crop_mode=BOARD_PARAM_EXPECTED['ocr_crop_mode'],
            ocr_resize_mode=BOARD_PARAM_EXPECTED['ocr_resize_mode'],
            ocr_resize_kernel=BOARD_PARAM_EXPECTED['ocr_resize_kernel'],
            ocr_preproc=BOARD_PARAM_EXPECTED['ocr_preproc'],
            ocr_min_occ_ratio=BOARD_PARAM_EXPECTED['ocr_min_occ_ratio'],
            ocr_quad_pad_ratio=BOARD_PARAM_EXPECTED['ocr_quad_pad_ratio'],
        )
        if len(ds) > 0:
            local_indices = sample_indices(len(ds), min(5, len(ds)), args.seed)
            samples = []
            for idx in local_indices:
                image, label, length = ds[idx]
                samples.append({
                    'index': idx,
                    'img_path': ds.img_paths[idx],
                    'shape': list(image.shape),
                    'length': int(length),
                    'text': ds.img_labels[idx],
                })
            loader_sample_report = samples

    report = {
        'manifest': str(manifest),
        'row_count': len(rows),
        'issue_summary': summarize_issues(issues),
        'issues': [issue.__dict__ for issue in issues[:200]],
        'missing_files': missing_files[:200],
        'board_row_count': len(board_rows),
        'pseudo_geom_row_count': len(pseudo_geom_rows),
        'style_hint_row_count': len(style_hint_rows),
        'board_param_expected': BOARD_PARAM_EXPECTED,
        'board_sample': board_sample,
        'board_sample_unique_geom_count': len(unique_geom),
        'loader_sample_report': loader_sample_report,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
