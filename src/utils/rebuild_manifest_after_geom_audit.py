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

GEOM_UNTRUSTED = {'crpd_ccpd_strict_yolo_v2', 'git_plate_pseudo_geom'}


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def downgrade_row(r: dict):
    out = dict(r)
    out['preprocess_group'] = 'plain_plate'
    out['has_bbox'] = 0
    out['has_quad'] = 0
    out['can_parse_ccpd_geom'] = 0
    out['can_perspective'] = 0
    out['bbox_source'] = 'none'
    out['quad_source'] = 'none'
    out['ocr_crop_mode'] = 'plain_plate'
    out['ocr_resize_mode'] = 'letterbox'
    out['ocr_resize_kernel'] = 'nn'
    out['ocr_preproc'] = 'none'
    out['ocr_min_occ_ratio'] = 1.0
    out['ocr_quad_pad_ratio'] = 0.0
    return out


def summarize(rows):
    return {
        'count': len(rows),
        'datasets': dict(sorted(Counter(r['dataset_name'] for r in rows).items())),
        'families': dict(sorted(Counter(r['family'] for r in rows).items())),
        'sources': dict(sorted(Counter(r['source'] for r in rows).items())),
        'splits': dict(sorted(Counter(r['split'] for r in rows).items())),
        'preprocess_groups': dict(sorted(Counter(r['preprocess_group'] for r in rows).items())),
    }


def build_round1(rows):
    green_limits = {
        'ccpd2020_green': 6000,
        'targeted_green_missing_18': 2000,
        'targeted_green_missing_18_pseudo_geom': 2000,
        'cblprd_pseudo_geom': 4000,
        'cblprd_plain': 4000,
    }
    selected_green = Counter()
    out = []
    for r in rows:
        split = r['split']
        if split in {'val', 'test', 'eval'}:
            out.append(r)
            continue
        if split != 'train':
            continue
        if r['family'] == 'normal7':
            out.append(r)
            continue
        if r['family'] == 'special':
            continue
        if r['family'] == 'green8':
            ds = r['dataset_name']
            if selected_green[ds] < green_limits.get(ds, 0):
                out.append(r)
                selected_green[ds] += 1
    return out, dict(sorted(selected_green.items()))


def main():
    ap = argparse.ArgumentParser(description='Rebuild manifest after geometry QA audit.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    ap.add_argument('--out-round1', required=True)
    ap.add_argument('--out-round1-summary', required=True)
    args = ap.parse_args()

    rows = read_rows(Path(args.base_manifest))
    audited = []
    downgraded = Counter()
    for r in rows:
        if r['dataset_name'] in GEOM_UNTRUSTED:
            audited.append(downgrade_row(r))
            downgraded[r['dataset_name']] += 1
        else:
            audited.append(r)

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(audited)

    summary = {
        'base_manifest': str(Path(args.base_manifest).resolve()),
        'downgraded_geometry_datasets': dict(sorted(downgraded.items())),
        'audited_summary': summarize(audited),
    }
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    round1_rows, green_selected = build_round1(audited)
    with Path(args.out_round1).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(round1_rows)
    round1_summary = {
        'from_manifest': str(Path(args.out_manifest).resolve()),
        'green_train_selected': green_selected,
        'round1_summary': summarize(round1_rows),
    }
    Path(args.out_round1_summary).write_text(json.dumps(round1_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps({'audited': summary, 'round1': round1_summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
