#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from lpr_pipeline_policy import apply_board_params

BASE_FIELDS = [
    "img_path",
    "img_rel_path",
    "dataset_name",
    "split",
    "text",
    "plate_len",
    "family",
    "sub_type",
    "source",
    "is_real",
    "need_tilt_aug",
    "preprocess_group",
    "has_bbox",
    "has_quad",
    "can_parse_ccpd_geom",
    "can_perspective",
    "bbox_source",
    "quad_source",
    "ocr_channel_order",
    "ocr_crop_mode",
    "ocr_resize_mode",
    "ocr_resize_kernel",
    "ocr_preproc",
    "ocr_min_occ_ratio",
    "ocr_quad_pad_ratio",
]


def read_csv_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def success_row_to_manifest(row: dict, root: Path) -> dict:
    img_path = Path(row['output_image']).resolve()
    rel_path = str(img_path.relative_to(root)).replace('\\', '/')
    text = row['text']
    family = row['family']
    sub_type = row['sub_type']
    split = row['split']
    dataset_name = f"{row['dataset_name']}_pseudo_geom"
    need_tilt_aug = 1 if family in {'green8', 'normal7'} else 0
    out = {
        'img_path': str(img_path),
        'img_rel_path': rel_path,
        'dataset_name': dataset_name,
        'split': split,
        'text': text,
        'plate_len': len(text),
        'family': family,
        'sub_type': sub_type,
        'source': 'pseudo_geom',
        'is_real': 0,
        'need_tilt_aug': need_tilt_aug,
        'preprocess_group': 'ccpd_board',
        'has_bbox': 1,
        'has_quad': 1,
        'can_parse_ccpd_geom': 1,
        'can_perspective': 1,
        'bbox_source': 'detector_obb',
        'quad_source': 'detector_obb',
    }
    return apply_board_params(out)


def summarize(rows):
    ds = Counter()
    fam = Counter()
    src = Counter()
    split = Counter()
    prep = Counter()
    for r in rows:
        ds[r['dataset_name']] += 1
        fam[r['family']] += 1
        src[r['source']] += 1
        split[r['split']] += 1
        prep[r['preprocess_group']] += 1
    return {
        'sample_count': len(rows),
        'datasets': dict(sorted(ds.items())),
        'families': dict(sorted(fam.items())),
        'sources': dict(sorted(src.items())),
        'splits': dict(sorted(split.items())),
        'preprocess_groups': dict(sorted(prep.items())),
    }


def main():
    ap = argparse.ArgumentParser(description='Append non-CCPD pseudo-geometry success samples into unified manifest.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--pseudo-success-csv', required=True)
    ap.add_argument('--root', default='/home/wzzz/LPRNet')
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    base_rows = read_csv_rows(Path(args.base_manifest))
    success_rows = read_csv_rows(Path(args.pseudo_success_csv))
    pseudo_rows = [success_row_to_manifest(r, root) for r in success_rows]
    merged = base_rows + pseudo_rows

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=BASE_FIELDS)
        w.writeheader()
        w.writerows(merged)

    summary = {
        'base_count': len(base_rows),
        'pseudo_geom_added': len(pseudo_rows),
        'merged_count': len(merged),
        'pseudo_geom_datasets': dict(sorted(Counter(r['dataset_name'] for r in pseudo_rows).items())),
        'pseudo_geom_families': dict(sorted(Counter(r['family'] for r in pseudo_rows).items())),
        'merged_summary': summarize(merged),
    }
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
