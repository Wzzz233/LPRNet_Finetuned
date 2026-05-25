#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


MANIFEST_FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type', 'source',
    'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'bbox_source', 'quad_source', 'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
]


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    if not rows:
        raise RuntimeError(f'no rows to write: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def dedup_rows(rows):
    seen = set()
    out = []
    for row in rows:
        key = (row.get('img_path', ''), row.get('text', ''), row.get('source', ''), row.get('split', ''))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_dump_rows(cluster2_csv: Path, repeat: int, dataset_name: str, source_name: str):
    out = []
    with cluster2_csv.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            img_path = row.get('local_ocrin_path') or row.get('ocr_input_path') or row.get('img_path')
            gt_text = (row.get('gt_text') or '').strip()
            if not img_path or len(gt_text) != 8:
                continue
            img_path = str(Path(img_path))
            if not Path(img_path).exists():
                continue
            for rep_idx in range(repeat):
                out.append({
                    'img_path': img_path,
                    'img_rel_path': f'{img_path}#rep{rep_idx:02d}',
                    'dataset_name': dataset_name,
                    'split': 'train',
                    'text': gt_text,
                    'plate_len': '8',
                    'family': 'green8',
                    'sub_type': 'green',
                    'source': source_name,
                    'is_real': '1',
                    'need_tilt_aug': '0',
                    'preprocess_group': 'dump_replay',
                    'has_bbox': '0',
                    'has_quad': '0',
                    'can_parse_ccpd_geom': '0',
                    'can_perspective': '0',
                    'bbox_source': 'none',
                    'quad_source': 'none',
                    'ocr_channel_order': 'bgr',
                    'ocr_crop_mode': 'obb_warp',
                    'ocr_resize_mode': 'letterbox',
                    'ocr_resize_kernel': 'nn',
                    'ocr_preproc': 'none',
                    'ocr_min_occ_ratio': '0.9',
                    'ocr_quad_pad_ratio': '0.0',
                })
    return out


def summarize(rows):
    return {
        'count': len(rows),
        'source_counts': dict(sorted(Counter(r.get('source', '') for r in rows).items())),
        'dataset_counts': dict(sorted(Counter(r.get('dataset_name', '') for r in rows).items())),
        'prefix_counts': dict(sorted(Counter((r.get('text') or '')[:1] for r in rows).items())),
    }


def main():
    ap = argparse.ArgumentParser(description='Build E28 cluster2 specialist training manifest from existing focused assets.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--repr-manifest', required=True)
    ap.add_argument('--cluster2-csv', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    ap.add_argument('--contrast-source', default='e20a_cluster2_beijing_prefix_contrast_1200')
    ap.add_argument('--repr-source', default='e25a_cluster2_repr_boarddump_6000')
    ap.add_argument('--dump-source', default='e28_cluster2_dump_replay')
    ap.add_argument('--dump-dataset-name', default='green_e28_cluster2_dump_replay')
    ap.add_argument('--dump-repeat', type=int, default=16)
    args = ap.parse_args()

    base_rows = read_rows(Path(args.base_manifest))
    repr_rows = read_rows(Path(args.repr_manifest))

    contrast_rows = [
        row for row in base_rows
        if row.get('split') == 'train'
        and row.get('family') == 'green8'
        and row.get('source') == args.contrast_source
    ]
    repr_train_rows = [
        row for row in repr_rows
        if row.get('split') == 'train'
        and row.get('family') == 'green8'
        and row.get('source') == args.repr_source
    ]
    dump_rows = build_dump_rows(
        cluster2_csv=Path(args.cluster2_csv),
        repeat=max(1, int(args.dump_repeat)),
        dataset_name=args.dump_dataset_name,
        source_name=args.dump_source,
    )

    merged_rows = dedup_rows(contrast_rows) + dedup_rows(repr_train_rows) + dump_rows
    write_rows(Path(args.out_manifest), merged_rows)

    summary = {
        'base_manifest': args.base_manifest,
        'repr_manifest': args.repr_manifest,
        'cluster2_csv': args.cluster2_csv,
        'dump_repeat': int(args.dump_repeat),
        'contrast_rows': summarize(contrast_rows),
        'repr_rows': summarize(repr_train_rows),
        'dump_rows': summarize(dump_rows),
        'merged_rows': summarize(merged_rows),
    }
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
