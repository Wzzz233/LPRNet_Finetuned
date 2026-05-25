#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import generate_green_e16a_nonanhui_ad_balance as e16a

PNG_COMPRESSION = 3
MANIFEST_NAMES = {
    'train': 'train_synthetic_labels.txt',
    'val': 'val_synthetic_labels.txt',
    'test': 'test_synthetic_labels.txt',
}


def parse_args():
    ap = argparse.ArgumentParser(
        description='Rebuild green_exact_quad synthetic sources with cleaner exact-quad rendering.'
    )
    ap.add_argument(
        '--src-root',
        default='/home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1',
        help='existing exact source dataset root; only manifests are read',
    )
    ap.add_argument(
        '--out-root',
        default='/home/wzzz/LPRNet/tmp/green_exact_quad_synthetic_v3_frontclean_20260428',
        help='output root for rebuilt exact-quad synthetic dataset',
    )
    ap.add_argument(
        '--repo-root',
        default='/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/repo_license_plate_generator',
    )
    ap.add_argument('--lpr-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--seed', type=int, default=20260428)
    ap.add_argument(
        '--limit-per-split',
        type=int,
        default=0,
        help='0 means rebuild all rows; positive value limits each split for smoke testing',
    )
    ap.add_argument('--preview-per-split', type=int, default=24)
    return ap.parse_args()


def read_label_rows(path: Path):
    rows = []
    for idx, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        line = line.strip()
        if not line:
            continue
        rel_path, text = line.split(maxsplit=1)
        rows.append({'row_index': idx, 'rel_path': rel_path, 'text': text})
    return rows


def order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(ang)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[3, 0]:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], np.float32)
    return ordered.astype(np.float32)


def quad_bbox(quad, img_w, img_h):
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    q[:, 0] = np.clip(q[:, 0], 0, img_w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, img_h - 1)
    x1 = int(np.floor(np.min(q[:, 0])))
    y1 = int(np.floor(np.min(q[:, 1])))
    x2 = int(np.ceil(np.max(q[:, 0])))
    y2 = int(np.ceil(np.max(q[:, 1])))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1, min(x2, img_w - 1))
    y2 = max(y1, min(y2, img_h - 1))
    return x1, y1, x2, y2


def make_ccpd_like_name(quad, split, row_index, img_w, img_h):
    q = order_quad(quad)
    x1, y1, x2, y2 = quad_bbox(q, img_w, img_h)
    quad_int = np.rint(q).astype(np.int32)
    quad_part = '_'.join(f'{int(x)}&{int(y)}' for x, y in quad_int)
    return f'genx-0-{x1}&{y1}_{x2}&{y2}-{quad_part}-{split}-{row_index:05d}.png'


def preview_image(realized, quad):
    img = realized.copy()
    pts = np.rint(np.asarray(quad, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    return img


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    manifests_root = out_root / 'manifests'
    details_root = out_root / 'details'
    preview_root = out_root / 'preview'
    for path in [out_root, manifests_root, details_root, preview_root]:
        path.mkdir(parents=True, exist_ok=True)

    chars_gen, augmenter = e16a.ensure_repo_imports(args.repo_root)
    augmenter = e16a.setup_augmenter(augmenter, e16a.GEOMETRY_CFG)
    prepare_board = e16a.load_prepare_board(args.lpr_root)
    canonical_quad = e16a.detect_canonical_plate_quad(augmenter.template_image)

    accepted = []
    manifest_rel_rows = {}
    split_counts = {}
    province_counts = {}

    for split in ['train', 'val', 'test']:
        label_path = src_root / 'manifests' / MANIFEST_NAMES[split]
        rows = read_label_rows(label_path)
        if args.limit_per_split > 0:
            rows = rows[: args.limit_per_split]

        out_rows = []
        split_counter = Counter()
        for row in rows:
            text = row['text'].strip()
            rel_path = Path(row['rel_path'])
            province_dir = rel_path.parent.name
            split_dir = out_root / 'images' / split / province_dir
            split_dir.mkdir(parents=True, exist_ok=True)

            rng = random.Random(args.seed + {'train': 0, 'val': 100000, 'test': 200000}[split] + row['row_index'])
            render = e16a.render_clean_source_exact_quad(
                text,
                chars_gen,
                augmenter,
                canonical_quad,
                prepare_board,
            )
            realized = render['realized']
            quad = np.asarray(render['quad'], dtype=np.float32)
            name = make_ccpd_like_name(quad, split, row['row_index'], realized.shape[1], realized.shape[0])
            out_path = split_dir / name
            ok = cv2.imwrite(str(out_path), realized, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            if not ok:
                raise RuntimeError(f'failed to write {out_path}')

            out_rel = out_path.relative_to(out_root).as_posix()
            out_rows.append({'rel_path': out_rel, 'text': text})
            split_counter[text[0]] += 1
            accepted.append({
                'split': split,
                'province': text[0],
                'text': text,
                'rel_path': out_rel,
                'horizontal_sight_direction': render['horizontal_sight_direction'],
                'vertical_sight_direction': render['vertical_sight_direction'],
                'occ_ratio': round(float(render['occ_ratio']), 6),
                'max_char_angle_error_deg': round(float(render['max_char_angle_error_deg']), 6),
                'quad': np.asarray(quad, dtype=np.float32).tolist(),
            })

            if len([x for x in accepted if x['split'] == split]) <= args.preview_per_split:
                cv2.imwrite(str(preview_root / f'{split}_{row["row_index"]:05d}.png'), preview_image(realized, quad))

        manifest_rel_rows[split] = out_rows
        split_counts[split] = len(out_rows)
        province_counts[split] = dict(sorted(split_counter.items()))

    for split, rows in manifest_rel_rows.items():
        out_manifest = manifests_root / MANIFEST_NAMES[split]
        with out_manifest.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(f"{row['rel_path']} {row['text']}\n")

    accepted_tsv = details_root / 'accepted.tsv'
    with accepted_tsv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'split',
                'province',
                'text',
                'rel_path',
                'horizontal_sight_direction',
                'vertical_sight_direction',
                'occ_ratio',
                'max_char_angle_error_deg',
                'quad',
            ],
            delimiter='\t',
        )
        writer.writeheader()
        for row in accepted:
            out = dict(row)
            out['quad'] = json.dumps(out['quad'], ensure_ascii=False)
            writer.writerow(out)

    report = {
        'src_root': str(src_root),
        'out_root': str(out_root),
        'manifests_root': str(manifests_root),
        'accepted_tsv': str(accepted_tsv),
        'preview_root': str(preview_root),
        'split_counts': split_counts,
        'province_counts': province_counts,
        'render_cfg': dict(e16a.EXACT_RENDER_CFG),
        'geometry_cfg': dict(e16a.GEOMETRY_CFG),
        'render_mode': 'front_clean_single_plate',
        'canonical_quad': np.asarray(canonical_quad, dtype=np.float32).tolist(),
        'png_compression': PNG_COMPRESSION,
        'limit_per_split': args.limit_per_split,
    }
    (out_root / 'rebuild_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
