#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from generate_green_board_native_preview_v1 import (
    gray_stats,
    load_qspec,
    within_spec,
    spec_score,
    sample_params,
    transform_dumplike_to_board_native,
    ensure_94x24,
    make_card,
    write_contact_sheet,
)


def write_ppm(path: Path, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode('ascii'))
        f.write(rgb.tobytes())


def read_csv_rows(path: Path, delimiter=','):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def join_manifest_and_details(manifest_rows, detail_rows):
    detail_map = {row['out_rel_path']: row for row in detail_rows}
    joined = []
    for row in manifest_rows:
        d = detail_map.get(row['img_rel_path'])
        if d is None:
            continue
        joined.append((row, d))
    return joined


def source_filter(detail_row, args):
    if detail_row['bucket'] != args.required_bucket:
        return False
    mean = float(detail_row['mean'])
    left_edge = float(detail_row['left_edge'])
    mid_edge = float(detail_row['mid_edge'])
    return (
        args.mean_min <= mean <= args.mean_max
        and args.left_edge_min <= left_edge <= args.left_edge_max
        and args.mid_edge_min <= mid_edge <= args.mid_edge_max
    )


def append_manifest(base_manifest: Path, append_rows, out_manifest: Path):
    base_rows = read_csv_rows(base_manifest)
    fieldnames = list(base_rows[0].keys())
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(base_rows)
        w.writerows(append_rows)
    return len(base_rows), len(base_rows) + len(append_rows)


def main():
    ap = argparse.ArgumentParser(description='Generate accepted-only green board-native append data from transformable dumplike geometry_clean sources.')
    ap.add_argument('--input-manifest', default='/home/wzzz/LPRNet/tmp/green_dumplike_boarddump_bright_v1_20260412_a3100/manifests/train_manifest_dumplike_boarddump_v1.csv')
    ap.add_argument('--details-tsv', default='/home/wzzz/LPRNet/tmp/green_dumplike_boarddump_bright_v1_20260412_a3100/details/accepted.tsv')
    ap.add_argument('--spec-json', default='/home/wzzz/LPRNet/tmp/green_board_domain_spec_v1.json')
    ap.add_argument('--base-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/tmp/green_board_native_append_v1_20260413')
    ap.add_argument('--out-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e6c_boardnative_append_v1_20260413.csv')
    ap.add_argument('--required-bucket', default='geometry_clean')
    ap.add_argument('--mean-min', type=float, default=155.0)
    ap.add_argument('--mean-max', type=float, default=185.0)
    ap.add_argument('--left-edge-min', type=float, default=80.0)
    ap.add_argument('--left-edge-max', type=float, default=145.0)
    ap.add_argument('--mid-edge-min', type=float, default=85.0)
    ap.add_argument('--mid-edge-max', type=float, default=150.0)
    ap.add_argument('--candidate-limit', type=int, default=0)
    ap.add_argument('--target-count', type=int, default=800)
    ap.add_argument('--per-source-target', type=int, default=10)
    ap.add_argument('--max-tries-per-source', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20260413)
    ap.add_argument('--dataset-name', default='green_board_native_cluster1_append_v1')
    ap.add_argument('--source-name', default='board_native_cluster1_append_v1')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images' / 'train'
    cards_dir = out_dir / 'cards'
    details_dir = out_dir / 'details'
    manifests_dir = out_dir / 'manifests'
    for d in [images_dir, cards_dir, details_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv_rows(Path(args.input_manifest))
    detail_rows = read_csv_rows(Path(args.details_tsv), delimiter='\t')
    joined = join_manifest_and_details(manifest_rows, detail_rows)
    candidates = [(m, d) for (m, d) in joined if source_filter(d, args)]
    candidates.sort(key=lambda x: float(x[1].get('score', '9999')))
    if args.candidate_limit > 0:
        candidates = candidates[:args.candidate_limit]

    qspec = load_qspec(Path(args.spec_json))
    generated_manifest_rows = []
    generated_detail_rows = []
    cards = []
    province_counts = Counter()
    source_success_counts = Counter()
    source_try_counts = {}
    all_stats = []

    for source_idx, (src_manifest_row, src_detail_row) in enumerate(candidates):
        if len(generated_manifest_rows) >= args.target_count:
            break
        src_img = cv2.imread(src_manifest_row['img_path'], cv2.IMREAD_COLOR)
        if src_img is None:
            continue
        src_img = ensure_94x24(src_img)
        tries = 0
        accepted = 0
        while tries < args.max_tries_per_source and accepted < args.per_source_target and len(generated_manifest_rows) < args.target_count:
            tries += 1
            params = sample_params(rng)
            gen = transform_dumplike_to_board_native(src_img, params)
            stats = gray_stats(gen)
            if not within_spec(stats, qspec):
                continue
            score = spec_score(stats, qspec)
            prov = src_manifest_row['text'][0]
            text = src_manifest_row['text']
            out_name = f's{source_idx:03d}_v{accepted:02d}_{text}.ppm'
            rel_path = f'images/train/{args.required_bucket}/{prov}/{out_name}'
            abs_path = out_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            write_ppm(abs_path, gen)

            new_row = dict(src_manifest_row)
            new_row['img_path'] = str(abs_path)
            new_row['img_rel_path'] = rel_path
            new_row['dataset_name'] = args.dataset_name
            new_row['source'] = args.source_name
            generated_manifest_rows.append(new_row)
            generated_detail_rows.append({
                'source_idx': source_idx,
                'variant_idx': accepted,
                'text': text,
                'province': prov,
                'src_img_path': src_manifest_row['img_path'],
                'src_img_rel_path': src_manifest_row['img_rel_path'],
                'out_img_path': str(abs_path),
                'out_img_rel_path': rel_path,
                'score': f'{score:.6f}',
                'mean': f"{stats['mean']:.6f}",
                'left_minus_right': f"{stats['left_minus_right']:.6f}",
                'border_dark_ratio': f"{stats['border_dark_ratio']:.6f}",
                'left_edge': f"{stats['left_edge']:.6f}",
                'mid_edge': f"{stats['mid_edge']:.6f}",
                'occ_ratio': f"{stats['occ_ratio']:.6f}",
                'params_json': json.dumps(params, ensure_ascii=False),
            })
            province_counts[prov] += 1
            source_success_counts[src_manifest_row['img_rel_path']] += 1
            all_stats.append(stats)
            if accepted < 2:
                cards.append(make_card(src_img, gen, f'{text} v{accepted:02d}', stats, score))
            accepted += 1
        source_try_counts[src_manifest_row['img_rel_path']] = {'tries': tries, 'accepted': accepted}

    append_manifest_path = manifests_dir / 'train_manifest_green_board_native_append_v1.csv'
    if generated_manifest_rows:
        fieldnames = list(generated_manifest_rows[0].keys())
        with append_manifest_path.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(generated_manifest_rows)

    details_path = details_dir / 'accepted.tsv'
    if generated_detail_rows:
        with details_path.open('w', encoding='utf-8', newline='') as f:
            fieldnames = list(generated_detail_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            w.writeheader()
            w.writerows(generated_detail_rows)

    contact_sheet_path = out_dir / 'contact_sheet.png'
    write_contact_sheet(cards, contact_sheet_path, cols=3)

    base_total, new_total = append_manifest(Path(args.base_manifest), generated_manifest_rows, Path(args.out_manifest))

    summary = {
        'experiment_name': 'E6C-A_board_native_append_v1',
        'base_manifest': args.base_manifest,
        'out_manifest': args.out_manifest,
        'append_manifest': str(append_manifest_path),
        'details_path': str(details_path),
        'contact_sheet_path': str(contact_sheet_path),
        'candidate_source_count': len(candidates),
        'successful_source_count': int(sum(1 for v in source_try_counts.values() if v['accepted'] > 0)),
        'append_count': len(generated_manifest_rows),
        'target_count': args.target_count,
        'base_total': base_total,
        'new_total': new_total,
        'candidate_limit': args.candidate_limit,
        'per_source_target': args.per_source_target,
        'max_tries_per_source': args.max_tries_per_source,
        'filter': {
            'bucket': args.required_bucket,
            'mean': [args.mean_min, args.mean_max],
            'left_edge': [args.left_edge_min, args.left_edge_max],
            'mid_edge': [args.mid_edge_min, args.mid_edge_max],
        },
        'province_counts': dict(province_counts),
        'source_success_counts_top20': dict(sorted(source_success_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]),
        'aggregate': {},
    }
    if all_stats:
        for key in all_stats[0].keys():
            vals = np.asarray([x[key] for x in all_stats], dtype=np.float32)
            summary['aggregate'][key] = {
                'mean': float(vals.mean()),
                'min': float(vals.min()),
                'max': float(vals.max()),
            }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
