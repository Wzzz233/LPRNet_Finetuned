#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


MANIFEST_FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type', 'source',
    'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'bbox_source', 'quad_source', 'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
]

BUCKET_KEYS = ['geometry_clean', 'board_mid_occ', 'board_low_occ', 'board_extreme_tail']


def parse_bucket_plan(text):
    out = []
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = int(value.strip())
        if key not in BUCKET_KEYS:
            raise ValueError(f'unknown bucket: {key}')
        if value < 0:
            raise ValueError(f'negative bucket count: {part}')
        out.append((key, value))
    if not out:
        raise ValueError('empty --bucket-plan')
    return out


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


def manifest_row(row):
    return {field: row.get(field, '') for field in MANIFEST_FIELDS}


def detect_bucket(row):
    img_path = str(row.get('img_path', ''))
    img_rel_path = str(row.get('img_rel_path', ''))
    for key in BUCKET_KEYS:
        token = f'/{key}/'
        if token in img_path or token in img_rel_path:
            return key
    return ''


def build_bucket_pools(source_manifests):
    pools = defaultdict(list)
    source_counts = Counter()
    for manifest_path in source_manifests:
        rows = read_rows(Path(manifest_path))
        for row in rows:
            if row.get('split') != 'train' or row.get('family') != 'green8':
                continue
            bucket = detect_bucket(row)
            if not bucket:
                continue
            clean = manifest_row(row)
            clean['_origin_source'] = row.get('source', '')
            clean['_origin_dataset'] = row.get('dataset_name', '')
            clean['_origin_manifest'] = str(manifest_path)
            pools[bucket].append(clean)
            source_counts[str(manifest_path)] += 1
    return pools, source_counts


def clone_bucket_rows(pool, count, dataset_name, source_name, seed, bucket):
    if not pool:
        raise RuntimeError(f'empty source pool for bucket={bucket}')
    rng = random.Random(seed)
    order = list(pool)
    rng.shuffle(order)
    out = []
    origin_source_counts = Counter()
    origin_dataset_counts = Counter()
    cycle = 0
    while len(out) < count:
        if cycle > 0:
            rng.shuffle(order)
        remaining = count - len(out)
        chunk = order[:min(len(order), remaining)]
        if not chunk:
            raise RuntimeError(f'failed to sample bucket={bucket}')
        for row in chunk:
            item = manifest_row(row)
            item['dataset_name'] = dataset_name
            item['source'] = source_name
            item['img_rel_path'] = f"{item.get('img_rel_path') or item.get('img_path') or bucket}#boost{bucket}-{len(out):04d}"
            item['split'] = 'train'
            out.append(item)
            origin_source_counts[row.get('_origin_source', '')] += 1
            origin_dataset_counts[row.get('_origin_dataset', '')] += 1
        cycle += 1
    return out, origin_source_counts, origin_dataset_counts


def load_replay_entries(csv_paths):
    entries = []
    for path in csv_paths:
        with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                img_path = row.get('local_ocrin_path') or row.get('ocr_input_path') or row.get('img_path')
                gt_text = (row.get('gt_text') or '').strip()
                if not img_path or len(gt_text) != 8:
                    continue
                img_path = str(Path(img_path))
                if not Path(img_path).exists():
                    continue
                entries.append({
                    'csv_path': str(path),
                    'row_idx': row_idx,
                    'img_path': img_path,
                    'gt_text': gt_text,
                    'cluster': (row.get('cluster') or '').strip(),
                    'failure_type': (row.get('failure_type') or '').strip(),
                })
    return entries


def build_replay_rows(entries, count, dataset_name, source_name, seed):
    if count <= 0:
        return [], Counter(), Counter()
    if not entries:
        raise RuntimeError('replay_count > 0 but no usable replay entries were found')
    rng = random.Random(seed)
    order = list(entries)
    rng.shuffle(order)
    out = []
    csv_counts = Counter()
    text_counts = Counter()
    cycle = 0
    while len(out) < count:
        if cycle > 0:
            rng.shuffle(order)
        remaining = count - len(out)
        chunk = order[:min(len(order), remaining)]
        for entry in chunk:
            rep_idx = len(out)
            out.append({
                'img_path': entry['img_path'],
                'img_rel_path': f"{entry['img_path']}#replay{rep_idx:04d}",
                'dataset_name': dataset_name,
                'split': 'train',
                'text': entry['gt_text'],
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
            csv_counts[entry['csv_path']] += 1
            text_counts[entry['gt_text']] += 1
        cycle += 1
    return out, csv_counts, text_counts


def summarize_rows(rows):
    return {
        'count': len(rows),
        'dataset_counts': dict(sorted(Counter(r.get('dataset_name', '') for r in rows).items())),
        'source_counts': dict(sorted(Counter(r.get('source', '') for r in rows).items())),
        'real_counts': dict(sorted(Counter(r.get('is_real', '') for r in rows).items())),
    }


def main():
    ap = argparse.ArgumentParser(description='Build a local-only green target manifest by reweighting existing train manifests and optional replay dumps.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--source-manifest', action='append', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--source-name', required=True)
    ap.add_argument('--bucket-plan', required=True)
    ap.add_argument('--replay-csv', action='append', default=[])
    ap.add_argument('--replay-count', type=int, default=0)
    ap.add_argument('--replay-source', default='')
    ap.add_argument('--replay-dataset-name', default='')
    ap.add_argument('--seed', type=int, default=20260419)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    details_dir = out_dir / 'details'
    manifests_dir = out_dir / 'manifests'
    for d in [out_dir, details_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base_rows = [manifest_row(row) for row in read_rows(Path(args.base_manifest))]
    bucket_plan = parse_bucket_plan(args.bucket_plan)
    bucket_pools, pool_manifest_counts = build_bucket_pools(args.source_manifest)

    appended_rows = []
    bucket_origin_sources = {}
    bucket_origin_datasets = {}
    for idx, (bucket, count) in enumerate(bucket_plan):
        rows, source_counts, dataset_counts = clone_bucket_rows(
            pool=bucket_pools[bucket],
            count=count,
            dataset_name=args.dataset_name,
            source_name=args.source_name,
            seed=args.seed + idx * 1009,
            bucket=bucket,
        )
        appended_rows.extend(rows)
        bucket_origin_sources[bucket] = dict(sorted(source_counts.items()))
        bucket_origin_datasets[bucket] = dict(sorted(dataset_counts.items()))

    replay_rows = []
    replay_csv_counts = {}
    replay_text_counts = {}
    if args.replay_count > 0:
        replay_entries = load_replay_entries(args.replay_csv)
        replay_rows, replay_csv_counter, replay_text_counter = build_replay_rows(
            entries=replay_entries,
            count=args.replay_count,
            dataset_name=args.replay_dataset_name or args.dataset_name,
            source_name=args.replay_source or f'{args.source_name}_replay',
            seed=args.seed + 9001,
        )
        appended_rows.extend(replay_rows)
        replay_csv_counts = dict(sorted(replay_csv_counter.items()))
        replay_text_counts = dict(sorted(replay_text_counter.items()))

    merged_rows = base_rows + appended_rows
    local_manifest = manifests_dir / f'train_manifest_{args.dataset_name}.csv'
    write_rows(local_manifest, appended_rows)
    write_rows(Path(args.out_manifest), merged_rows)

    summary = {
        'base_manifest': args.base_manifest,
        'source_manifests': list(args.source_manifest),
        'dataset_name': args.dataset_name,
        'source_name': args.source_name,
        'bucket_plan': {bucket: count for bucket, count in bucket_plan},
        'replay_count': int(args.replay_count),
        'pool_manifest_counts': dict(sorted(pool_manifest_counts.items())),
        'bucket_origin_sources': bucket_origin_sources,
        'bucket_origin_datasets': bucket_origin_datasets,
        'replay_csv_counts': replay_csv_counts,
        'replay_text_counts': replay_text_counts,
        'base_rows': summarize_rows(base_rows),
        'appended_rows': summarize_rows(appended_rows),
        'merged_rows': summarize_rows(merged_rows),
        'paths': {
            'out_dir': str(out_dir),
            'local_manifest': str(local_manifest),
            'merged_manifest': str(Path(args.out_manifest)),
            'summary_json': str(details_dir / 'summary.json'),
        },
    }
    (details_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
