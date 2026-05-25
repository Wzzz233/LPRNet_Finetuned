#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

SUPPORTED_CHARS = set([
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'I', 'O', '-'
])


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def text_supported(text: str) -> bool:
    return all(ch in SUPPORTED_CHARS for ch in text)


def pick_limited_green_train(rows, per_dataset_limits):
    picked = []
    counters = Counter()
    for r in rows:
        if r['split'] != 'train':
            continue
        if r['family'] != 'green8':
            continue
        if not text_supported(r['text']):
            continue
        ds = r['dataset_name']
        limit = per_dataset_limits.get(ds, 0)
        if counters[ds] >= limit:
            continue
        picked.append(r)
        counters[ds] += 1
    return picked, counters


def summarize(rows):
    return {
        'count': len(rows),
        'datasets': dict(sorted(Counter(r['dataset_name'] for r in rows).items())),
        'families': dict(sorted(Counter(r['family'] for r in rows).items())),
        'sub_types': dict(sorted(Counter(r['sub_type'] for r in rows).items())),
        'splits': dict(sorted(Counter(r['split'] for r in rows).items())),
        'sources': dict(sorted(Counter(r['source'] for r in rows).items())),
        'preprocess_groups': dict(sorted(Counter(r['preprocess_group'] for r in rows).items())),
    }


def main():
    ap = argparse.ArgumentParser(description='Build conservative round1 manifest: keep supported blue base, add limited supported green train, full eval unchanged except unsupported texts filtered out.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    args = ap.parse_args()

    rows = read_rows(Path(args.base_manifest))
    train_keep = []
    eval_keep = []
    skipped_unsupported = Counter()

    green_limits = {
        'ccpd2020_green': 6000,
        'targeted_green_missing_18': 2000,
        'targeted_green_missing_18_pseudo_geom': 2000,
        'cblprd_pseudo_geom': 4000,
        'cblprd_plain': 4000,
    }

    for r in rows:
        if not text_supported(r['text']):
            skipped_unsupported[r['dataset_name']] += 1
            continue
        split = r['split']
        if split in {'val', 'test', 'eval'}:
            eval_keep.append(r)
            continue
        if split != 'train':
            continue
        if r['family'] == 'normal7':
            train_keep.append(r)
            continue
        if r['family'] == 'special':
            continue

    green_train_rows, green_counts = pick_limited_green_train(rows, green_limits)
    merged = train_keep + green_train_rows + eval_keep

    write_rows(Path(args.out_manifest), merged)
    summary = {
        'strategy': 'conservative_round1_green',
        'base_manifest': str(Path(args.base_manifest).resolve()),
        'green_train_limits': green_limits,
        'green_train_selected': dict(sorted(green_counts.items())),
        'skipped_unsupported_text_by_dataset': dict(sorted(skipped_unsupported.items())),
        'train_blue_count': sum(1 for r in train_keep if r['split'] == 'train'),
        'train_green_count': len(green_train_rows),
        'eval_rows_kept': len(eval_keep),
        'merged_summary': summarize(merged),
    }
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
