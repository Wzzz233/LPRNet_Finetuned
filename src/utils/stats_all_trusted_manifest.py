#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROVINCES = {
    '京','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新'
}


def province_of(text: str) -> str:
    if not text:
        return 'UNKNOWN'
    ch = text[0]
    return ch if ch in PROVINCES else 'UNKNOWN'


def trusted_group(row: dict) -> str:
    family = row.get('family', '')
    sub_type = row.get('sub_type', '')
    if family == 'normal7' and sub_type == 'blue':
        return 'blue'
    if family == 'green8' and sub_type in {'green', 'green_small', 'green_large'}:
        return sub_type
    if sub_type == 'yellow_single':
        return 'yellow_single'
    if sub_type == 'yellow_double':
        return 'yellow_double'
    if family == 'special':
        return f'special::{sub_type}' if sub_type else 'special'
    return f'{family}::{sub_type}'


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description='Stats for all currently trusted data, including CCPD and CRPD raw replacement.')
    ap.add_argument('--manifest', default='/home/wzzz/LPRNet/manifests/crpd_all_raw_board_v1_supported.csv')
    ap.add_argument('--out-json', default='/home/wzzz/LPRNet/reports/all_trusted_manifest_stats.json')
    args = ap.parse_args()

    rows = read_rows(Path(args.manifest))

    by_dataset = Counter()
    by_family = Counter()
    by_sub_type = Counter()
    by_group = Counter()
    by_split = Counter()
    by_source = Counter()
    by_preprocess = Counter()
    province_all = Counter()
    province_by_group = defaultdict(Counter)
    province_by_dataset = defaultdict(Counter)

    for r in rows:
        dataset = r.get('dataset_name', '')
        text = (r.get('text') or '').strip().upper()
        group = trusted_group(r)
        prov = province_of(text)
        by_dataset[dataset] += 1
        by_family[r.get('family', '')] += 1
        by_sub_type[r.get('sub_type', '')] += 1
        by_group[group] += 1
        by_split[r.get('split', '')] += 1
        by_source[r.get('source', '')] += 1
        by_preprocess[r.get('preprocess_group', '')] += 1
        province_all[prov] += 1
        province_by_group[group][prov] += 1
        province_by_dataset[dataset][prov] += 1

    summary = {
        'manifest': args.manifest,
        'trusted_total': len(rows),
        'by_dataset': dict(sorted(by_dataset.items())),
        'by_family': dict(sorted(by_family.items())),
        'by_sub_type': dict(sorted(by_sub_type.items())),
        'by_group': dict(sorted(by_group.items())),
        'by_split': dict(sorted(by_split.items())),
        'by_source': dict(sorted(by_source.items())),
        'by_preprocess_group': dict(sorted(by_preprocess.items())),
        'province_all': dict(sorted(province_all.items())),
        'province_by_group': {k: dict(sorted(v.items())) for k, v in sorted(province_by_group.items())},
        'province_by_dataset': {k: dict(sorted(v.items())) for k, v in sorted(province_by_dataset.items())},
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
