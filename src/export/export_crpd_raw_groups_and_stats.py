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

EXPORTS = {
    'blue_supported': lambda r: r['family'] == 'normal7' and r['sub_type'] == 'blue',
    'yellow_single_supported': lambda r: r['sub_type'] == 'yellow_single',
    'yellow_double_supported': lambda r: r['sub_type'] == 'yellow_double',
    'special_supported': lambda r: r['family'] == 'special' and r['sub_type'] == 'special',
}


def read_csv_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def province_of(text: str) -> str:
    if not text:
        return 'UNKNOWN'
    ch = text[0]
    return ch if ch in PROVINCES else 'UNKNOWN'


def group_name(row: dict) -> str:
    if row['family'] == 'normal7' and row['sub_type'] == 'blue':
        return 'blue'
    if row['sub_type'] == 'yellow_single':
        return 'yellow_single'
    if row['sub_type'] == 'yellow_double':
        return 'yellow_double'
    if row['family'] == 'special' and row['sub_type'] == 'special':
        return 'special'
    if row['family'] == 'green8':
        return row['sub_type']
    return f"{row['family']}::{row['sub_type']}"


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description='Export trusted CRPD raw groups and province stats.')
    ap.add_argument('--manifest', default='/home/wzzz/LPRNet/manifests/crpd_all_raw_board_v1_supported.csv')
    ap.add_argument('--all-records', default='/home/wzzz/LPRNet/reports/crpd_all_raw_board_v1_all_records.csv')
    ap.add_argument('--skipped', default='/home/wzzz/LPRNet/reports/crpd_all_raw_board_v1_skipped.csv')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/reports/crpd_raw_group_exports')
    args = ap.parse_args()

    manifest_rows = read_csv_rows(Path(args.manifest))
    all_records = read_csv_rows(Path(args.all_records))
    skipped_rows = read_csv_rows(Path(args.skipped))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_fields = list(manifest_rows[0].keys()) if manifest_rows else []
    all_fields = list(all_records[0].keys()) if all_records else []
    skipped_fields = list(skipped_rows[0].keys()) if skipped_rows else []

    # 导出可信分组
    export_counts = {}
    for name, fn in EXPORTS.items():
        rows = [r for r in manifest_rows if fn(r)]
        write_csv(out_dir / f'{name}.csv', rows, manifest_fields)
        export_counts[name] = len(rows)

    # 导出不可信 special（当前字符表不支持）
    unsupported_special = [
        r for r in skipped_rows
        if r.get('reason') == 'unsupported_chars' and r.get('family') == 'special'
    ]
    if unsupported_special:
        write_csv(out_dir / 'special_unsupported.csv', unsupported_special, skipped_fields)

    # 可信统计
    trusted_counts = Counter(group_name(r) for r in manifest_rows)
    trusted_province = defaultdict(Counter)
    for r in manifest_rows:
        trusted_province[group_name(r)][province_of(r['text'])] += 1

    # 全量原始统计（含不支持字符）
    all_counts = Counter(r['plate_group'] for r in all_records)
    all_sub_types = Counter(r['sub_type'] for r in all_records)
    all_province = defaultdict(Counter)
    for r in all_records:
        all_province[r['plate_group']][province_of(r['text'])] += 1

    # 跳过的特殊牌分类
    unsupported_special_sub = Counter(r.get('unsupported_chars', '') for r in unsupported_special)
    unsupported_special_province = Counter(province_of(r.get('text', '')) for r in unsupported_special)

    summary = {
        'trusted_total': len(manifest_rows),
        'trusted_counts_by_group': dict(sorted(trusted_counts.items())),
        'trusted_counts_by_export': dict(sorted(export_counts.items())),
        'trusted_province_by_group': {k: dict(sorted(v.items())) for k, v in sorted(trusted_province.items())},
        'all_raw_total': len(all_records),
        'all_raw_counts_by_plate_group': dict(sorted(all_counts.items())),
        'all_raw_counts_by_sub_type': dict(sorted(all_sub_types.items())),
        'all_raw_province_by_plate_group': {k: dict(sorted(v.items())) for k, v in sorted(all_province.items())},
        'unsupported_special_count': len(unsupported_special),
        'unsupported_special_chars': dict(sorted(unsupported_special_sub.items())),
        'unsupported_special_province': dict(sorted(unsupported_special_province.items())),
    }

    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
