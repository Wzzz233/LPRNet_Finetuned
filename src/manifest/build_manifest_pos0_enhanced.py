#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_manifest_pos0_enhanced.py

将现有最佳 manifest（H32B 训练用）与 CBLPRD CV-geom manifest 合并，
生成 pos0 head 增强版 manifest。

合并策略：
- 保留原始 manifest 全部行（原有 train/val split 不变）
- 追加 CBLPRD 数据（仅 train split）
- CBLPRD 行带 pos0_data_source=cblprd 标记，便于后续按需过滤
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROVINCE_CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
]


def read_manifest(path: Path) -> tuple:
    """返回 (fieldnames, rows)"""
    rows = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    return fieldnames, rows


def province_distribution(rows: list, split_filter=None) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        if split_filter and row.get('split') != split_filter:
            continue
        text = row.get('text', '')
        if text:
            counter[text[0]] += 1
    return counter


def print_province_table(title: str, dist: Counter):
    print(f'\n  [{title}]')
    total = sum(dist.values())
    for prov in PROVINCE_CHARS:
        cnt = dist.get(prov, 0)
        bar = '█' * (cnt // 200)
        pct = 100 * cnt / max(total, 1)
        print(f'    {prov}: {cnt:6d} ({pct:4.1f}%) {bar}')
    others = {k: v for k, v in dist.items() if k not in PROVINCE_CHARS}
    if others:
        print(f'    其他: {sum(others.values())} {others}')


def main():
    parser = argparse.ArgumentParser(description='Build pos0-enhanced manifest by merging base + CBLPRD')
    parser.add_argument(
        '--base_train_manifest',
        default='manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv',
        help='H32B 训练 manifest（作为基础）',
    )
    parser.add_argument(
        '--base_eval_manifest',
        default='manifests/unified_manifest_green_balance_aggr_v1.csv',
        help='H32B 评估 manifest（eval 部分原样保留）',
    )
    parser.add_argument(
        '--cblprd_manifest',
        default='manifests/cblprd_cv_geom_manifest.csv',
        help='CBLPRD CV-geom manifest（待追加）',
    )
    parser.add_argument(
        '--out_train_manifest',
        default='manifests/unified_manifest_pos0_enhanced_v1_train.csv',
        help='输出训练 manifest',
    )
    parser.add_argument(
        '--out_eval_manifest',
        default='manifests/unified_manifest_pos0_enhanced_v1_eval.csv',
        help='输出评估 manifest（eval 不加 CBLPRD）',
    )
    parser.add_argument(
        '--cblprd_family_filter',
        default='',
        help='只保留 CBLPRD 中指定 family（逗号分隔，空=全部）',
    )
    parser.add_argument(
        '--out_summary',
        default='manifests/unified_manifest_pos0_enhanced_v1_summary.json',
        help='输出统计 JSON',
    )
    args = parser.parse_args()

    base_train_path = Path(args.base_train_manifest)
    base_eval_path = Path(args.base_eval_manifest)
    cblprd_path = Path(args.cblprd_manifest)
    out_train = Path(args.out_train_manifest)
    out_eval = Path(args.out_eval_manifest)
    out_summary = Path(args.out_summary)

    # 读取基础 manifest
    print(f'[Read] base train: {base_train_path}')
    base_fields, base_train_rows = read_manifest(base_train_path)
    print(f'  -> {len(base_train_rows)} rows, fields: {base_fields[:6]}...')

    print(f'[Read] base eval: {base_eval_path}')
    eval_fields, eval_rows = read_manifest(base_eval_path)
    print(f'  -> {len(eval_rows)} rows')

    # 读取 CBLPRD manifest
    print(f'[Read] cblprd: {cblprd_path}')
    cblprd_fields, cblprd_rows = read_manifest(cblprd_path)
    print(f'  -> {len(cblprd_rows)} rows')

    # 过滤 CBLPRD（只保留 train split）
    cblprd_train = [r for r in cblprd_rows if r.get('split') == 'train']
    print(f'  -> {len(cblprd_train)} train rows after split filter')

    # 按 family 过滤（可选）
    if args.cblprd_family_filter:
        allowed = set(args.cblprd_family_filter.split(','))
        cblprd_train = [r for r in cblprd_train if r.get('family') in allowed]
        print(f'  -> {len(cblprd_train)} rows after family filter: {allowed}')

    # 统一 fieldnames：取并集，以 base_fields 为主顺序，CBLPRD 新字段追加到末尾
    cblprd_extra = [f for f in cblprd_fields if f not in base_fields]
    merged_fields = base_fields + cblprd_extra
    # 确保 pos0_data_source 字段存在
    if 'pos0_data_source' not in merged_fields:
        merged_fields.append('pos0_data_source')

    # 省份分布统计（合并前）
    base_dist = province_distribution(base_train_rows)
    cblprd_dist = province_distribution(cblprd_train)

    print_province_table('基础 manifest 省份分布（train）', base_dist)
    print_province_table('CBLPRD manifest 省份分布（train）', cblprd_dist)

    # 合并后的训练 rows
    merged_train_rows = []

    # base train rows：填充缺失的 CBLPRD 新字段为空
    for row in base_train_rows:
        new_row = dict(row)
        for f in merged_fields:
            if f not in new_row:
                new_row[f] = ''
        merged_train_rows.append(new_row)

    # CBLPRD rows：填充缺失的 base 字段为空
    for row in cblprd_train:
        new_row = dict(row)
        for f in merged_fields:
            if f not in new_row:
                new_row[f] = ''
        merged_train_rows.append(new_row)

    merged_dist = province_distribution(merged_train_rows)
    print_province_table('合并后省份分布（train）', merged_dist)

    # 写训练 manifest
    out_train.parent.mkdir(parents=True, exist_ok=True)
    with out_train.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged_train_rows)
    print(f'\n[Output] train manifest: {out_train} ({len(merged_train_rows)} rows)')

    # eval manifest 原样输出（不加 CBLPRD）
    eval_merged_fields = eval_fields + [f for f in merged_fields if f not in eval_fields]
    eval_out_rows = []
    for row in eval_rows:
        new_row = dict(row)
        for f in eval_merged_fields:
            if f not in new_row:
                new_row[f] = ''
        eval_out_rows.append(new_row)

    with out_eval.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=eval_merged_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(eval_out_rows)
    print(f'[Output] eval manifest: {out_eval} ({len(eval_out_rows)} rows)')

    # Summary
    summary = {
        'base_train_rows': len(base_train_rows),
        'cblprd_train_rows_added': len(cblprd_train),
        'merged_train_rows': len(merged_train_rows),
        'eval_rows': len(eval_out_rows),
        'province_distribution_base': dict(base_dist),
        'province_distribution_cblprd': dict(cblprd_dist),
        'province_distribution_merged': dict(merged_dist),
    }
    with out_summary.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'[Output] summary: {out_summary}')

    # 关键省份增益提示
    print('\n--- 关键省份增益 ---')
    key_provinces = ['苏', '沪', '湘', '粤', '闽', '浙', '豫', '川']
    for prov in key_provinces:
        before = base_dist.get(prov, 0)
        after = merged_dist.get(prov, 0)
        gain = after - before
        print(f'  {prov}: {before} -> {after} (+{gain})')


if __name__ == '__main__':
    main()
