#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ALLOWED_SPECIAL_SUBTYPES = {'yellow_single', 'yellow_double', 'black'}
EXCLUDED_SPECIAL_SUFFIXES = {'学', '挂', '警', '港', '澳'}


def read_csv(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader]
    return fieldnames, rows


def province_counter(rows):
    c = Counter()
    for row in rows:
        text = row.get('text', '')
        if text:
            c[text[0]] += 1
    return c


def summarize_rows(rows):
    fam = Counter()
    src = Counter()
    sub = Counter()
    prep = Counter()
    prov = province_counter(rows)
    for row in rows:
        fam[row.get('family', '')] += 1
        src[row.get('source', '')] += 1
        sub[row.get('sub_type', '')] += 1
        prep[row.get('preprocess_group', '')] += 1
    total = max(1, len(rows))
    return {
        'total_rows': len(rows),
        'family': dict(fam),
        'source': dict(src),
        'sub_type_top20': dict(sub.most_common(20)),
        'preprocess_group': dict(prep),
        'province_top15': dict(prov.most_common(15)),
        'province_count': len(prov),
        'province_min': min(prov.values()) if prov else 0,
        'province_max': max(prov.values()) if prov else 0,
        'anhui_ratio': prov.get('皖', 0) / total,
    }


def ensure_fields(rows, merged_fields):
    out = []
    for row in rows:
        new_row = dict(row)
        for f in merged_fields:
            if f not in new_row:
                new_row[f] = ''
        out.append(new_row)
    return out


def selected_special(row):
    if row.get('family') != 'special':
        return False
    if row.get('sub_type') not in ALLOWED_SPECIAL_SUBTYPES:
        return False
    text = row.get('text', '')
    if not text:
        return False
    if text[-1] in EXCLUDED_SPECIAL_SUFFIXES:
        return False
    if len(text) >= 2 and text[1].isdigit():
        return False
    return True


def write_manifest(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='Build batch1 first-char manifests D1/D2/D3')
    ap.add_argument('--base_manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_from_rawtrain.csv')
    ap.add_argument('--cblprd_manifest', default='/home/wzzz/LPRNet/manifests/cblprd_cv_geom_manifest.csv')
    ap.add_argument('--out_dir', default='/home/wzzz/LPRNet/manifests/firstchar_batch1')
    args = ap.parse_args()

    base_path = Path(args.base_manifest)
    cblprd_path = Path(args.cblprd_manifest)
    out_dir = Path(args.out_dir)

    base_fields, base_rows_all = read_csv(base_path)
    cbl_fields, cbl_rows_all = read_csv(cblprd_path)
    merged_fields = list(base_fields)
    for f in cbl_fields:
        if f not in merged_fields:
            merged_fields.append(f)

    base_train = [r for r in base_rows_all if r.get('split') == 'train']
    cbl_train = [r for r in cbl_rows_all if r.get('split') == 'train']
    cbl_g8 = [r for r in cbl_train if r.get('family') == 'green8']
    cbl_n7 = [r for r in cbl_train if r.get('family') == 'normal7']
    cbl_sp = [r for r in cbl_train if selected_special(r)]

    d1 = ensure_fields(base_train + cbl_g8, merged_fields)
    d2 = ensure_fields(base_train + cbl_g8 + cbl_n7, merged_fields)
    d3 = ensure_fields(base_train + cbl_g8 + cbl_n7 + cbl_sp, merged_fields)

    outputs = {
        'D1_firstchar_manifest_green8_only_v1_train.csv': d1,
        'D2_firstchar_manifest_green8_normal7_v1_train.csv': d2,
        'D3_firstchar_manifest_green8_normal7_selectedspecial_v1_train.csv': d3,
    }

    summary = {
        'inputs': {
            'base_manifest': str(base_path),
            'cblprd_manifest': str(cblprd_path),
            'base_train_rows': len(base_train),
            'cblprd_train_rows': len(cbl_train),
            'cblprd_green8_rows': len(cbl_g8),
            'cblprd_normal7_rows': len(cbl_n7),
            'cblprd_selected_special_rows': len(cbl_sp),
            'selected_special_rule': {
                'allowed_sub_types': sorted(ALLOWED_SPECIAL_SUBTYPES),
                'excluded_suffixes': sorted(EXCLUDED_SPECIAL_SUFFIXES),
                'exclude_if_second_char_is_digit': True,
            },
        },
        'outputs': {},
    }

    for name, rows in outputs.items():
        out_path = out_dir / name
        write_manifest(out_path, merged_fields, rows)
        summary['outputs'][name] = summarize_rows(rows)
        summary['outputs'][name]['path'] = str(out_path)

    summary_path = out_dir / 'firstchar_batch1_manifest_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
