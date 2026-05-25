#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_csv(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bucket_of(row):
    rel = row.get('img_rel_path') or ''
    parts = rel.split('/')
    if len(parts) >= 3:
        return parts[2]
    return ''


def main():
    ap = argparse.ArgumentParser(description='Append extra v4 extreme rows onto an existing E2 manifest without removing anything.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--append-manifest', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    args = ap.parse_args()

    base_path = Path(args.base_manifest)
    append_path = Path(args.append_manifest)
    out_path = Path(args.out_manifest)
    summary_path = Path(args.out_summary)

    base_fields, base_rows = load_csv(base_path)
    append_fields, append_rows = load_csv(append_path)
    if base_fields != append_fields:
        raise RuntimeError('Fieldnames mismatch between base manifest and append manifest')

    append_train = [r for r in append_rows if (r.get('split') or '') == 'train']
    if not append_train:
        raise RuntimeError('append manifest has no train rows')

    bad_buckets = [r.get('img_rel_path') for r in append_train if bucket_of(r) != 'board_extreme_tail']
    if bad_buckets:
        raise RuntimeError(f'append manifest contains non-extreme rows, example={bad_buckets[:3]}')

    existing_pairs = {(r.get('img_path') or '', r.get('text') or '') for r in base_rows}
    duplicate_rows = [
        {'img_path': r.get('img_path'), 'text': r.get('text')}
        for r in append_train if (r.get('img_path') or '', r.get('text') or '') in existing_pairs
    ]
    if duplicate_rows:
        raise RuntimeError(f'append manifest duplicates existing base rows, example={duplicate_rows[:3]}')

    new_rows = list(base_rows) + append_train
    write_csv(out_path, base_fields, new_rows)

    base_source = Counter((r.get('source') or '') for r in base_rows if (r.get('split') or '') == 'train')
    new_source = Counter((r.get('source') or '') for r in new_rows if (r.get('split') or '') == 'train')
    append_provinces = Counter((r.get('text') or '')[:1] for r in append_train if r.get('text'))
    append_buckets = Counter(bucket_of(r) for r in append_train)

    summary = {
        'mode': 'append_only',
        'base_manifest': str(base_path),
        'append_manifest': str(append_path),
        'out_manifest': str(out_path),
        'base_total': len(base_rows),
        'append_total': len(append_train),
        'new_total': len(new_rows),
        'base_train_source_counts': dict(base_source),
        'new_train_source_counts': dict(new_source),
        'append_bucket_counts': dict(append_buckets),
        'append_province_counts': dict(sorted(append_provinces.items())),
        'append_examples': [
            {
                'img_rel_path': r.get('img_rel_path'),
                'text': r.get('text'),
                'source': r.get('source'),
            }
            for r in append_train[:10]
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
