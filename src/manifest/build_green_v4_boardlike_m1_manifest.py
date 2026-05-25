#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REMOVE_SOURCE = 'synthetic_exact_quad_edgefit_tier3_v3_su_conservative'
ADD_SOURCE = 'v4_boardlike_edgefit'


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


def choose_removed_rows(train_rows, remove_count):
    candidates = [r for r in train_rows if (r.get('source') or '') == REMOVE_SOURCE]
    if len(candidates) < remove_count:
        raise RuntimeError(f'Not enough {REMOVE_SOURCE} rows in train: have={len(candidates)} need={remove_count}')
    candidates = sorted(candidates, key=lambda r: ((r.get('img_rel_path') or r.get('img_path') or ''), r.get('text') or ''))
    return candidates[:remove_count]


def main():
    ap = argparse.ArgumentParser(description='Build M1 manifest by equal-count replacing old tier3 train rows with v4 boardlike train rows.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--v4-manifest', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-summary', required=True)
    args = ap.parse_args()

    base_path = Path(args.base_manifest)
    v4_path = Path(args.v4_manifest)
    out_path = Path(args.out_manifest)
    out_summary = Path(args.out_summary)

    fieldnames, base_rows = load_csv(base_path)
    v4_fields, v4_rows = load_csv(v4_path)
    if fieldnames != v4_fields:
        raise RuntimeError('Fieldnames mismatch between base manifest and v4 manifest')

    base_train = [r for r in base_rows if (r.get('split') or '') == 'train']
    base_other = [r for r in base_rows if (r.get('split') or '') != 'train']
    v4_train = [r for r in v4_rows if (r.get('split') or '') == 'train']

    remove_count = len(v4_train)
    removed_rows = choose_removed_rows(base_train, remove_count)
    removed_keys = {(r.get('img_path') or '', r.get('text') or '') for r in removed_rows}
    kept_train = [r for r in base_train if (r.get('img_path') or '', r.get('text') or '') not in removed_keys]
    new_train = kept_train + v4_train
    new_rows = new_train + base_other
    write_csv(out_path, fieldnames, new_rows)

    base_train_src = Counter((r.get('source') or '') for r in base_train)
    new_train_src = Counter((r.get('source') or '') for r in new_train)
    v4_bucket = Counter((r.get('dataset_name') or '', r.get('source') or '') for r in v4_train)
    prov = Counter((r.get('text') or '')[:1] for r in v4_train if r.get('text'))

    summary = {
        'base_manifest': str(base_path),
        'v4_manifest': str(v4_path),
        'out_manifest': str(out_path),
        'remove_source': REMOVE_SOURCE,
        'add_source': ADD_SOURCE,
        'removed_count': len(removed_rows),
        'added_count': len(v4_train),
        'base_train_total': len(base_train),
        'new_train_total': len(new_train),
        'base_train_source_counts': dict(base_train_src),
        'new_train_source_counts': dict(new_train_src),
        'removed_examples': [
            {
                'img_rel_path': r.get('img_rel_path'),
                'text': r.get('text'),
                'source': r.get('source'),
            }
            for r in removed_rows[:10]
        ],
        'added_examples': [
            {
                'img_rel_path': r.get('img_rel_path'),
                'text': r.get('text'),
                'source': r.get('source'),
            }
            for r in v4_train[:10]
        ],
        'v4_dataset_source_counts': {f'{ds}|{src}': c for (ds, src), c in sorted(v4_bucket.items())},
        'v4_top_provinces': prov.most_common(10),
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
