#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build E9 training manifests by appending exact-template generated rows to a fixed base manifest.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_manifest', required=True)
    ap.add_argument('--generated_manifest', required=True)
    ap.add_argument('--out_manifest', required=True)
    ap.add_argument('--label', required=True)
    args = ap.parse_args()

    base_rows = read_csv(args.base_manifest)
    gen_rows = read_csv(args.generated_manifest)
    gen_rows = [r for r in gen_rows if r.get('split') == 'train']

    fieldnames = list(base_rows[0].keys())
    for r in gen_rows:
        for k in fieldnames:
            r.setdefault(k, '')
    out_rows = list(base_rows) + gen_rows

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    green_train = [r for r in out_rows if r.get('family') == 'green8' and r.get('split') == 'train']
    source_counts = Counter(r.get('source', '?') for r in green_train)
    province_counts = Counter((r.get('text', '')[:1] or '?') for r in gen_rows if r.get('text'))

    report = {
        'label': args.label,
        'base_manifest': args.base_manifest,
        'generated_manifest': args.generated_manifest,
        'out_manifest': args.out_manifest,
        'base_rows': len(base_rows),
        'generated_train_rows': len(gen_rows),
        'out_rows': len(out_rows),
        'green8_train_rows': len(green_train),
        'generated_province_counts': dict(province_counts),
        'green8_train_source_counts_top20': dict(source_counts.most_common(20)),
    }
    report_path = str(Path(args.out_manifest).with_suffix('.report.json'))
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
