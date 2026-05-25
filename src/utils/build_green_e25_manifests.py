#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f'no rows to write: {path}')
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def dedup_rows(rows):
    seen = set()
    out = []
    for r in rows:
        key = (r.get('img_path', ''), r.get('text', ''))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def prefix(row):
    text = str(row.get('text') or '')
    return text[:1]


def sample_rows(rows, n, seed):
    if n <= 0 or not rows:
        return []
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    return [rows[i] for i in idxs]


def summarize(rows):
    return {
        'count': len(rows),
        'source_counts': dict(sorted(Counter(r.get('source', '') for r in rows).items())),
        'dataset_counts': dict(sorted(Counter(r.get('dataset_name', '') for r in rows).items())),
        'prefix_counts': dict(sorted(Counter(prefix(r) for r in rows).items())),
    }


def main():
    ap = argparse.ArgumentParser(description='Build E25 stageA/stageB manifests for cluster2 representation rebuild.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--append-manifest', required=True)
    ap.add_argument('--out-stagea', required=True)
    ap.add_argument('--out-stageb', required=True)
    ap.add_argument('--out-summary', required=True)
    ap.add_argument('--seed', type=int, default=20260419)
    args = ap.parse_args()

    base_rows = read_rows(Path(args.base_manifest))
    append_rows = read_rows(Path(args.append_manifest))

    train_rows = [r for r in base_rows if str(r.get('split')) == 'train' and str(r.get('family')) == 'green8']
    append_train = [r for r in append_rows if str(r.get('split')) == 'train' and str(r.get('family')) == 'green8']

    # Stage A: target-domain preadaptation with explicit anchors and hard negatives.
    stagea = []
    target_sources_all = {'e12_boarddump_anticollapse_5prov_1200', 'board_native_e7_v2', 'board_native_cluster1_append_v1'}
    stagea.extend([r for r in train_rows if r.get('source') in target_sources_all])

    real_non_anhui = [r for r in train_rows if r.get('is_real') == '1' and prefix(r) != '皖']
    real_anhui = [r for r in train_rows if r.get('is_real') == '1' and prefix(r) == '皖']
    stagea.extend(real_non_anhui)
    stagea.extend(sample_rows(real_anhui, 200, args.seed + 1))

    v4_non_anhui = [r for r in train_rows if r.get('source') == 'v4_boardlike_edgefit' and prefix(r) != '皖']
    syn_non_anhui = [r for r in train_rows if r.get('source') == 'synthetic_exact_quad' and prefix(r) != '皖']
    syn_anhui = [r for r in train_rows if r.get('source') == 'synthetic_exact_quad' and prefix(r) == '皖']
    stagea.extend(sample_rows(v4_non_anhui, 600, args.seed + 2))
    stagea.extend(sample_rows(syn_non_anhui, 600, args.seed + 3))
    stagea.extend(sample_rows(syn_anhui, 300, args.seed + 4))

    tier3_non_anhui = [r for r in train_rows if r.get('source') == 'synthetic_exact_quad_edgefit_tier3_v3_su_conservative' and prefix(r) != '皖']
    stagea.extend(sample_rows(tier3_non_anhui, 400, args.seed + 5))

    stagea.extend(append_train)
    stagea = dedup_rows(stagea)

    # Stage B: full reintegration while preserving strong target exposure.
    stageb = dedup_rows(list(train_rows) + list(append_train))

    write_rows(Path(args.out_stagea), stagea)
    write_rows(Path(args.out_stageb), stageb)

    summary = {
        'base_manifest': args.base_manifest,
        'append_manifest': args.append_manifest,
        'append_train_count': len(append_train),
        'stagea': summarize(stagea),
        'stageb': summarize(stageb),
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
