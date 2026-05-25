#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import generate_green_edgefit_v4_boardlike_equalprov as base


def read_manifest_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def read_dump_suffixes(path: Path):
    suffixes = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            gt = (row.get('gt_text') or '').strip()
            if len(gt) != 8:
                continue
            suffix = gt[1:]
            if len(suffix) != 7:
                continue
            suffixes.append(suffix)
    return suffixes


def normalize_suffix_city_a(suffix: str):
    suffix = suffix.strip().upper()
    if len(suffix) != 7:
        return None
    return 'A' + suffix[1:]


def repeated_tail_score(text: str):
    tail = text[-4:]
    counts = Counter(tail)
    score = 0
    score += 2.0 * max(counts.values())
    score += 1.0 * sum(ch in {'0', '1', '2'} for ch in tail)
    score += 0.8 * (tail in {'0111', '1111', '0222', '0022', '0202', '0220', '2022', '2202', '2222'})
    return score


def build_cluster2_bank(rows, dump_cluster2_suffixes, seed, target_count):
    rng = random.Random(seed)
    df_pool = []
    ov_pool = []
    for row in rows:
        if row.get('family') != 'green8' or row.get('split') != 'train':
            continue
        text = (row.get('text') or '').strip().upper()
        if len(text) != 8 or text[0] == '皖':
            continue
        suffix = normalize_suffix_city_a(text[1:])
        if not suffix:
            continue
        pos2 = suffix[1]
        if pos2 in {'D', 'F'}:
            df_pool.append(suffix)
        else:
            ov_pool.append(suffix)
    df_pool = sorted(set(df_pool))
    ov_pool = sorted(set(ov_pool))
    rng.shuffle(df_pool)
    rng.shuffle(ov_pool)

    bank = []
    for suffix in dump_cluster2_suffixes:
        suffix = normalize_suffix_city_a(suffix)
        if suffix and suffix not in bank:
            bank.append(suffix)

    want_df = max(0, min(len(df_pool), 20))
    want_ov = max(0, min(len(ov_pool), target_count - len(bank) - want_df))

    for suffix in df_pool:
        if len([x for x in bank if x[1] in {'D', 'F'}]) >= want_df:
            break
        if suffix not in bank:
            bank.append(suffix)
    for suffix in ov_pool:
        if len(bank) >= target_count:
            break
        if suffix not in bank:
            bank.append(suffix)

    merged_pool = df_pool + ov_pool
    for suffix in merged_pool:
        if len(bank) >= target_count:
            break
        if suffix not in bank:
            bank.append(suffix)
    return bank[:target_count]


def build_cluster3_bank(rows, dump_cluster3_suffixes, seed, target_count):
    rng = random.Random(seed + 1)
    candidates = []
    for row in rows:
        if row.get('family') != 'green8' or row.get('split') != 'train':
            continue
        text = (row.get('text') or '').strip().upper()
        if len(text) != 8 or text[0] == '皖':
            continue
        suffix = normalize_suffix_city_a(text[1:])
        if not suffix:
            continue
        score = repeated_tail_score('X' + suffix)
        pos2 = suffix[1]
        if pos2 in {'D', 'F'}:
            score += 0.5
        candidates.append((score, suffix))
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: x[0], reverse=True)

    bank = []
    for suffix in dump_cluster3_suffixes:
        suffix = normalize_suffix_city_a(suffix)
        if suffix and suffix not in bank:
            bank.append(suffix)

    for _, suffix in candidates:
        if len(bank) >= target_count:
            break
        if suffix not in bank:
            bank.append(suffix)
    return bank[:target_count]


def main():
    ap = argparse.ArgumentParser(description='Build shared suffix/text banks for E17 experiments.')
    ap.add_argument('--base-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_e12_e9c_append_boarddump_anticollapse_5prov_1200.csv')
    ap.add_argument('--cluster2-csv', default='/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv')
    ap.add_argument('--cluster3-csv', default='/home/wzzz/LPRNet/tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/tmp/e17_shared_banks')
    ap.add_argument('--seed', type=int, default=20260417)
    ap.add_argument('--cluster2-count', type=int, default=30)
    ap.add_argument('--cluster3-count', type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest_rows(Path(args.base_manifest))
    c2 = build_cluster2_bank(rows, read_dump_suffixes(Path(args.cluster2_csv)), args.seed, args.cluster2_count)
    c3 = build_cluster3_bank(rows, read_dump_suffixes(Path(args.cluster3_csv)), args.seed, args.cluster3_count)

    cluster2_json = {
        'name': 'e17_cluster2_suffix_bank',
        'description': 'Shared non-Anhui suffix bank for province-only collapse recovery.',
        'provinces': list(base.NON_ANHUI_PROVINCES),
        'suffixes': c2,
        'count': len(c2),
    }
    cluster3_json = {
        'name': 'e17_cluster3_suffix_bank',
        'description': 'Shared non-Anhui suffix bank for cluster3 transition/tail recovery.',
        'provinces': list(base.NON_ANHUI_PROVINCES),
        'suffixes': c3,
        'count': len(c3),
    }

    (out_dir / 'suffix_bank_cluster2.json').write_text(json.dumps(cluster2_json, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'suffix_bank_cluster3.json').write_text(json.dumps(cluster3_json, ensure_ascii=False, indent=2), encoding='utf-8')

    summary = {
        'base_manifest': str(args.base_manifest),
        'cluster2_bank_count': len(c2),
        'cluster3_bank_count': len(c3),
        'cluster2_pos2_counts': dict(sorted(Counter(x[1] for x in c2).items())),
        'cluster3_pos2_counts': dict(sorted(Counter(x[1] for x in c3).items())),
        'cluster2_examples': c2[:10],
        'cluster3_examples': c3[:10],
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
