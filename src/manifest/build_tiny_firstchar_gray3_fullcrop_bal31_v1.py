#!/usr/bin/env python3
import csv
import json
import random
from collections import Counter
from pathlib import Path

BASE_MANIFEST = Path('/home/wzzz/LPRNet/manifests/firstchar_tiny_gray_alldata_v1/train.csv')
OUT_DIR = Path('/home/wzzz/LPRNet/manifests/firstchar_tiny_gray3_fullcrop_bal31_v1')
TRAIN_OUT = OUT_DIR / 'train.csv'
SUMMARY_OUT = OUT_DIR / 'summary.json'
SEED = 20260421
EXPECTED_PROVINCES = 31

FIELDNAMES = None


def read_rows(path: Path):
    global FIELDNAMES
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        FIELDNAMES = list(reader.fieldnames)
        return [row for row in reader if (row.get('text') or '')[:1]]


def province_of(row):
    return (row.get('text') or '')[:1]


def summarize(rows):
    prov = Counter(province_of(r) for r in rows)
    fam = Counter((r.get('family') or '').strip() for r in rows)
    source = Counter((r.get('source') or '').strip() for r in rows)
    dataset = Counter((r.get('dataset_name') or '').strip() for r in rows)
    split = Counter((r.get('split') or '').strip() for r in rows)
    counts = list(prov.values())
    return {
        'total': len(rows),
        'province_count': len(prov),
        'min_per_province': min(counts) if counts else 0,
        'max_per_province': max(counts) if counts else 0,
        'family': dict(fam),
        'split': dict(split),
        'source_top10': [{k: v} for k, v in source.most_common(10)],
        'dataset_top10': [{k: v} for k, v in dataset.most_common(10)],
        'province_counts': dict(sorted(prov.items())),
    }


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read_rows(BASE_MANIFEST)
    by_prov = {}
    for row in rows:
        by_prov.setdefault(province_of(row), []).append(row)

    if len(by_prov) != EXPECTED_PROVINCES:
        raise RuntimeError(f'Expected {EXPECTED_PROVINCES} provinces, got {len(by_prov)}')

    target = min(len(v) for v in by_prov.values())
    rng = random.Random(SEED)
    balanced = []
    for prov in sorted(by_prov):
        bucket = list(by_prov[prov])
        rng.shuffle(bucket)
        chosen = bucket[:target]
        for row in chosen:
            row = dict(row)
            row['split'] = 'train'
            row['ocr_preproc'] = 'gray3'
            row['ocr_resize_mode'] = 'stretch'
            balanced.append(row)

    rng.shuffle(balanced)
    write_rows(TRAIN_OUT, balanced)

    summary = {
        'seed': SEED,
        'base_manifest': str(BASE_MANIFEST),
        'target_per_province': target,
        'train': summarize(balanced),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
