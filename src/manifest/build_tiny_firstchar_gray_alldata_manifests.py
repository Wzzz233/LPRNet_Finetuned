#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path

BASE_GREEN = Path('/home/wzzz/LPRNet/manifests/firstchar_batch1/D2_firstchar_manifest_green8_normal7_v1_train.csv')
FULL_CBLPRD = Path('/home/wzzz/LPRNet/manifests/cblprd_cv_geom_manifest.csv')
OUT_DIR = Path('/home/wzzz/LPRNet/manifests/firstchar_tiny_gray_alldata_v1')
TRAIN_OUT = OUT_DIR / 'train.csv'
TEST_OUT = OUT_DIR / 'test.csv'
SUMMARY_OUT = OUT_DIR / 'summary.json'

FIELDNAMES = None


def read_rows(path, split=None, allowed_families=None):
    rows = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        global FIELDNAMES
        if FIELDNAMES is None:
            FIELDNAMES = list(reader.fieldnames)
        for row in reader:
            if split is not None and (row.get('split') or '').strip() != split:
                continue
            if allowed_families is not None and (row.get('family') or '').strip() not in allowed_families:
                continue
            rows.append(row)
    return rows


def key_of(row):
    return ((row.get('img_path') or '').strip(), (row.get('text') or '').strip())


def summarize(rows):
    fam = Counter((r.get('family') or '').strip() for r in rows)
    split = Counter((r.get('split') or '').strip() for r in rows)
    source = Counter((r.get('source') or '').strip() for r in rows)
    prov = Counter(((r.get('text') or '')[:1] or '__empty__') for r in rows)
    return {
        'total': len(rows),
        'family': dict(fam),
        'split': dict(split),
        'source_top10': prov_and_count(source, 10),
        'province_top15': prov_and_count(prov, 15),
    }


def prov_and_count(counter, n):
    return [{k: v} for k, v in counter.most_common(n)]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    train_rows = read_rows(BASE_GREEN, split='train', allowed_families={'green8', 'normal7'})
    seen = {key_of(r) for r in train_rows}

    cblprd_train = []
    for row in read_rows(FULL_CBLPRD, split='train', allowed_families={'green8', 'normal7'}):
        k = key_of(row)
        if k in seen:
            continue
        seen.add(k)
        cblprd_train.append(row)

    merged_train = train_rows + cblprd_train
    for row in merged_train:
        row['split'] = 'train'

    test_rows = [
        row for row in read_rows(FULL_CBLPRD, split='val', allowed_families={'green8', 'normal7'})
        if (row.get('text') or '')[:1]
    ]
    for row in test_rows:
        row['split'] = 'test'

    write_rows(TRAIN_OUT, merged_train)
    write_rows(TEST_OUT, test_rows)

    summary = {
        'train': summarize(merged_train),
        'test': summarize(test_rows),
        'paths': {
            'train_manifest': str(TRAIN_OUT),
            'test_manifest': str(TEST_OUT),
            'base_green': str(BASE_GREEN),
            'cblprd': str(FULL_CBLPRD),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
