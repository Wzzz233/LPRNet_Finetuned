#!/usr/bin/env python3
"""
StageA redesign manifest builder
- real-primary, synthetic-support
- explicit source quotas instead of giant-pool random truncation
- build stageA train/val and foundation proxies
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

BASE = Path('/home/wzzz/LPRNet')
LABEL_DIR = BASE / 'labels/curriculum_gray3'
OUT_DIR = BASE / 'manifests/curriculum_gray3_stagea_redesign'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIAL_CHARS = set('警学挂领使字')

BLUE_TOTAL = 44694
GREEN_TOTAL = 37161
BLUE_QUOTA = {
    'ccpd2019': 7106,
    'crpd_clean': 19588,
    'cblprd': 18000,
}
GREEN_QUOTA = {
    'ccpd2020': 996,
    'cblprd': 22000,
    'green_exact_quad': 8973,
    'green_edgefit_simple': 5192,
}


def load_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    return rows


def write_manifest(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['img_path', 'text', 'family', 'source', 'split'])
        for r in rows:
            w.writerow([r['img_path'], r['text'], r['family'], r['source'], 'test'])


def filter_crpd(rows):
    return [r for r in rows if r['family'] == 'normal7' and not any(c in r['text'] for c in SPECIAL_CHARS)]


def province_counter(rows):
    c = Counter()
    for r in rows:
        if r.get('text'):
            c[r['text'][0]] += 1
    return c


def source_summary(rows):
    return Counter(r['source'] for r in rows)


def rebalance_major_province(rows, major='皖', cap=3000, seed=42):
    rng = random.Random(seed)
    major_rows = [r for r in rows if r['text'][0] == major]
    other_rows = [r for r in rows if r['text'][0] != major]
    if len(major_rows) > cap:
        major_rows = rng.sample(major_rows, cap)
    out = other_rows + major_rows
    rng.shuffle(out)
    return out


def sample_rows(rows, target, seed=42):
    rng = random.Random(seed)
    if len(rows) <= target:
        return list(rows)
    return rng.sample(rows, target)


def sample_balanced_by_province(rows, target, seed=42):
    rng = random.Random(seed)
    if len(rows) <= target:
        out = list(rows)
        rng.shuffle(out)
        return out
    by = defaultdict(list)
    for r in rows:
        by[r['text'][0]].append(r)
    provs = sorted(by.keys())
    base = target // len(provs)
    rem = target % len(provs)
    selected = []
    leftovers = []
    for idx, prov in enumerate(provs):
        bucket = list(by[prov])
        rng.shuffle(bucket)
        take = min(len(bucket), base + (1 if idx < rem else 0))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])
    if len(selected) < target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:target-len(selected)])
    rng.shuffle(selected)
    return selected[:target]


def no_overlap(*groups):
    named_sets = []
    for idx, rows in enumerate(groups):
        named_sets.append((idx, set(r['img_path'] for r in rows)))
    for i in range(len(named_sets)):
        for j in range(i+1, len(named_sets)):
            idx_i, set_i = named_sets[i]
            idx_j, set_j = named_sets[j]
            inter = set_i & set_j
            if inter and not ({idx_i, idx_j} == {2, 4}) and not ({idx_i, idx_j} == {3, 4}):
                raise RuntimeError(f'leakage detected: group {idx_i} vs {idx_j}, count={len(inter)}')


def build_stagea_train():
    ccpd2019 = [r for r in load_csv(LABEL_DIR / 'ccpd2019_train.csv') if r['family'] == 'normal7']
    ccpd2019_rb = rebalance_major_province(ccpd2019, major='皖', cap=3000, seed=42)

    crpd_clean = []
    for name in ['crpd_crpd_single_train.csv', 'crpd_crpd_double_train.csv', 'crpd_crpd_multi_train.csv']:
        crpd_clean.extend(filter_crpd(load_csv(LABEL_DIR / name)))

    cblprd_blue = [r for r in load_csv(LABEL_DIR / 'cblprd_blue_train.csv') if r['family'] == 'normal7']

    blue_rows = []
    blue_rows.extend(sample_balanced_by_province(ccpd2019_rb, BLUE_QUOTA['ccpd2019'], seed=42))
    blue_rows.extend(sample_balanced_by_province(crpd_clean, BLUE_QUOTA['crpd_clean'], seed=43))
    blue_rows.extend(sample_balanced_by_province(cblprd_blue, BLUE_QUOTA['cblprd'], seed=44))

    ccpd2020 = [r for r in load_csv(LABEL_DIR / 'ccpd2020_train.csv') if r['family'] == 'green8']
    ccpd2020_rb = rebalance_major_province(ccpd2020, major='皖', cap=800, seed=45)
    cblprd_green = [r for r in load_csv(LABEL_DIR / 'cblprd_green_train.csv') if r['family'] == 'green8']
    green_exact = [r for r in load_csv(LABEL_DIR / 'green_exact_quad_train.csv') if r['family'] == 'green8']
    edgefit = [r for r in load_csv(LABEL_DIR / 'green_edgefit_simple_train.csv') if r['family'] == 'green8']

    green_rows = []
    green_rows.extend(sample_balanced_by_province(ccpd2020_rb, GREEN_QUOTA['ccpd2020'], seed=46))
    green_rows.extend(sample_balanced_by_province(cblprd_green, GREEN_QUOTA['cblprd'], seed=47))
    green_rows.extend(sample_balanced_by_province(green_exact, GREEN_QUOTA['green_exact_quad'], seed=48))
    green_rows.extend(sample_balanced_by_province(edgefit, GREEN_QUOTA['green_edgefit_simple'], seed=49))

    assert len(blue_rows) == BLUE_TOTAL, len(blue_rows)
    assert len(green_rows) == GREEN_TOTAL, len(green_rows)

    stagea = blue_rows + green_rows
    random.Random(50).shuffle(stagea)
    return stagea


def build_val():
    blue = []
    ccpd_val = [r for r in load_csv(LABEL_DIR / 'ccpd2019_val.csv') if r['family'] == 'normal7']
    ccpd_val_rb = rebalance_major_province(ccpd_val, major='皖', cap=330, seed=60)
    blue.extend(sample_balanced_by_province(ccpd_val_rb, 1800, seed=61))
    blue.extend(sample_balanced_by_province([r for r in load_csv(LABEL_DIR / 'cblprd_blue_val.csv') if r['family']=='normal7'], 1800, seed=62))
    crpd_val = []
    for name in ['crpd_crpd_single_val.csv', 'crpd_crpd_double_val.csv', 'crpd_crpd_multi_val.csv']:
        crpd_val.extend(filter_crpd(load_csv(LABEL_DIR / name)))
    blue.extend(sample_balanced_by_province(crpd_val, 2400, seed=63))

    green = []
    ccpd2020_val = [r for r in load_csv(LABEL_DIR / 'ccpd2020_val.csv') if r['family'] == 'green8']
    green.extend(sample_balanced_by_province(rebalance_major_province(ccpd2020_val, major='皖', cap=180, seed=64), min(800, len(ccpd2020_val)), seed=65))
    green.extend(sample_balanced_by_province([r for r in load_csv(LABEL_DIR / 'cblprd_green_val.csv') if r['family']=='green8'], 1800, seed=66))
    green.extend(sample_balanced_by_province([r for r in load_csv(LABEL_DIR / 'green_exact_quad_val.csv') if r['family']=='green8'], 1700, seed=67))
    green.extend(sample_balanced_by_province([r for r in load_csv(LABEL_DIR / 'green_edgefit_simple_val.csv') if r['family']=='green8'], 1700, seed=68))

    val_rows = blue + green
    random.Random(69).shuffle(val_rows)
    return val_rows


def build_proxies(train_rows, val_rows):
    used_paths = set(r['img_path'] for r in train_rows) | set(r['img_path'] for r in val_rows)

    blue_proxy = []
    for rows, target, seed in [
        ([r for r in load_csv(LABEL_DIR / 'ccpd2019_val.csv') if r['family'] == 'normal7' and r['img_path'] not in used_paths], 600, 71),
        ([r for r in load_csv(LABEL_DIR / 'cblprd_blue_val.csv') if r['family'] == 'normal7' and r['img_path'] not in used_paths], 600, 72),
        (sum([filter_crpd(load_csv(LABEL_DIR / n)) for n in ['crpd_crpd_single_val.csv', 'crpd_crpd_double_val.csv', 'crpd_crpd_multi_val.csv']], []), 800, 73),
    ]:
        rows = [r for r in rows if r['img_path'] not in used_paths]
        blue_proxy.extend(sample_balanced_by_province(rows, min(target, len(rows)), seed=seed))

    green_proxy = []
    for rows, target, seed in [
        ([r for r in load_csv(LABEL_DIR / 'ccpd2020_val.csv') if r['family'] == 'green8' and r['img_path'] not in used_paths], 300, 74),
        ([r for r in load_csv(LABEL_DIR / 'cblprd_green_val.csv') if r['family'] == 'green8' and r['img_path'] not in used_paths], 700, 75),
        ([r for r in load_csv(LABEL_DIR / 'green_exact_quad_val.csv') if r['family'] == 'green8' and r['img_path'] not in used_paths], 500, 76),
        ([r for r in load_csv(LABEL_DIR / 'green_edgefit_simple_val.csv') if r['family'] == 'green8' and r['img_path'] not in used_paths], 500, 77),
    ]:
        rows = [r for r in rows if r['img_path'] not in used_paths]
        green_proxy.extend(sample_balanced_by_province(rows, min(target, len(rows)), seed=seed))

    mixed = list(blue_proxy) + list(green_proxy)
    random.Random(78).shuffle(mixed)
    return blue_proxy, green_proxy, mixed


def build_summary(stagea, val_rows, blue_proxy, green_proxy, mixed_proxy):
    return {
        'stageA_train_total': len(stagea),
        'stageA_train_source': dict(source_summary(stagea)),
        'stageA_train_family': dict(Counter(r['family'] for r in stagea)),
        'stageA_train_blue_top10_province': province_counter([r for r in stagea if r['family']=='normal7']).most_common(10),
        'stageA_train_green_top10_province': province_counter([r for r in stagea if r['family']=='green8']).most_common(10),
        'val_total': len(val_rows),
        'val_source': dict(source_summary(val_rows)),
        'proxy_blue_total': len(blue_proxy),
        'proxy_blue_source': dict(source_summary(blue_proxy)),
        'proxy_green_total': len(green_proxy),
        'proxy_green_source': dict(source_summary(green_proxy)),
        'proxy_mixed_total': len(mixed_proxy),
    }


def main():
    stagea = build_stagea_train()
    val_rows = build_val()
    blue_proxy, green_proxy, mixed_proxy = build_proxies(stagea, val_rows)

    no_overlap(stagea, val_rows, blue_proxy, green_proxy, mixed_proxy)

    write_manifest(stagea, OUT_DIR / 'train_stageA.csv')
    write_manifest(val_rows, OUT_DIR / 'val.csv')
    write_manifest(blue_proxy, OUT_DIR / 'proxy_stageA_blue_simple.csv')
    write_manifest(green_proxy, OUT_DIR / 'proxy_stageA_green_simple.csv')
    write_manifest(mixed_proxy, OUT_DIR / 'proxy_stageA_mixed_foundation.csv')

    summary = build_summary(stagea, val_rows, blue_proxy, green_proxy, mixed_proxy)
    with open(OUT_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        for k, v in summary.items():
            f.write(f'{k}: {v}\n')
    print('Wrote manifests to', OUT_DIR)
    for k, v in summary.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
