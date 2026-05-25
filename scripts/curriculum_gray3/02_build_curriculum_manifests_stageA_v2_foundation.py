#!/usr/bin/env python3
"""
Build StageA v2 manifests with explicit semantic groups:
- blue_real_primary
- blue_support
- green_real_primary
- green_support
And build graduation proxies:
- blue_real_foundation_proxy
- green_real_foundation_proxy
- green_bridge_proxy
- support_proxy

Output CSVs keep full geometry-aware fields so they are directly train/eval ready.
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/home/wzzz/LPRNet')
LABEL_DIR = BASE / 'labels/curriculum_gray3'
OUT_DIR = BASE / 'manifests/curriculum_gray3_stagea_v2_foundation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
SPECIAL_CHARS = set('警学挂领使字')
GREEN_MIN_OTHER_PROV = 400

TRAIN_CONFIG = {
    'blue_real_primary': {
        'ccpd2019': 12000,
        'crpd_crpd_single': 12000,
        'crpd_crpd_double': 4500,
        'crpd_crpd_multi': 1800,
    },
    'blue_support': {
        'cblprd': 9000,
    },
    'green_real_primary': {
        'ccpd2020': 2400,
    },
    'green_support': {
        'cblprd': 8000,
        'green_exact_quad': 4500,
        'green_edgefit_simple': 2000,
    },
}

VAL_CONFIG = {
    'ccpd2019': 1800,
    'crpd_crpd_single': 1475,
    'crpd_crpd_double': 655,
    'crpd_crpd_multi': 270,
    'ccpd2020': 216,
    'cblprd_blue': 1800,
    'cblprd_green': 1800,
    'green_exact_quad': 1120,
    'green_edgefit_simple': 1054,
}

FIELDNAMES = [
    'img_path', 'text', 'family', 'source', 'split',
    'semantic_group', 'domain_role', 'source_family',
    'preprocess_group',
    'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source', 'bbox_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_quad_pad_ratio',
]


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return [dict(r) for r in csv.DictReader(f)]


def read_crpd_quad(img_path: str):
    p = Path(img_path)
    label_dir = p.parent.parent / 'labels'
    label_file = label_dir / (p.stem + '.txt')
    if not label_file.exists():
        return None
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                return tuple(parts[:8])
    return None


def enrich_geometry(row):
    out = {k: row.get(k, '') for k in ['img_path', 'text', 'family', 'source', 'split']}
    source = row['source']
    if source in ('ccpd2019', 'ccpd2020'):
        out.update({
            'preprocess_group': 'ccpd_board',
            'has_quad': '1',
            'can_parse_ccpd_geom': '1',
            'can_perspective': '1',
            'quad_source': 'ccpd_filename',
            'bbox_source': 'ccpd_filename',
            'ocr_quad_pad_ratio': '0.0',
            'quad_1x': '', 'quad_1y': '', 'quad_2x': '', 'quad_2y': '',
            'quad_3x': '', 'quad_3y': '', 'quad_4x': '', 'quad_4y': '',
        })
    elif source.startswith('crpd'):
        quad = read_crpd_quad(row['img_path'])
        if quad is None:
            raise RuntimeError(f'missing CRPD quad for {row["img_path"]}')
        out.update({
            'preprocess_group': 'ccpd_board',
            'has_quad': '1',
            'can_parse_ccpd_geom': '1',
            'can_perspective': '1',
            'quad_source': 'crpd_label',
            'bbox_source': 'quad_derived',
            'ocr_quad_pad_ratio': '0.0',
            'quad_1x': quad[0], 'quad_1y': quad[1], 'quad_2x': quad[2], 'quad_2y': quad[3],
            'quad_3x': quad[4], 'quad_3y': quad[5], 'quad_4x': quad[6], 'quad_4y': quad[7],
        })
    else:
        out.update({
            'preprocess_group': 'plain_plate',
            'has_quad': '0',
            'can_parse_ccpd_geom': '0',
            'can_perspective': '0',
            'quad_source': 'none',
            'bbox_source': 'none',
            'ocr_quad_pad_ratio': '0.0',
            'quad_1x': '', 'quad_1y': '', 'quad_2x': '', 'quad_2y': '',
            'quad_3x': '', 'quad_3y': '', 'quad_4x': '', 'quad_4y': '',
        })
    return out


def with_semantic(rows, semantic_group):
    domain_role = 'real_primary' if 'real_primary' in semantic_group else 'support'
    out = []
    for r in rows:
        e = enrich_geometry(r)
        e['semantic_group'] = semantic_group
        e['domain_role'] = domain_role
        e['source_family'] = f"{r['source']}__{r['family']}"
        out.append(e)
    return out


def filter_crpd(rows):
    return [r for r in rows if r['family'] == 'normal7' and not any(c in r['text'] for c in SPECIAL_CHARS)]


def province(text):
    return text[0] if text else ''


def balanced_sample(rows, target, seed=42, max_major=None, major='皖'):
    rng = random.Random(seed)
    rows = list(rows)
    by = defaultdict(list)
    for r in rows:
        by[province(r['text'])].append(r)
    if max_major is not None and major in by and len(by[major]) > max_major:
        rng.shuffle(by[major])
        by[major] = by[major][:max_major]
    rows2 = []
    for k, bucket in by.items():
        rng.shuffle(bucket)
        rows2.extend(bucket)
    if len(rows2) <= target:
        rng.shuffle(rows2)
        return rows2
    by = defaultdict(list)
    for r in rows2:
        by[province(r['text'])].append(r)
    provs = sorted(by.keys())
    base = target // len(provs)
    rem = target % len(provs)
    selected, leftovers = [], []
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


def load_train_pools():
    pools = {}
    pools['ccpd2019'] = [r for r in read_csv(LABEL_DIR / 'ccpd2019_train.csv') if r['family'] == 'normal7']
    pools['crpd_crpd_single'] = filter_crpd(read_csv(LABEL_DIR / 'crpd_crpd_single_train.csv'))
    pools['crpd_crpd_double'] = filter_crpd(read_csv(LABEL_DIR / 'crpd_crpd_double_train.csv'))
    pools['crpd_crpd_multi'] = filter_crpd(read_csv(LABEL_DIR / 'crpd_crpd_multi_train.csv'))
    pools['cblprd_blue'] = [r for r in read_csv(LABEL_DIR / 'cblprd_blue_train.csv') if r['family'] == 'normal7']
    pools['ccpd2020'] = [r for r in read_csv(LABEL_DIR / 'ccpd2020_train.csv') if r['family'] == 'green8']
    pools['cblprd_green'] = [r for r in read_csv(LABEL_DIR / 'cblprd_green_train.csv') if r['family'] == 'green8']
    pools['green_exact_quad'] = [r for r in read_csv(LABEL_DIR / 'green_exact_quad_train.csv') if r['family'] == 'green8']
    pools['green_edgefit_simple'] = [r for r in read_csv(LABEL_DIR / 'green_edgefit_simple_train.csv') if r['family'] == 'green8']
    return pools


def load_val_pools():
    pools = {}
    pools['ccpd2019'] = [r for r in read_csv(LABEL_DIR / 'ccpd2019_val.csv') if r['family'] == 'normal7']
    pools['crpd_crpd_single'] = filter_crpd(read_csv(LABEL_DIR / 'crpd_crpd_single_val.csv'))
    pools['crpd_crpd_double'] = filter_crpd(read_csv(LABEL_DIR / 'crpd_crpd_double_val.csv'))
    pools['crpd_crpd_multi'] = filter_crpd(read_csv(LABEL_DIR / 'crpd_crpd_multi_val.csv'))
    pools['cblprd_blue'] = [r for r in read_csv(LABEL_DIR / 'cblprd_blue_val.csv') if r['family'] == 'normal7']
    pools['ccpd2020'] = [r for r in read_csv(LABEL_DIR / 'ccpd2020_val.csv') if r['family'] == 'green8']
    pools['cblprd_green'] = [r for r in read_csv(LABEL_DIR / 'cblprd_green_val.csv') if r['family'] == 'green8']
    pools['green_exact_quad'] = [r for r in read_csv(LABEL_DIR / 'green_exact_quad_val.csv') if r['family'] == 'green8']
    pools['green_edgefit_simple'] = [r for r in read_csv(LABEL_DIR / 'green_edgefit_simple_val.csv') if r['family'] == 'green8']
    pools['green_exact_quad_train'] = [r for r in read_csv(LABEL_DIR / 'green_exact_quad_train.csv') if r['family'] == 'green8']
    return pools


def sample_source(rows, source_key, target, seed):
    if source_key in ('ccpd2019', 'ccpd2020'):
        max_major = int(target * 0.35)
        return balanced_sample(rows, target, seed=seed, max_major=max_major, major='皖')
    return balanced_sample(rows, target, seed=seed)


def write_manifest(rows, path, split):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            row = dict(r)
            row['split'] = split
            w.writerow(row)


def count_by(rows, key):
    return Counter(r[key] for r in rows)


def green_province_counter(rows):
    c = Counter()
    for r in rows:
        if r['family'] == 'green8' and r['text']:
            c[r['text'][0]] += 1
    return c


def ratio(a, b):
    return 0.0 if b == 0 else a / b


def main():
    train_pools = load_train_pools()
    val_pools = load_val_pools()

    train_rows = []
    seed = 100
    for semantic_group, sources in TRAIN_CONFIG.items():
        for source_key, target in sources.items():
            pool_key = source_key
            if source_key == 'cblprd' and 'blue' in semantic_group:
                pool_key = 'cblprd_blue'
            elif source_key == 'cblprd' and 'green' in semantic_group:
                pool_key = 'cblprd_green'
            sampled = sample_source(train_pools[pool_key], source_key if source_key != 'cblprd' else pool_key, target, seed)
            train_rows.extend(with_semantic(sampled, semantic_group))
            seed += 1

    rng = random.Random(999)
    rng.shuffle(train_rows)

    val_rows = []
    val_rows.extend(with_semantic(sample_source(val_pools['ccpd2019'], 'ccpd2019', VAL_CONFIG['ccpd2019'], 200), 'blue_real_primary'))
    val_rows.extend(with_semantic(sample_source(val_pools['crpd_crpd_single'], 'crpd_crpd_single', VAL_CONFIG['crpd_crpd_single'], 201), 'blue_real_primary'))
    val_rows.extend(with_semantic(sample_source(val_pools['crpd_crpd_double'], 'crpd_crpd_double', VAL_CONFIG['crpd_crpd_double'], 202), 'blue_real_primary'))
    val_rows.extend(with_semantic(sample_source(val_pools['crpd_crpd_multi'], 'crpd_crpd_multi', VAL_CONFIG['crpd_crpd_multi'], 203), 'blue_real_primary'))
    val_rows.extend(with_semantic(sample_source(val_pools['cblprd_blue'], 'cblprd_blue', VAL_CONFIG['cblprd_blue'], 204), 'blue_support'))
    val_rows.extend(with_semantic(sample_source(val_pools['ccpd2020'], 'ccpd2020', VAL_CONFIG['ccpd2020'], 205), 'green_real_primary'))
    val_rows.extend(with_semantic(sample_source(val_pools['cblprd_green'], 'cblprd_green', VAL_CONFIG['cblprd_green'], 206), 'green_support'))
    val_rows.extend(with_semantic(sample_source(val_pools['green_exact_quad'], 'green_exact_quad', VAL_CONFIG['green_exact_quad'], 207), 'green_support'))
    val_rows.extend(with_semantic(sample_source(val_pools['green_edgefit_simple'], 'green_edgefit_simple', VAL_CONFIG['green_edgefit_simple'], 208), 'green_support'))
    rng.shuffle(val_rows)

    used = set(r['img_path'] for r in train_rows) | set(r['img_path'] for r in val_rows)

    def fresh(pool_name, family=None):
        rows = [r for r in val_pools[pool_name] if r['img_path'] not in used]
        if family:
            rows = [r for r in rows if r['family'] == family]
        return rows

    blue_real_proxy = []
    blue_real_proxy.extend(with_semantic(sample_source(fresh('ccpd2019', 'normal7'), 'ccpd2019', 600, 300), 'blue_real_primary'))
    blue_real_proxy.extend(with_semantic(sample_source(fresh('crpd_crpd_single', 'normal7'), 'crpd_crpd_single', 528, 301), 'blue_real_primary'))
    blue_real_proxy.extend(with_semantic(sample_source(fresh('crpd_crpd_double', 'normal7'), 'crpd_crpd_double', 214, 302), 'blue_real_primary'))
    blue_real_proxy.extend(with_semantic(sample_source(fresh('crpd_crpd_multi', 'normal7'), 'crpd_crpd_multi', 58, 303), 'blue_real_primary'))

    used |= set(r['img_path'] for r in blue_real_proxy)
    green_real_proxy = with_semantic(sample_source([r for r in fresh('ccpd2020', 'green8') if r['img_path'] not in used], 'ccpd2020', 300, 310), 'green_real_primary')
    used |= set(r['img_path'] for r in green_real_proxy)
    bridge_pool = [r for r in fresh('green_exact_quad', 'green8') if r['img_path'] not in used]
    if len(bridge_pool) < 500:
        bridge_pool = [r for r in val_pools['green_exact_quad_train'] if r['img_path'] not in used and r['img_path'] not in set(x['img_path'] for x in train_rows)]
    green_bridge_proxy = with_semantic(sample_source(bridge_pool, 'green_exact_quad', min(500, len(bridge_pool)), 320), 'green_support')
    used |= set(r['img_path'] for r in green_bridge_proxy)

    support_proxy = []
    support_proxy.extend(with_semantic(sample_source([r for r in fresh('cblprd_blue', 'normal7') if r['img_path'] not in used], 'cblprd_blue', 600, 330), 'blue_support'))
    support_proxy.extend(with_semantic(sample_source([r for r in fresh('cblprd_green', 'green8') if r['img_path'] not in used], 'cblprd_green', 700, 331), 'green_support'))
    support_proxy.extend(with_semantic(sample_source([r for r in fresh('green_edgefit_simple', 'green8') if r['img_path'] not in used], 'green_edgefit_simple', 500, 332), 'green_support'))
    rng.shuffle(blue_real_proxy)
    rng.shuffle(green_real_proxy)
    rng.shuffle(green_bridge_proxy)
    rng.shuffle(support_proxy)

    write_manifest(train_rows, OUT_DIR / 'train_stageA_v2.csv', 'train')
    write_manifest(val_rows, OUT_DIR / 'val_stageA_v2.csv', 'test')
    write_manifest(blue_real_proxy, OUT_DIR / 'proxy_blue_real_foundation.csv', 'test')
    write_manifest(green_real_proxy, OUT_DIR / 'proxy_green_real_foundation.csv', 'test')
    write_manifest(green_bridge_proxy, OUT_DIR / 'proxy_green_bridge.csv', 'test')
    write_manifest(support_proxy, OUT_DIR / 'proxy_support.csv', 'test')

    train_green = [r for r in train_rows if r['family'] == 'green8']
    train_green_real = [r for r in train_green if r['domain_role'] == 'real_primary']
    train_green_support = [r for r in train_green if r['domain_role'] == 'support']
    train_blue = [r for r in train_rows if r['family'] == 'normal7']
    train_blue_real = [r for r in train_blue if r['domain_role'] == 'real_primary']
    train_blue_support = [r for r in train_blue if r['domain_role'] == 'support']

    green_real_prov = green_province_counter(train_green_real)
    green_support_prov = green_province_counter(train_green_support)
    green_total_prov = green_province_counter(train_green)
    non_anhui_threshold = {prov: cnt >= GREEN_MIN_OTHER_PROV for prov, cnt in sorted(green_total_prov.items()) if prov != '皖'}

    summary = {
        'train_total': len(train_rows),
        'train_by_semantic_group': dict(count_by(train_rows, 'semantic_group')),
        'train_by_source': dict(count_by(train_rows, 'source')),
        'train_by_family': dict(count_by(train_rows, 'family')),
        'train_green_by_province': dict(sorted(green_total_prov.items())),
        'train_green_real_by_province': dict(sorted(green_real_prov.items())),
        'train_green_support_by_province': dict(sorted(green_support_prov.items())),
        'green_synthetic_to_real_ratio': ratio(len(train_green_support), len(train_green_real)),
        'blue_synthetic_to_real_ratio': ratio(len(train_blue_support), len(train_blue_real)),
        'green_ccpd2020_by_province': dict(sorted(green_real_prov.items())),
        'green_synthetic_added_by_province': dict(sorted(green_support_prov.items())),
        'green_final_anhui_ratio': ratio(green_total_prov.get('皖', 0), len(train_green)),
        'green_non_anhui_min_threshold': GREEN_MIN_OTHER_PROV,
        'green_non_anhui_threshold_met': non_anhui_threshold,
        'val_total': len(val_rows),
        'val_by_semantic_group': dict(count_by(val_rows, 'semantic_group')),
        'proxy_blue_real_foundation_source': dict(count_by(blue_real_proxy, 'source')),
        'proxy_green_real_foundation_source': dict(count_by(green_real_proxy, 'source')),
        'proxy_green_bridge_source': dict(count_by(green_bridge_proxy, 'source')),
        'proxy_support_source': dict(count_by(support_proxy, 'source')),
    }

    with open(OUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        for k, v in summary.items():
            f.write(f'{k}: {v}\n')

    print('Wrote manifests to', OUT_DIR)
    for k, v in summary.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
