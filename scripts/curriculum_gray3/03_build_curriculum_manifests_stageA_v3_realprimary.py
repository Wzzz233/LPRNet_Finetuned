#!/usr/bin/env python3
"""Build StageA v3 real-primary manifests.

A0: real-primary Gray3 foundation. CCPD/CRPD define the basin; synthetic is light support.
A1: light synthetic-support injection after A0 passes graduation.
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/home/wzzz/LPRNet')
LABEL_DIR = BASE / 'labels/curriculum_gray3'
OUT_DIR = BASE / 'manifests/curriculum_gray3_stagea_v3_realprimary'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIAL_CHARS = set('警学挂领使字港澳')
RNG_SEED = 42

A0_TRAIN_CONFIG = {
    'blue_real_primary': {
        'ccpd2019': 30000,
        'crpd_crpd_single': 16000,
        'crpd_crpd_double': 6000,
        'crpd_crpd_multi': 2500,
    },
    'blue_support': {
        'cblprd_blue': 3000,
    },
    'green_real_primary': {
        'ccpd2020': 5769,
    },
    'green_support': {
        'green_exact_quad': 1800,
        'green_edgefit_simple': 1200,
        'cblprd_green': 1200,
    },
}

A1_TRAIN_CONFIG = {
    'blue_real_primary': {
        'ccpd2019': 25000,
        'crpd_crpd_single': 14000,
        'crpd_crpd_double': 5000,
        'crpd_crpd_multi': 2200,
    },
    'blue_support': {
        'cblprd_blue': 2500,
    },
    'green_real_primary': {
        'ccpd2020': 5769,
    },
    'green_support': {
        'green_exact_quad': 3500,
        'green_edgefit_simple': 2500,
        'cblprd_green': 2500,
    },
}

VAL_CONFIG = {
    'ccpd2019': 3000,
    'crpd_crpd_single': 1800,
    'crpd_crpd_double': 800,
    'crpd_crpd_multi': 300,
    'ccpd2020': 1001,
    'cblprd_blue': 800,
    'cblprd_green': 800,
    'green_exact_quad': 800,
    'green_edgefit_simple': 600,
}

PROXY_CONFIG = {
    'blue_ccpd2019_real': ('ccpd2019', 'normal7', 2000, 'blue_real_primary'),
    'blue_crpd_real': ('CRPD_COMBINED', 'normal7', 2000, 'blue_real_primary'),
    'green_ccpd2020_real': ('ccpd2020', 'green8', 1001, 'green_real_primary'),
    'green_nonanhui_template_synth': ('GREEN_SYNTH_NONANHUI', 'green8', 1500, 'green_support'),
    'green_bridge_exactquad': ('green_exact_quad', 'green8', 800, 'green_support'),
    'support_cblprd': ('CBLPRD_MIXED', None, 1200, 'support_mixed'),
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
    label_file = p.parent.parent / 'labels' / (p.stem + '.txt')
    if not label_file.exists():
        return None
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
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


def province(text):
    return text[0] if text else ''


def filter_supported(rows, family=None):
    allowed = set('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZIO')
    out = []
    for r in rows:
        txt = r.get('text', '')
        if family and r.get('family') != family:
            continue
        if not txt or any(c in SPECIAL_CHARS for c in txt):
            continue
        if any(c not in allowed for c in txt):
            continue
        out.append(r)
    return out


def balanced_sample(rows, target, seed, max_major=None, major='皖', prefer_non_major=False):
    rng = random.Random(seed)
    rows = list(rows)
    by = defaultdict(list)
    for r in rows:
        by[province(r['text'])].append(r)
    if max_major is not None and major in by and len(by[major]) > max_major:
        rng.shuffle(by[major])
        by[major] = by[major][:max_major]
    if prefer_non_major and major in by:
        major_rows = by.pop(major)
        pool = []
        for b in by.values():
            rng.shuffle(b); pool.extend(b)
        rng.shuffle(pool)
        if len(pool) >= target:
            return pool[:target]
        rng.shuffle(major_rows)
        pool.extend(major_rows[:target-len(pool)])
        rng.shuffle(pool)
        return pool[:target]
    rows2 = []
    for b in by.values():
        rng.shuffle(b); rows2.extend(b)
    if len(rows2) <= target:
        rng.shuffle(rows2)
        return rows2
    # province-balanced selection with leftovers
    by2 = defaultdict(list)
    for r in rows2:
        by2[province(r['text'])].append(r)
    provs = sorted(by2.keys())
    base = target // max(1, len(provs))
    rem = target % max(1, len(provs))
    selected, leftovers = [], []
    for i, prov in enumerate(provs):
        bucket = by2[prov]
        rng.shuffle(bucket)
        take = min(len(bucket), base + (1 if i < rem else 0))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])
    if len(selected) < target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:target-len(selected)])
    rng.shuffle(selected)
    return selected[:target]


def load_pools(split):
    suffix = 'train' if split == 'train' else 'val'
    pools = {}
    names = ['ccpd2019', 'ccpd2020', 'crpd_crpd_single', 'crpd_crpd_double', 'crpd_crpd_multi',
             'cblprd_blue', 'cblprd_green', 'green_exact_quad', 'green_edgefit_simple']
    for name in names:
        p = LABEL_DIR / f'{name}_{suffix}.csv'
        pools[name] = read_csv(p)
    pools['ccpd2019'] = filter_supported(pools['ccpd2019'], 'normal7')
    pools['ccpd2020'] = filter_supported(pools['ccpd2020'], 'green8')
    for k in ['crpd_crpd_single','crpd_crpd_double','crpd_crpd_multi']:
        pools[k] = filter_supported(pools[k], 'normal7')
    pools['cblprd_blue'] = filter_supported(pools['cblprd_blue'], 'normal7')
    pools['cblprd_green'] = filter_supported(pools['cblprd_green'], 'green8')
    pools['green_exact_quad'] = filter_supported(pools['green_exact_quad'], 'green8')
    pools['green_edgefit_simple'] = filter_supported(pools['green_edgefit_simple'], 'green8')
    return pools


def sample_source(rows, source_key, target, seed):
    if source_key == 'ccpd2019':
        return balanced_sample(rows, target, seed, max_major=int(target * 0.70), major='皖')
    if source_key == 'ccpd2020':
        return balanced_sample(rows, target, seed, max_major=None, major='皖')
    if source_key in ('green_exact_quad', 'green_edgefit_simple', 'cblprd_green'):
        return balanced_sample(rows, target, seed, prefer_non_major=True)
    return balanced_sample(rows, target, seed)


def build_train(config, pools, seed0):
    rows=[]; seed=seed0
    for sem, sources in config.items():
        for src, target in sources.items():
            sampled = sample_source(pools[src], src, target, seed)
            rows.extend(with_semantic(sampled, sem))
            seed += 1
    random.Random(seed0+999).shuffle(rows)
    return rows


def build_val(pools, seed0=900):
    rows=[]; seed=seed0
    mapping = [
        ('ccpd2019','blue_real_primary'), ('crpd_crpd_single','blue_real_primary'),
        ('crpd_crpd_double','blue_real_primary'), ('crpd_crpd_multi','blue_real_primary'),
        ('ccpd2020','green_real_primary'), ('cblprd_blue','blue_support'),
        ('cblprd_green','green_support'), ('green_exact_quad','green_support'),
        ('green_edgefit_simple','green_support'),
    ]
    for src, sem in mapping:
        sampled = sample_source(pools[src], src, VAL_CONFIG[src], seed)
        rows.extend(with_semantic(sampled, sem)); seed += 1
    random.Random(seed0+999).shuffle(rows)
    return rows


def write_manifest(rows, path, split):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, '') for k in FIELDNAMES}
            row['split'] = split
            w.writerow(row)


def build_proxies(val_pools, train_rows, val_rows):
    # Proxies must be disjoint from training rows. They may reuse validation rows because
    # they are evaluation-only views with different semantic grouping.
    used_train = {r['img_path'] for r in train_rows}
    seed=1300
    proxies={}
    def fresh(src):
        return [r for r in val_pools[src] if r['img_path'] not in used_train]
    # If val rows are insufficient for a non-primary synthetic proxy, use remaining train-pool rows not in train.
    train_pools = load_pools('train')
    def fresh_any(src):
        rows = fresh(src)
        if len(rows) < 10:
            rows += [r for r in train_pools[src] if r['img_path'] not in used_train]
        return [r for r in rows if r['img_path'] not in used_train]
    used_proxy = set()
    proxies['blue_ccpd2019_real'] = with_semantic(sample_source(fresh_any('ccpd2019'), 'ccpd2019', 2000, seed), 'blue_real_primary'); seed+=1
    used_proxy |= {r['img_path'] for r in proxies['blue_ccpd2019_real']}
    crpd_pool = []
    for src in ['crpd_crpd_single','crpd_crpd_double','crpd_crpd_multi']:
        crpd_pool.extend([r for r in fresh_any(src) if r['img_path'] not in used_proxy])
    proxies['blue_crpd_real'] = with_semantic(balanced_sample(crpd_pool, 2000, seed), 'blue_real_primary'); seed+=1
    used_proxy |= {r['img_path'] for r in proxies['blue_crpd_real']}
    proxies['green_ccpd2020_real'] = with_semantic(sample_source([r for r in fresh_any('ccpd2020') if r['img_path'] not in used_proxy], 'ccpd2020', 1001, seed), 'green_real_primary'); seed+=1
    used_proxy |= {r['img_path'] for r in proxies['green_ccpd2020_real']}
    synth_non = []
    for src in ['green_exact_quad','green_edgefit_simple','cblprd_green']:
        synth_non.extend([r for r in fresh_any(src) if r['img_path'] not in used_proxy and province(r['text']) != '皖'])
    proxies['green_nonanhui_template_synth'] = with_semantic(balanced_sample(synth_non, 1500, seed, prefer_non_major=True), 'green_support'); seed+=1
    used_proxy |= {r['img_path'] for r in proxies['green_nonanhui_template_synth']}
    proxies['green_bridge_exactquad'] = with_semantic(sample_source([r for r in fresh_any('green_exact_quad') if r['img_path'] not in used_proxy], 'green_exact_quad', 800, seed), 'green_support'); seed+=1
    used_proxy |= {r['img_path'] for r in proxies['green_bridge_exactquad']}
    cbl = []
    cbl.extend([r for r in fresh_any('cblprd_blue') if r['img_path'] not in used_proxy][:600])
    cbl.extend([r for r in fresh_any('cblprd_green') if r['img_path'] not in used_proxy][:600])
    proxies['support_cblprd'] = with_semantic(balanced_sample(cbl, 1200, seed), 'blue_support')
    for name in proxies:
        random.Random(seed+17).shuffle(proxies[name])
    return proxies


def counter(rows, key):
    return dict(Counter(r.get(key, '') for r in rows))


def prov_counter(rows, family=None):
    c=Counter()
    for r in rows:
        if family and r.get('family') != family:
            continue
        c[province(r.get('text',''))]+=1
    return dict(sorted(c.items()))


def summarize(name, rows, val_rows, proxies):
    green=[r for r in rows if r['family']=='green8']
    green_real=[r for r in green if r['domain_role']=='real_primary']
    green_sup=[r for r in green if r['domain_role']=='support']
    blue=[r for r in rows if r['family']=='normal7']
    blue_real=[r for r in blue if r['domain_role']=='real_primary']
    blue_sup=[r for r in blue if r['domain_role']=='support']
    return {
        'name': name,
        'train_total': len(rows),
        'train_by_family': counter(rows,'family'),
        'train_by_semantic_group': counter(rows,'semantic_group'),
        'train_by_source': counter(rows,'source'),
        'train_green_by_province': prov_counter(green),
        'train_green_real_by_province': prov_counter(green_real),
        'train_green_support_by_province': prov_counter(green_sup),
        'train_blue_by_province': prov_counter(blue),
        'green_support_to_real_ratio': 0 if not green_real else len(green_sup)/len(green_real),
        'blue_support_to_real_ratio': 0 if not blue_real else len(blue_sup)/len(blue_real),
        'green_final_anhui_ratio': 0 if not green else sum(1 for r in green if province(r['text'])=='皖')/len(green),
        'ccpd2020_train_count': counter(rows,'source').get('ccpd2020',0),
        'val_total': len(val_rows),
        'val_by_family': counter(val_rows,'family'),
        'proxy_counts': {k: {'n': len(v), 'by_family': counter(v,'family'), 'by_source': counter(v,'source')} for k,v in proxies.items()},
    }


def assert_no_overlap(a_name, a_rows, b_name, b_rows):
    inter = {r['img_path'] for r in a_rows} & {r['img_path'] for r in b_rows}
    if inter:
        raise RuntimeError(f'path leakage {a_name} vs {b_name}: {len(inter)} example={next(iter(inter))}')


def main():
    train_pools = load_pools('train')
    val_pools = load_pools('val')
    a0 = build_train(A0_TRAIN_CONFIG, train_pools, 100)
    a1 = build_train(A1_TRAIN_CONFIG, train_pools, 200)
    val = build_val(val_pools)
    proxies = build_proxies(val_pools, a0 + a1, val)
    for name, rows in [('train_A0',a0),('train_A1',a1),('val',val)]:
        if not rows:
            raise RuntimeError(f'{name} empty')
    assert_no_overlap('train_A0', a0, 'val', val)
    assert_no_overlap('train_A1', a1, 'val', val)
    for pname, prows in proxies.items():
        assert_no_overlap('train_A0', a0, pname, prows)
        assert_no_overlap('train_A1', a1, pname, prows)
    write_manifest(a0, OUT_DIR/'train_A0.csv', 'train')
    write_manifest(a1, OUT_DIR/'train_A1.csv', 'train')
    write_manifest(val, OUT_DIR/'val_A0.csv', 'test')
    write_manifest(val, OUT_DIR/'val_A1.csv', 'test')
    for name, rows in proxies.items():
        write_manifest(rows, OUT_DIR/f'proxy_{name}.csv', 'test')
    s0=summarize('A0', a0, val, proxies)
    s1=summarize('A1', a1, val, proxies)
    (OUT_DIR/'summary_A0.json').write_text(json.dumps(s0, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT_DIR/'summary_A1.json').write_text(json.dumps(s1, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(OUT_DIR/'summary.txt','w',encoding='utf-8') as f:
        for s in [s0,s1]:
            f.write(f"[{s['name']}]\n")
            for k,v in s.items():
                if k != 'name': f.write(f'{k}: {v}\n')
            f.write('\n')
    print('Wrote', OUT_DIR)
    print(json.dumps({'A0': s0, 'A1': s1}, ensure_ascii=False, indent=2)[:6000])

if __name__ == '__main__':
    main()
