#!/usr/bin/env python3
"""Build StageB difficulty-finetuning manifests from the audited A1D mother line.

StageB invariant:
- mother must be A1D iter_002000, not A1D Final/best/last.
- keep A1D real-primary foundation data unchanged as the base.
- add only green_edgefit_hard/extreme training data as the StageB variable.
- evaluation views are disjoint from train by image path; hard/extreme test are held out.
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
SRC_MANIFEST_DIR = ROOT / 'manifests/curriculum_gray3_stagea_v3_realprimary'
LABEL_DIR = ROOT / 'labels/curriculum_gray3'
OUT_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty'
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

PROVINCES = set('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新')
RNG_SEED = 20260425

# A1D train_A1B already contains 1500 green_edgefit_simple.  A conservative StageB train
# append is hard=900, extreme=300.  The resulting edgefit simple:hard:extreme exposure is
# 1500:900:300, i.e. difficulty is introduced without letting synthetic difficulty dominate
# the real-primary base.
TRAIN_HARD_TARGET = 900
TRAIN_EXTREME_TARGET = 300

PROXY_TARGETS = {
    'green_edgefit_hard': 310,
    'green_edgefit_extreme': 124,
}


def read_csv(path: Path):
    with path.open(encoding='utf-8') as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(rows, path: Path, split: str):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, '') for k in FIELDNAMES}
            out['split'] = split
            w.writerow(out)


def province(text):
    return text[0] if text else ''


def validate_label_row(r, source, split):
    txt = r.get('text', '')
    if r.get('family') != 'green8':
        raise RuntimeError(f'{source}/{split}: unexpected family {r.get("family")} for {r}')
    if r.get('source') != source:
        raise RuntimeError(f'{source}/{split}: unexpected source {r.get("source")} for {r}')
    if r.get('split') != split:
        raise RuntimeError(f'{source}/{split}: unexpected split {r.get("split")} for {r}')
    if len(txt) != 8 or province(txt) not in PROVINCES:
        raise RuntimeError(f'{source}/{split}: bad green8 text {txt}')
    if not Path(r.get('img_path','')).exists():
        raise RuntimeError(f'{source}/{split}: missing image {r.get("img_path")}')


def enrich_plain(r, semantic_group='green_support'):
    return {
        'img_path': r['img_path'],
        'text': r['text'],
        'family': 'green8',
        'source': r['source'],
        'split': r.get('split',''),
        'semantic_group': semantic_group,
        'domain_role': 'support',
        'source_family': f"{r['source']}__green8",
        'preprocess_group': 'plain_plate',
        'has_quad': '0',
        'can_parse_ccpd_geom': '0',
        'can_perspective': '0',
        'quad_source': 'none',
        'bbox_source': 'none',
        'quad_1x': '', 'quad_1y': '', 'quad_2x': '', 'quad_2y': '',
        'quad_3x': '', 'quad_3y': '', 'quad_4x': '', 'quad_4y': '',
        'ocr_quad_pad_ratio': '0.0',
    }


def balanced_sample(rows, target, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[province(r['text'])].append(r)
    selected, leftovers = [], []
    provs = sorted(by)
    base = target // len(provs)
    rem = target % len(provs)
    for i, p in enumerate(provs):
        bucket = list(by[p])
        rng.shuffle(bucket)
        take = min(len(bucket), base + (1 if i < rem else 0))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])
    if len(selected) < target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:target-len(selected)])
    rng.shuffle(selected)
    if len(selected) != target:
        raise RuntimeError(f'could not sample target={target}, got={len(selected)}')
    return selected


def rows_by_source(rows):
    return dict(Counter(r.get('source','') for r in rows))


def prov_counter(rows):
    return dict(sorted(Counter(province(r.get('text','')) for r in rows if r.get('family') == 'green8').items()))


def assert_no_path_overlap(a_name, a_rows, b_name, b_rows):
    inter = {r['img_path'] for r in a_rows} & {r['img_path'] for r in b_rows}
    if inter:
        raise RuntimeError(f'path leakage {a_name} vs {b_name}: {len(inter)} example={next(iter(inter))}')


def main():
    train_base = read_csv(SRC_MANIFEST_DIR / 'train_A1B.csv')
    val_base = read_csv(SRC_MANIFEST_DIR / 'val_A1B.csv')
    proxies = {
        'proxy_blue_ccpd2019_real': read_csv(SRC_MANIFEST_DIR / 'proxy_blue_ccpd2019_real.csv'),
        'proxy_blue_crpd_real': read_csv(SRC_MANIFEST_DIR / 'proxy_blue_crpd_real.csv'),
        'proxy_green_ccpd2020_real': read_csv(SRC_MANIFEST_DIR / 'proxy_green_ccpd2020_real.csv'),
        'proxy_green_nonanhui_template_synth': read_csv(SRC_MANIFEST_DIR / 'proxy_green_nonanhui_template_synth.csv'),
        'proxy_green_bridge_exactquad': read_csv(SRC_MANIFEST_DIR / 'proxy_green_bridge_exactquad.csv'),
        'proxy_support_cblprd': read_csv(SRC_MANIFEST_DIR / 'proxy_support_cblprd.csv'),
    }

    hard_train = read_csv(LABEL_DIR / 'green_edgefit_hard_train.csv')
    extreme_train = read_csv(LABEL_DIR / 'green_edgefit_extreme_train.csv')
    hard_test = read_csv(LABEL_DIR / 'green_edgefit_hard_test.csv')
    extreme_test = read_csv(LABEL_DIR / 'green_edgefit_extreme_test.csv')
    for rows, src, split in [
        (hard_train, 'green_edgefit_hard', 'train'),
        (extreme_train, 'green_edgefit_extreme', 'train'),
        (hard_test, 'green_edgefit_hard', 'test'),
        (extreme_test, 'green_edgefit_extreme', 'test'),
    ]:
        for r in rows:
            validate_label_row(r, src, split)

    hard_add = [enrich_plain(r) for r in balanced_sample(hard_train, TRAIN_HARD_TARGET, RNG_SEED + 1)]
    extreme_add = [enrich_plain(r) for r in balanced_sample(extreme_train, TRAIN_EXTREME_TARGET, RNG_SEED + 2)]
    rng = random.Random(RNG_SEED + 9)
    train_b1a = list(train_base) + hard_add + extreme_add
    rng.shuffle(train_b1a)

    hard_proxy = [enrich_plain(r) for r in balanced_sample(hard_test, PROXY_TARGETS['green_edgefit_hard'], RNG_SEED + 3)]
    extreme_proxy = [enrich_plain(r) for r in balanced_sample(extreme_test, PROXY_TARGETS['green_edgefit_extreme'], RNG_SEED + 4)]
    val_b1a = list(val_base) + hard_proxy + extreme_proxy
    rng.shuffle(val_b1a)

    # safety: preserve no image leakage from train into eval views.
    assert_no_path_overlap('train_B1A', train_b1a, 'val_B1A', val_b1a)
    for pname, prows in proxies.items():
        assert_no_path_overlap('train_B1A', train_b1a, pname, prows)
    assert_no_path_overlap('train_B1A', train_b1a, 'proxy_green_edgefit_hard', hard_proxy)
    assert_no_path_overlap('train_B1A', train_b1a, 'proxy_green_edgefit_extreme', extreme_proxy)

    write_csv(train_b1a, OUT_DIR / 'train_B1A.csv', 'train')
    write_csv(val_b1a, OUT_DIR / 'val_B1A.csv', 'test')
    for pname, prows in proxies.items():
        write_csv(prows, OUT_DIR / f'{pname}.csv', 'test')
    write_csv(hard_proxy, OUT_DIR / 'proxy_green_edgefit_hard.csv', 'test')
    write_csv(extreme_proxy, OUT_DIR / 'proxy_green_edgefit_extreme.csv', 'test')

    summary = {
        'name': 'StageB_v1_B1A_difficulty_conservative',
        'mother_required': '/home/wzzz/LPRNet/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth',
        'train_total': len(train_b1a),
        'train_added': {'green_edgefit_hard': len(hard_add), 'green_edgefit_extreme': len(extreme_add)},
        'train_by_family': dict(Counter(r.get('family','') for r in train_b1a)),
        'train_by_source': rows_by_source(train_b1a),
        'train_green_by_province': prov_counter(train_b1a),
        'val_total': len(val_b1a),
        'val_added_eval_only': {'green_edgefit_hard': len(hard_proxy), 'green_edgefit_extreme': len(extreme_proxy)},
        'val_by_family': dict(Counter(r.get('family','') for r in val_b1a)),
        'val_by_source': rows_by_source(val_b1a),
        'proxy_counts': {k: len(v) for k, v in proxies.items()} | {'proxy_green_edgefit_hard': len(hard_proxy), 'proxy_green_edgefit_extreme': len(extreme_proxy)},
        'source_note': 'green_edgefit_hard/extreme are synthetic/support difficulty data from labels/curriculum_gray3, plain_plate/gray3 path; not real or board-like evidence by themselves.',
    }
    (OUT_DIR / 'summary_B1A.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
