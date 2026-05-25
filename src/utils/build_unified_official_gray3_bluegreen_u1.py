#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path('/home/wzzz/LPRNet')
SRC_UTILS = ROOT / 'src' / 'utils'
if str(SRC_UTILS) not in sys.path:
    sys.path.insert(0, str(SRC_UTILS))

from prepare_ccpd_splits import decode_ccpd_plate as decode_ccpd2019_plate  # type: ignore
from prepare_ccpd_green_labels import decode_green_plate_from_stem as decode_ccpd2020_plate  # type: ignore

DATA_ROOTS = {
    'ccpd2019': ROOT / 'datasets' / 'CCPD2019',
    'ccpd2020': ROOT / 'datasets' / 'CCPD2020' / 'ccpd_green',
    'cblprd': ROOT / 'datasets' / 'CBLPRD-330k_v1',
    'crpd': ROOT / 'datasets' / 'CRPD_all',
}

DEFAULT_E2_MANIFEST = ROOT / 'manifests' / 'Archive' / 'unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv'
DEFAULT_CBLPRD_MANIFEST = ROOT / 'manifests' / 'cblprd_cv_geom_manifest.csv'
DEFAULT_CRPD_MANIFEST = ROOT / 'manifests' / 'crpd_all_raw_board_v1_supported.csv'
DEFAULT_OUT = ROOT / 'manifests' / 'unified_manifest_official_gray3_bluegreen_u1c.csv'
DEFAULT_SUMMARY = ROOT / 'manifests' / 'unified_manifest_official_gray3_bluegreen_u1c.summary.json'
DEFAULT_REPORT = ROOT / 'reports' / 'GREEN_U1C_MANIFEST_REPORT.md'
DEFAULT_EXTRA_EXTREME_MANIFESTS = [
    ROOT / 'manifests' / 'unified_manifest_green_edgefit_v4_e4_extreme_append10_20260412_train.csv',
]
DEFAULT_EXTRA_EXTREME_TREES = [
    {
        'root_dir': ROOT / 'green_edgefit_tier3_full_v2',
        'source_tag': 'u1_tier3_v3_pool',
        'dataset_name': 'green_edgefit_tier3_full_v2',
        'source_name': 'synthetic_edgefit_tier3_v3',
        'bucket_to_difficulty': {'simple': 'simple', 'hard': 'hard', 'extreme': 'extreme'},
        'split_allowlist': ['train'],
    },
    {
        'root_dir': ROOT / 'green_edgefit_v4_boardlike_a3000',
        'source_tag': 'u1_a3000_boardlike_pool',
        'dataset_name': 'green_edgefit_v4_boardlike_a3000',
        'source_name': 'v4_boardlike_edgefit_a3000',
        'bucket_to_difficulty': {
            'geometry_clean': 'simple',
            'board_mid_occ': 'hard',
            'board_low_occ': 'hard',
            'board_extreme_tail': 'extreme',
        },
        'split_allowlist': ['train'],
    },
]

MANIFEST_FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type', 'source',
    'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'bbox_source', 'quad_source', 'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio', 'source_root_tag', 'difficulty_bucket',
    'sample_weight', 'weight_family_scale', 'weight_province_scale', 'weight_difficulty_scale'
]


def iter_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def normalize_row(row, source_tag, difficulty='real'):
    out = {k: row.get(k, '') for k in MANIFEST_FIELDS}
    out.update(dict(row))
    out['source_root_tag'] = source_tag
    out['ocr_channel_order'] = 'bgr'
    out['ocr_crop_mode'] = 'obb_warp'
    out['ocr_resize_mode'] = 'letterbox'
    out['ocr_resize_kernel'] = 'nn'
    out['ocr_preproc'] = 'gray3'
    out['ocr_min_occ_ratio'] = '0.9'
    out['ocr_quad_pad_ratio'] = '0.0'
    out['difficulty_bucket'] = difficulty
    if not out.get('plate_len') and out.get('text'):
        out['plate_len'] = str(len(out['text']))
    return out


def make_row(img_path, rel_path, split, text, family, sub_type, source, root_tag, difficulty='real'):
    return {
        'img_path': str(img_path),
        'img_rel_path': rel_path,
        'dataset_name': root_tag,
        'split': split,
        'text': text,
        'plate_len': str(len(text)),
        'family': family,
        'sub_type': sub_type,
        'source': source,
        'is_real': '1',
        'need_tilt_aug': '0',
        'preprocess_group': root_tag,
        'has_bbox': '0',
        'has_quad': '0',
        'can_parse_ccpd_geom': '1',
        'can_perspective': '1',
        'bbox_source': 'filename',
        'quad_source': 'filename',
        'ocr_channel_order': 'bgr',
        'ocr_crop_mode': 'obb_warp',
        'ocr_resize_mode': 'letterbox',
        'ocr_resize_kernel': 'nn',
        'ocr_preproc': 'gray3',
        'ocr_min_occ_ratio': '0.9',
        'ocr_quad_pad_ratio': '0.0',
        'source_root_tag': root_tag,
        'difficulty_bucket': difficulty,
        'sample_weight': '1.000000',
        'weight_family_scale': '1.000000',
        'weight_province_scale': '1.000000',
        'weight_difficulty_scale': '1.000000',
    }


def ingest_ccpd2019(root):
    rows = []
    split_dir = root / 'splits'
    for split_name in ['train', 'val', 'test']:
        fp = split_dir / f'{split_name}.txt'
        if not fp.exists():
            continue
        for line in fp.read_text(encoding='utf-8').splitlines():
            rel = line.strip()
            if not rel:
                continue
            img = root / rel
            if not img.exists():
                continue
            try:
                text = decode_ccpd2019_plate(rel)
            except Exception:
                continue
            rows.append(make_row(img, rel, split_name, text, 'normal7', 'blue', 'ccpd2019_real', 'ccpd2019', difficulty='real'))
    return rows


def ingest_ccpd2020_green(root):
    rows = []
    for split_name in ['train', 'val', 'test']:
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        for img in sorted(split_dir.glob('*.jpg')):
            try:
                text = decode_ccpd2020_plate(img.stem)
            except Exception:
                continue
            rows.append(make_row(img, str(img.relative_to(root)), split_name, text, 'green8', 'green', 'ccpd2020_green_real', 'ccpd2020', difficulty='real'))
    return rows


def infer_e2_difficulty(row):
    src = row.get('source', '') or ''
    comment = row.get('comment', '') or ''
    dataset_name = row.get('dataset_name', '') or ''
    blob = ' '.join([src, comment, dataset_name]).lower()
    if 'extreme' in blob:
        return 'extreme'
    if 'hard' in blob or 'tier3' in blob or 'edgefit' in blob:
        return 'hard'
    if 'synthetic' in blob or 'exact_quad' in blob or 'boardlike' in blob:
        return 'simple'
    return 'real'


def add_counter(row, counters):
    text = row.get('text', '')
    counters['family'][row.get('family', '')] += 1
    counters['split'][row.get('split', '')] += 1
    counters['source'][row.get('source', '')] += 1
    counters['root_tag'][row.get('source_root_tag', '')] += 1
    counters['difficulty'][row.get('difficulty_bucket', '')] += 1
    if text:
        counters['province'][text[0]] += 1


def text_key(row):
    return (row.get('split', ''), row.get('family', ''), row.get('text', ''))


def filter_train_image_leak(src_rows, seen_eval_image, raw_counts, root_tag):
    out = []
    dropped = 0
    for r in src_rows:
        if r.get('split') == 'train' and r.get('img_path', '') in seen_eval_image:
            dropped += 1
            continue
        out.append(r)
    raw_counts[f'{root_tag}_dropped_image_leak'] = dropped
    return out


def filter_blue_text_leak(src_rows, eval_blue_texts, raw_counts, root_tag):
    out = []
    dropped = 0
    for r in src_rows:
        if r.get('split') == 'train' and r.get('family') != 'green8' and text_key(r) in eval_blue_texts:
            dropped += 1
            continue
        out.append(r)
    raw_counts[f'{root_tag}_dropped_text_leak_blue'] = dropped
    return out


def source_cap(src_rows, root_tag, cap, raw_counts):
    out = []
    seen = 0
    for r in src_rows:
        if r.get('split') != 'train':
            out.append(r)
            continue
        if seen >= cap:
            continue
        seen += 1
        out.append(r)
    raw_counts[f'{root_tag}_train_kept_after_source_cap'] = seen
    return out


def province_cap(src_rows, root_tag, cap, raw_counts):
    out = []
    prov_seen = Counter()
    for r in src_rows:
        if r.get('split') != 'train':
            out.append(r)
            continue
        text = r.get('text', '')
        if not text:
            continue
        prov = text[0]
        if prov_seen[prov] >= cap:
            continue
        prov_seen[prov] += 1
        out.append(r)
    raw_counts[f'{root_tag}_province_cap_unique'] = len(prov_seen)
    raw_counts[f'{root_tag}_province_train_counts'] = dict(sorted(prov_seen.items()))
    return out


def cap_cblprd_by_family(rows, fam_caps):
    out = []
    fam_seen = Counter()
    for r in rows:
        if r.get('split') != 'train':
            out.append(r)
            continue
        fam = r.get('family')
        cap = fam_caps.get(fam)
        if cap is None:
            out.append(r)
            continue
        if fam_seen[fam] >= cap:
            continue
        fam_seen[fam] += 1
        out.append(r)
    return out, dict(fam_seen)


def rebalance_e2_train(rows, weights):
    train_real = [r for r in rows if r.get('split') == 'train' and infer_e2_difficulty(r) == 'real']
    train_simple = [r for r in rows if r.get('split') == 'train' and infer_e2_difficulty(r) == 'simple']
    train_hard = [r for r in rows if r.get('split') == 'train' and infer_e2_difficulty(r) == 'hard']
    train_extreme = [r for r in rows if r.get('split') == 'train' and infer_e2_difficulty(r) == 'extreme']
    non_train = [r for r in rows if r.get('split') != 'train']

    pools = {'simple': train_simple, 'hard': train_hard, 'extreme': train_extreme}
    ratios = {'simple': weights[0], 'hard': weights[1], 'extreme': weights[2]}
    unit = None
    for key, pool in pools.items():
        ratio = ratios[key]
        if ratio <= 0 or not pool:
            continue
        candidate = len(pool) // ratio
        unit = candidate if unit is None else min(unit, candidate)
    unit = unit or 0

    kept = []
    kept.extend(train_real)
    kept.extend(train_simple[:unit * ratios['simple']])
    kept.extend(train_hard[:unit * ratios['hard']])
    kept.extend(train_extreme[:unit * ratios['extreme']])
    kept.extend(non_train)
    return kept, {
        'before_train': {
            'real': len(train_real),
            'simple': len(train_simple),
            'hard': len(train_hard),
            'extreme': len(train_extreme),
        },
        'after_train': {
            'real': len(train_real),
            'simple': min(len(train_simple), unit * ratios['simple']),
            'hard': min(len(train_hard), unit * ratios['hard']),
            'extreme': min(len(train_extreme), unit * ratios['extreme']),
        }
    }


def per_split_family(rows):
    out = {}
    for split in ['train', 'val', 'test']:
        c = Counter(r.get('family', '') for r in rows if r.get('split') == split)
        out[split] = dict(sorted(c.items()))
    return out


def subset_rows(rows, **conds):
    out = rows
    for k, v in conds.items():
        out = [r for r in out if r.get(k) == v]
    return out


def counts_by(rows, key_fn):
    c = Counter()
    for r in rows:
        key = key_fn(r)
        if key:
            c[key] += 1
    return dict(c.most_common())


def attach_extra_manifest(base_rows, manifest_path, source_tag, forced_difficulty=None):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f'missing extra manifest: {manifest_path}')
    out = list(base_rows)
    for row in iter_rows(manifest_path):
        difficulty = forced_difficulty if forced_difficulty is not None else infer_e2_difficulty(row)
        out.append(normalize_row(row, source_tag, difficulty=difficulty))
    return out


def extract_plate_text_from_path(path):
    stem = Path(path).stem
    provinces = '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'
    match = __import__('re').search(rf'([{provinces}][A-Z][A-Z0-9]{{5,6}})', stem)
    if not match:
        return ''
    return match.group(1)


def attach_extra_tree_rows(base_rows, root_dir, source_tag, dataset_name, source_name, bucket_to_difficulty, split_allowlist=None):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f'missing extra tree root: {root_dir}')
    out = list(base_rows)
    split_allowlist = set(split_allowlist or [])
    image_suffixes = {'.jpg', '.jpeg', '.png', '.ppm'}
    images_root = root_dir / 'images'
    if not images_root.exists():
        return out
    for split_dir in sorted(images_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if split_allowlist and split not in split_allowlist:
            continue
        for bucket_dir in sorted(split_dir.iterdir()):
            if not bucket_dir.is_dir():
                continue
            bucket = bucket_dir.name
            difficulty = bucket_to_difficulty.get(bucket)
            if difficulty is None:
                continue
            for img in sorted(bucket_dir.rglob('*')):
                if not img.is_file() or img.suffix.lower() not in image_suffixes:
                    continue
                text = extract_plate_text_from_path(img)
                if len(text) != 8:
                    continue
                out.append({
                    'img_path': str(img),
                    'img_rel_path': str(img.relative_to(root_dir)),
                    'dataset_name': dataset_name,
                    'split': split,
                    'text': text,
                    'plate_len': str(len(text)),
                    'family': 'green8',
                    'sub_type': 'green',
                    'source': source_name,
                    'is_real': '0',
                    'need_tilt_aug': '0',
                    'preprocess_group': 'ccpd_board',
                    'has_bbox': '1',
                    'has_quad': '1',
                    'can_parse_ccpd_geom': '1',
                    'can_perspective': '1',
                    'bbox_source': 'filename',
                    'quad_source': 'filename',
                    'ocr_channel_order': 'bgr',
                    'ocr_crop_mode': 'obb_warp',
                    'ocr_resize_mode': 'letterbox',
                    'ocr_resize_kernel': 'nn',
                    'ocr_preproc': 'gray3',
                    'ocr_min_occ_ratio': '0.9',
                    'ocr_quad_pad_ratio': '0.0',
                    'source_root_tag': source_tag,
                    'difficulty_bucket': difficulty,
                    'sample_weight': '1.000000',
                    'weight_family_scale': '1.000000',
                    'weight_province_scale': '1.000000',
                    'weight_difficulty_scale': '1.000000',
                })
    return out


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def apply_balanced_sample_weights(rows, province_power=1.0, family_power=0.5, difficulty_boosts=None):
    difficulty_boosts = difficulty_boosts or {'extreme': 3.0, 'hard': 1.5, 'simple': 1.0, 'real': 1.0}
    train_rows = [r for r in rows if r.get('split') == 'train' and r.get('text')]
    family_counts = Counter(r.get('family', '') for r in train_rows)
    family_province_counts = Counter((r.get('family', ''), (r.get('text', '') or '?')[:1]) for r in train_rows)
    max_family = max(family_counts.values()) if family_counts else 1
    max_family_prov = {}
    for family in family_counts:
        vals = [cnt for (fam, _), cnt in family_province_counts.items() if fam == family]
        max_family_prov[family] = max(vals) if vals else 1

    out = []
    for row in rows:
        nr = dict(row)
        fam = nr.get('family', '')
        text = nr.get('text', '') or ''
        prov = text[:1] if text else ''
        if nr.get('split') == 'train' and text and fam in family_counts:
            fam_scale = (max_family / max(1, family_counts[fam])) ** family_power
            prov_scale = (max_family_prov[fam] / max(1, family_province_counts[(fam, prov)])) ** province_power
            diff_scale = float(difficulty_boosts.get(nr.get('difficulty_bucket', 'real'), 1.0))
            weight = fam_scale * prov_scale * diff_scale
        else:
            fam_scale = 1.0
            prov_scale = 1.0
            diff_scale = 1.0
            weight = 1.0
        nr['weight_family_scale'] = f'{fam_scale:.6f}'
        nr['weight_province_scale'] = f'{prov_scale:.6f}'
        nr['weight_difficulty_scale'] = f'{diff_scale:.6f}'
        nr['sample_weight'] = f'{weight:.6f}'
        out.append(nr)
    return out


def summarize_weight_stats(rows):
    train_rows = [r for r in rows if r.get('split') == 'train' and r.get('text')]
    family_stats = {}
    family_prov = {}
    by_family = Counter(r.get('family', '') for r in train_rows)
    for family in sorted(by_family):
        family_weights = [float(r.get('sample_weight', 1.0)) for r in train_rows if r.get('family') == family]
        family_stats[family] = {
            'count': len(family_weights),
            'mean_weight': _mean(family_weights),
            'min_weight': min(family_weights) if family_weights else 0.0,
            'max_weight': max(family_weights) if family_weights else 0.0,
        }
        province_map = {}
        provinces = sorted({(r.get('text', '') or '?')[:1] for r in train_rows if r.get('family') == family and r.get('text')})
        for prov in provinces:
            weights = [float(r.get('sample_weight', 1.0)) for r in train_rows if r.get('family') == family and (r.get('text', '') or '?')[:1] == prov]
            province_map[prov] = {
                'count': len(weights),
                'mean_weight': _mean(weights),
                'min_weight': min(weights) if weights else 0.0,
                'max_weight': max(weights) if weights else 0.0,
            }
        family_prov[family] = province_map
    return {
        'family_weight_stats': family_stats,
        'province_weight_stats': family_prov,
    }


def main():
    ap = argparse.ArgumentParser(description='Build training-ready unified blue+green official-gray3 manifest from audited roots.')
    ap.add_argument('--e2-manifest', default=str(DEFAULT_E2_MANIFEST))
    ap.add_argument('--cblprd-manifest', default=str(DEFAULT_CBLPRD_MANIFEST))
    ap.add_argument('--crpd-manifest', default=str(DEFAULT_CRPD_MANIFEST))
    ap.add_argument('--out-manifest', default=str(DEFAULT_OUT))
    ap.add_argument('--out-summary', default=str(DEFAULT_SUMMARY))
    ap.add_argument('--out-report', default=str(DEFAULT_REPORT))
    ap.add_argument('--province-train-cap', type=int, default=12000)
    ap.add_argument('--source-train-cap', type=int, default=80000)
    ap.add_argument('--cblprd-green-cap', type=int, default=6000)
    ap.add_argument('--cblprd-blue-cap', type=int, default=12000)
    ap.add_argument('--cblprd-special-cap', type=int, default=4000)
    ap.add_argument('--e2-simple-weight', type=int, default=7)
    ap.add_argument('--e2-hard-weight', type=int, default=2)
    ap.add_argument('--e2-extreme-weight', type=int, default=1)
    ap.add_argument('--extra-extreme-manifests', nargs='*', default=[str(p) for p in DEFAULT_EXTRA_EXTREME_MANIFESTS])
    ap.add_argument('--extra-extreme-tree-roots', nargs='*', default=[str(item['root_dir']) for item in DEFAULT_EXTRA_EXTREME_TREES])
    ap.add_argument('--weight-province-power', type=float, default=1.0)
    ap.add_argument('--weight-family-power', type=float, default=0.5)
    ap.add_argument('--weight-simple-boost', type=float, default=1.0)
    ap.add_argument('--weight-hard-boost', type=float, default=1.5)
    ap.add_argument('--weight-extreme-boost', type=float, default=3.0)
    ap.add_argument('--weight-real-boost', type=float, default=1.0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    for key, root in DATA_ROOTS.items():
        if not root.exists():
            raise FileNotFoundError(f'missing data root: {key} -> {root}')

    e2_manifest = Path(args.e2_manifest)
    cblprd_manifest = Path(args.cblprd_manifest)
    crpd_manifest = Path(args.crpd_manifest)
    for p in [e2_manifest, cblprd_manifest, crpd_manifest]:
        if not p.exists():
            raise FileNotFoundError(f'missing manifest: {p}')

    raw_counts = Counter()
    seen_eval_image = set()

    ccpd2019_rows = ingest_ccpd2019(DATA_ROOTS['ccpd2019'])
    ccpd2020_rows = ingest_ccpd2020_green(DATA_ROOTS['ccpd2020'])
    raw_counts['ccpd2019_rows'] = len(ccpd2019_rows)
    raw_counts['ccpd2020_rows'] = len(ccpd2020_rows)

    e2_rows = []
    for r in iter_rows(e2_manifest):
        nr = normalize_row(r, 'e2_green', difficulty=infer_e2_difficulty(r))
        if nr.get('split') in {'val', 'test'}:
            seen_eval_image.add(nr.get('img_path', ''))
        e2_rows.append(nr)
    raw_counts['e2_rows'] = len(e2_rows)

    cblprd_rows = [normalize_row(r, 'cblprd', difficulty='real') for r in iter_rows(cblprd_manifest)]
    crpd_rows = [normalize_row(r, 'crpd', difficulty='real') for r in iter_rows(crpd_manifest)]
    raw_counts['cblprd_rows'] = len(cblprd_rows)
    raw_counts['crpd_rows'] = len(crpd_rows)

    ccpd2019_rows = filter_train_image_leak(ccpd2019_rows, seen_eval_image, raw_counts, 'ccpd2019')
    ccpd2020_rows = filter_train_image_leak(ccpd2020_rows, seen_eval_image, raw_counts, 'ccpd2020')
    cblprd_rows = filter_train_image_leak(cblprd_rows, seen_eval_image, raw_counts, 'cblprd')
    crpd_rows = filter_train_image_leak(crpd_rows, seen_eval_image, raw_counts, 'crpd')

    eval_blue_texts = set()
    for pool in [ccpd2019_rows, cblprd_rows, crpd_rows]:
        for r in pool:
            if r.get('split') in {'val', 'test'} and r.get('family') != 'green8':
                eval_blue_texts.add(text_key(r))

    ccpd2019_rows = filter_blue_text_leak(ccpd2019_rows, eval_blue_texts, raw_counts, 'ccpd2019')
    cblprd_rows = filter_blue_text_leak(cblprd_rows, eval_blue_texts, raw_counts, 'cblprd')
    crpd_rows = filter_blue_text_leak(crpd_rows, eval_blue_texts, raw_counts, 'crpd')

    cblprd_rows, cblprd_kept = cap_cblprd_by_family(
        cblprd_rows,
        {'green8': args.cblprd_green_cap, 'normal7': args.cblprd_blue_cap, 'special': args.cblprd_special_cap},
    )
    raw_counts['cblprd_train_kept_by_family'] = cblprd_kept

    ccpd2019_rows = source_cap(ccpd2019_rows, 'ccpd2019', args.source_train_cap, raw_counts)
    ccpd2020_rows = source_cap(ccpd2020_rows, 'ccpd2020', args.source_train_cap, raw_counts)
    crpd_rows = source_cap(crpd_rows, 'crpd', args.source_train_cap, raw_counts)
    cblprd_rows = source_cap(cblprd_rows, 'cblprd', args.source_train_cap, raw_counts)

    ccpd2019_rows = province_cap(ccpd2019_rows, 'ccpd2019', args.province_train_cap, raw_counts)
    ccpd2020_rows = province_cap(ccpd2020_rows, 'ccpd2020', args.province_train_cap, raw_counts)
    crpd_rows = province_cap(crpd_rows, 'crpd', args.province_train_cap, raw_counts)
    cblprd_rows = province_cap(cblprd_rows, 'cblprd', args.province_train_cap, raw_counts)

    e2_rows, e2_rebalanced = rebalance_e2_train(
        e2_rows,
        (args.e2_simple_weight, args.e2_hard_weight, args.e2_extreme_weight),
    )
    for idx, extra_path in enumerate(args.extra_extreme_manifests):
        e2_rows = attach_extra_manifest(e2_rows, extra_path, f'u1_extra_extreme_manifest_{idx+1}', forced_difficulty='extreme')
    attached_tree_summaries = []
    configured_tree_roots = {str(Path(p).resolve()) for p in args.extra_extreme_tree_roots}
    for tree_cfg in DEFAULT_EXTRA_EXTREME_TREES:
        tree_root = Path(tree_cfg['root_dir'])
        if str(tree_root.resolve()) not in configured_tree_roots:
            continue
        before_count = len(e2_rows)
        e2_rows = attach_extra_tree_rows(
            e2_rows,
            tree_root,
            source_tag=tree_cfg['source_tag'],
            dataset_name=tree_cfg['dataset_name'],
            source_name=tree_cfg['source_name'],
            bucket_to_difficulty=tree_cfg['bucket_to_difficulty'],
            split_allowlist=tree_cfg['split_allowlist'],
        )
        attached_tree_summaries.append({
            'root_dir': str(tree_root),
            'source_tag': tree_cfg['source_tag'],
            'added_rows': len(e2_rows) - before_count,
        })
    raw_counts['extra_extreme_manifest_count'] = len(args.extra_extreme_manifests)
    raw_counts['extra_extreme_tree_count'] = len(attached_tree_summaries)

    all_rows = []
    counters = {'family': Counter(), 'split': Counter(), 'source': Counter(), 'province': Counter(), 'root_tag': Counter(), 'difficulty': Counter()}
    for pool in [e2_rows, ccpd2019_rows, ccpd2020_rows, cblprd_rows, crpd_rows]:
        for row in pool:
            add_counter(row, counters)
            all_rows.append(row)

    all_rows = apply_balanced_sample_weights(
        all_rows,
        province_power=args.weight_province_power,
        family_power=args.weight_family_power,
        difficulty_boosts={
            'simple': args.weight_simple_boost,
            'hard': args.weight_hard_boost,
            'extreme': args.weight_extreme_boost,
            'real': args.weight_real_boost,
        },
    )

    fieldnames = sorted({k for row in all_rows for k in row.keys()} | set(MANIFEST_FIELDS))

    split_family = per_split_family(all_rows)
    train_rows = [r for r in all_rows if r.get('split') == 'train']
    train_blue = [r for r in train_rows if r.get('family') == 'normal7']
    train_green = [r for r in train_rows if r.get('family') == 'green8']
    train_special = [r for r in train_rows if r.get('family') == 'special']
    synth_train = [r for r in train_rows if r.get('source_root_tag') == 'e2_green']
    weight_stats = summarize_weight_stats(all_rows)

    summary = {
        'data_roots': {k: str(v) for k, v in DATA_ROOTS.items()},
        'inputs': {
            'e2_manifest': str(e2_manifest),
            'cblprd_manifest': str(cblprd_manifest),
            'crpd_manifest': str(crpd_manifest),
            'extra_extreme_manifests': args.extra_extreme_manifests,
            'extra_extreme_tree_roots': args.extra_extreme_tree_roots,
        },
        'row_count': len(all_rows),
        'raw_counts': dict(raw_counts),
        'family': dict(counters['family']),
        'split': dict(counters['split']),
        'source': dict(counters['source']),
        'root_tag': dict(counters['root_tag']),
        'difficulty': dict(counters['difficulty']),
        'province_top30': counters['province'].most_common(30),
        'split_family_distribution': split_family,
        'train_family_distribution': {
            'normal7': len(train_blue),
            'green8': len(train_green),
            'special': len(train_special),
        },
        'train_province_distribution_top30': Counter(r['text'][0] for r in train_rows if r.get('text')).most_common(30),
        'train_blue_province_top30': Counter(r['text'][0] for r in train_blue if r.get('text')).most_common(30),
        'train_green_province_top30': Counter(r['text'][0] for r in train_green if r.get('text')).most_common(30),
        'train_synthetic_difficulty_distribution': dict(Counter(r.get('difficulty_bucket', '') for r in synth_train)),
        'train_synthetic_source_distribution': dict(Counter(r.get('source', '') for r in synth_train)),
        'sample_weight_policy': {
            'goal': '不删除样本，只用温和降权实现 family 内各省尽量均衡，并抬高 hard/extreme 权重',
            'province_power': args.weight_province_power,
            'family_power': args.weight_family_power,
            'difficulty_boosts': {
                'simple': args.weight_simple_boost,
                'hard': args.weight_hard_boost,
                'extreme': args.weight_extreme_boost,
                'real': args.weight_real_boost,
            },
        },
        'sample_weight_stats': weight_stats,
        'leak_policy': {
            'blue_text_leak_forbidden': True,
            'green_text_leak_allowed': True,
            'image_level_leak_forbidden_all_real_sources': True,
            'green_reason': '绿牌数据过少，可允许牌面文本重复，但仍禁止图像级泄露。',
        },
        'difficulty_policy': {
            'target_ratio_simple_hard_extreme': [args.e2_simple_weight, args.e2_hard_weight, args.e2_extreme_weight],
            'observed_e2_rebalance': e2_rebalanced,
            'extra_tree_attach_summary': attached_tree_summaries,
            'note': '生成集难度按 simple:hard:extreme=7:2:1 收敛；real 不参与该比例限制，作为真实锚点保留。extra_extreme_manifests 与 extra_extreme_tree_roots 会额外直接并入。',
        },
        'eval_protocol': {
            'primary': 'family-aware 作为常规离线主口径',
            'board_related': '涉及板端/board-native 问题时必须额外对齐 greedy CTC 口径',
            'green': 'green8 重点看 family-aware exact_plate_acc / first_char_acc / province_macro，并区分 real 与 synthetic buckets',
            'blue': 'normal7 看 family-aware exact_plate_acc / first_char_acc / province_macro',
            'compare_axis': '按 family、province、source_root_tag、difficulty_bucket、sample_weight 分层汇报',
        },
        'board_alignment': {
            'ocr_channel_order': 'bgr',
            'ocr_crop_mode': 'obb_warp',
            'ocr_resize_mode': 'letterbox',
            'ocr_resize_kernel': 'nn',
            'ocr_preproc': 'gray3',
            'ocr_min_occ_ratio': 0.9,
            'ocr_quad_pad_ratio': 0.0,
        },
        'note': 'U1 已重建并接入温和 sample_weight 平衡；蓝绿牌都不删除样本，只通过 family 内省份降权与 difficulty boost 抑制皖过强，并补回 extreme。',
    }

    if not args.dry_run:
        out_manifest = Path(args.out_manifest)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        with out_manifest.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        out_summary = Path(args.out_summary)
        out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        report = Path(args.out_report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            '# GREEN_U1_MANIFEST_REBUILD_REPORT\n\n'
            f'- manifest: {out_manifest}\n'
            f'- summary: {out_summary}\n\n'
            '## Train family distribution\n\n'
            + '\n'.join(f'- {k}: {v}' for k, v in summary['train_family_distribution'].items())
            + '\n\n## Train province top30\n\n'
            + '\n'.join(f'- {k}: {v}' for k, v in summary['train_province_distribution_top30'])
            + '\n\n## Train synthetic difficulty distribution\n\n'
            + '\n'.join(f'- {k}: {v}' for k, v in summary['train_synthetic_difficulty_distribution'].items())
            + '\n\n## Sample weight policy\n\n'
            + f"- province_power: {args.weight_province_power}\n"
            + f"- family_power: {args.weight_family_power}\n"
            + f"- boosts: simple={args.weight_simple_boost}, hard={args.weight_hard_boost}, extreme={args.weight_extreme_boost}, real={args.weight_real_boost}\n"
            + '\n',
            encoding='utf-8'
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
