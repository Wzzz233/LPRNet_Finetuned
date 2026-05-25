#!/usr/bin/env python3
"""Build StageB1A-C manifests.

C definition:
- Start from original StageB1A difficulty manifests.
- Replace only train source=green_edgefit_extreme rows (300 rows) with v4_e3 board_extreme_tail.
- Keep original validation/proxy files unchanged for the main/old benchmark.
- Also produce a separate new-proxy manifest dir with only proxy_green_edgefit_extreme replaced by non-overlapping v4_e3 rows.
- Preserve exact train extreme per-province quotas; fail hard on shortages.
"""
import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
SRC_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty'
OUT_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original'
NEW_PROXY_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy'
ORIG_TRAIN = SRC_DIR / 'train_B1A.csv'
ORIG_VAL = SRC_DIR / 'val_B1A.csv'
CAND_ROOT = ROOT / 'tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail'
SEED = 20260425
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
EXTRA_FIELDS = [
    'img_rel_path','plate_len','sub_type','is_real','need_tilt_aug','has_bbox',
    'ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc',
    'ocr_channel_order','ocr_min_occ_ratio','ocr_quad_pad_ratio'
]
REQUIRED_PROXIES = [
    'proxy_blue_ccpd2019_real.csv',
    'proxy_blue_crpd_real.csv',
    'proxy_green_ccpd2020_real.csv',
    'proxy_green_nonanhui_template_synth.csv',
    'proxy_green_bridge_exactquad.csv',
    'proxy_green_edgefit_hard.csv',
    'proxy_green_edgefit_extreme.csv',
    'proxy_support_cblprd.csv',
]


def read_rows(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        rd = csv.DictReader(f)
        return list(rd), list(rd.fieldnames or [])


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm(row, fields):
    return {k: row.get(k, '') for k in fields}


def infer_text(path):
    stem = Path(path).stem
    m = re.search(r'([\u4e00-\u9fa5][A-Z0-9]{6,7})$', stem)
    if m:
        return m.group(1)
    m = re.search(r'-([\u4e00-\u9fa5][A-Z0-9]{6,7})-', stem)
    if m:
        return m.group(1)
    return ''


def parse_quad(path):
    name = Path(path).name
    parts = name.split('-')
    # edgefit4-board_extreme_tail-bbox-quad-train-...
    if len(parts) < 4:
        return None
    vals = []
    for tok in parts[3].split('_'):
        try:
            x, y = tok.split('&')
            vals.extend([str(int(round(float(x)))), str(int(round(float(y))))])
        except Exception:
            return None
    return vals if len(vals) == 8 else None


def collect_candidates(root):
    by = defaultdict(list)
    total = 0
    skipped = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() not in IMG_EXT:
                continue
            p = str(Path(dp) / fn)
            total += 1
            txt = infer_text(p)
            q = parse_quad(p)
            if not txt or q is None:
                skipped.append(p)
                continue
            by[txt[0]].append((p, txt, q))
    for prov in by:
        by[prov].sort(key=lambda x: x[0])
    return by, total, skipped


def fill_v4e3_row(base, new_path, new_text, q, fields):
    rr = norm(base, fields)
    rr.update({
        'img_path': new_path,
        'img_rel_path': os.path.relpath(new_path, str(ROOT)),
        'text': new_text,
        'plate_len': str(len(new_text)),
        'family': 'green8',
        'sub_type': base.get('sub_type') or 'green8',
        'source': 'green_edgefit_extreme_v4e3_ccpdboard',
        'source_family': 'green_edgefit_extreme_v4e3_ccpdboard__green8',
        'is_real': '0',
        'need_tilt_aug': '0',
        'preprocess_group': 'ccpd_board',
        'has_bbox': '1',
        'has_quad': '1',
        'can_parse_ccpd_geom': '1',
        'can_perspective': '1',
        'bbox_source': 'ccpd_filename',
        'quad_source': 'ccpd_filename',
        'ocr_channel_order': 'bgr',
        'ocr_crop_mode': 'obb_warp',
        'ocr_resize_mode': 'letterbox',
        'ocr_resize_kernel': 'nn',
        'ocr_preproc': 'gray3',
        'ocr_min_occ_ratio': '0.0',
        'ocr_quad_pad_ratio': '0.0',
    })
    qkeys = ['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']
    for k, v in zip(qkeys, q):
        rr[k] = v
    return rr


def choose_by_quota(cand, quota, rng, used_paths):
    chosen = {}
    deficits = {}
    for prov, cnt in quota.items():
        arr = [x for x in cand.get(prov, []) if x[0] not in used_paths]
        rng.shuffle(arr)
        if len(arr) < cnt:
            deficits[prov] = {'need': cnt, 'have': len(arr), 'short': cnt - len(arr)}
        else:
            chosen[prov] = arr[:cnt]
            used_paths.update(x[0] for x in chosen[prov])
    if deficits:
        raise SystemExit(json.dumps({'fatal': 'candidate deficits', 'deficits': deficits}, ensure_ascii=False, indent=2))
    return chosen


def copy_all_proxies(src, dst, fields):
    dst.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_PROXIES:
        p = src / name
        if not p.exists():
            raise SystemExit(f'[FATAL] missing proxy {p}')
        rows, _ = read_rows(p)
        write_csv(dst / name, [norm(r, fields) for r in rows], fields)


def main():
    rng = random.Random(SEED)
    train, base_fields = read_rows(ORIG_TRAIN)
    val, _ = read_rows(ORIG_VAL)
    fields = base_fields + [f for f in EXTRA_FIELDS if f not in base_fields]
    cand, cand_total, skipped = collect_candidates(CAND_ROOT)

    train_extreme = [r for r in train if r.get('source') == 'green_edgefit_extreme']
    train_quota = Counter(r['text'][0] for r in train_extreme)
    used_paths = set()
    train_chosen = choose_by_quota(cand, train_quota, rng, used_paths)
    pools = {k: list(v) for k, v in train_chosen.items()}

    out_train = []
    train_mapping = []
    for r in train:
        if r.get('source') == 'green_edgefit_extreme':
            prov = r['text'][0]
            new_path, new_text, q = pools[prov].pop()
            out_row = fill_v4e3_row(r, new_path, new_text, q, fields)
            train_mapping.append({
                'province': prov,
                'old_text': r['text'],
                'new_text': new_text,
                'old_img_path': r['img_path'],
                'new_img_path': new_path,
                'split': 'train',
            })
            out_train.append(out_row)
        else:
            out_train.append(norm(r, fields))

    out_val = [norm(r, fields) for r in val]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / 'train_B1A_C_train_v4e3_ccpdboard_eval_original.csv', out_train, fields)
    write_csv(OUT_DIR / 'val_B1A_C_original_eval.csv', out_val, fields)
    copy_all_proxies(SRC_DIR, OUT_DIR, fields)

    with (OUT_DIR / 'extreme_train_swap_mapping.csv').open('w', encoding='utf-8', newline='') as f:
        mf = ['split','province','old_text','new_text','old_img_path','new_img_path']
        w = csv.DictWriter(f, fieldnames=mf)
        w.writeheader(); w.writerows(train_mapping)

    # Build new proxy dir: copy all old proxies, replace only extreme proxy with non-overlapping v4e3 ccpd_board rows.
    copy_all_proxies(SRC_DIR, NEW_PROXY_DIR, fields)
    old_proxy_rows, _ = read_rows(SRC_DIR / 'proxy_green_edgefit_extreme.csv')
    proxy_quota = Counter(r['text'][0] for r in old_proxy_rows)
    proxy_chosen = choose_by_quota(cand, proxy_quota, rng, used_paths)
    proxy_pools = {k: list(v) for k, v in proxy_chosen.items()}
    new_proxy_rows = []
    proxy_mapping = []
    for r in old_proxy_rows:
        prov = r['text'][0]
        new_path, new_text, q = proxy_pools[prov].pop()
        new_proxy_rows.append(fill_v4e3_row(r, new_path, new_text, q, fields))
        proxy_mapping.append({
            'split': 'proxy',
            'province': prov,
            'old_text': r['text'],
            'new_text': new_text,
            'old_img_path': r['img_path'],
            'new_img_path': new_path,
        })
    write_csv(NEW_PROXY_DIR / 'proxy_green_edgefit_extreme.csv', new_proxy_rows, fields)
    with (NEW_PROXY_DIR / 'extreme_proxy_swap_mapping.csv').open('w', encoding='utf-8', newline='') as f:
        mf = ['split','province','old_text','new_text','old_img_path','new_img_path']
        w = csv.DictWriter(f, fieldnames=mf)
        w.writeheader(); w.writerows(proxy_mapping)

    # Integrity checks.
    non_extreme_orig = [r for r in train if r.get('source') != 'green_edgefit_extreme']
    non_extreme_new = [r for r in out_train if r.get('source') != 'green_edgefit_extreme_v4e3_ccpdboard']
    non_extreme_same = len(non_extreme_orig) == len(non_extreme_new) and all(
        (a.get('img_path'), a.get('text'), a.get('source'), a.get('preprocess_group')) ==
        (b.get('img_path'), b.get('text'), b.get('source'), b.get('preprocess_group'))
        for a, b in zip(non_extreme_orig, non_extreme_new)
    )
    train_paths = {m['new_img_path'] for m in train_mapping}
    proxy_paths = {m['new_img_path'] for m in proxy_mapping}
    overlap = sorted(train_paths & proxy_paths)
    old_proxy_original, _ = read_rows(SRC_DIR / 'proxy_green_edgefit_extreme.csv')
    old_proxy_copied, _ = read_rows(OUT_DIR / 'proxy_green_edgefit_extreme.csv')
    old_proxy_same = [(r.get('img_path'), r.get('text')) for r in old_proxy_original] == [(r.get('img_path'), r.get('text')) for r in old_proxy_copied]

    summary = {
        'experiment': 'StageB1A-C_train_v4e3_ccpdboard_eval_original_plus_new_proxy',
        'seed': SEED,
        'candidate_root': str(CAND_ROOT),
        'candidate_total': cand_total,
        'candidate_skipped_unparseable': len(skipped),
        'train_manifest': str(OUT_DIR / 'train_B1A_C_train_v4e3_ccpdboard_eval_original.csv'),
        'val_manifest': str(OUT_DIR / 'val_B1A_C_original_eval.csv'),
        'old_proxy_manifest_dir': str(OUT_DIR),
        'new_proxy_manifest_dir': str(NEW_PROXY_DIR),
        'train_rows': len(out_train),
        'train_extreme_count': len(train_mapping),
        'train_extreme_by_province': dict(sorted(Counter(m['province'] for m in train_mapping).items())),
        'original_train_extreme_by_province': dict(sorted(train_quota.items())),
        'non_extreme_unchanged_by_key': non_extreme_same,
        'old_eval_proxy_unchanged_path_text': old_proxy_same,
        'new_proxy_extreme_count': len(proxy_mapping),
        'new_proxy_extreme_by_province': dict(sorted(Counter(m['province'] for m in proxy_mapping).items())),
        'original_proxy_extreme_by_province': dict(sorted(proxy_quota.items())),
        'train_new_proxy_path_overlap_count': len(overlap),
        'train_new_proxy_path_overlap_examples': overlap[:10],
        'definition_note': 'Train extreme is replaced with v4_e3 + ccpd_board; old proxy remains unchanged for main benchmark; new proxy dir replaces only proxy_green_edgefit_extreme for secondary difficulty evaluation.',
        'known_data_limits': 'User confirmed v4 e2/e3/e4 are medium-black-border medium-tilt extreme, but current samples may skew upward and plates are tight to black border with little background clutter.',
    }
    if not non_extreme_same:
        raise SystemExit(json.dumps({'fatal': 'non_extreme_changed'}, ensure_ascii=False, indent=2))
    if not old_proxy_same:
        raise SystemExit(json.dumps({'fatal': 'old_proxy_changed'}, ensure_ascii=False, indent=2))
    if overlap:
        raise SystemExit(json.dumps({'fatal': 'train_new_proxy_overlap', 'overlap': overlap[:10]}, ensure_ascii=False, indent=2))
    if len(train_mapping) != 300:
        raise SystemExit(json.dumps({'fatal': 'train_extreme_count', 'got': len(train_mapping)}, ensure_ascii=False, indent=2))
    (OUT_DIR / 'summary_C.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (NEW_PROXY_DIR / 'summary_C_new_proxy.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
