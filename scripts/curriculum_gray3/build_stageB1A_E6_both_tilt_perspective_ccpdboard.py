#!/usr/bin/env python3
"""Build StageB1A-E6 manifests from the generated axis-dominant perspective dataset."""
import csv, json, os, shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
SRC_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty'
DATA_ROOT = ROOT / 'tmp/green_extreme_stageB1A_E6_axis_dominant_perspective_20260427'
OUT_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_eval_original'
NEW_PROXY_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_E6_axis_dominant_perspective_new_proxy'
ORIG_TRAIN = SRC_DIR / 'train_B1A.csv'
ORIG_VAL = SRC_DIR / 'val_B1A.csv'
EXTRA_FIELDS = ['img_rel_path','plate_len','sub_type','is_real','need_tilt_aug','has_bbox','ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc','ocr_channel_order','ocr_min_occ_ratio','ocr_quad_pad_ratio','difficulty_tier','extreme_direction']
REQUIRED_PROXIES = ['proxy_blue_ccpd2019_real.csv','proxy_blue_crpd_real.csv','proxy_green_ccpd2020_real.csv','proxy_green_nonanhui_template_synth.csv','proxy_green_bridge_exactquad.csv','proxy_green_edgefit_hard.csv','proxy_green_edgefit_extreme.csv','proxy_support_cblprd.csv']
QKEYS = ['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']
NEW_SOURCE = 'green_edgefit_extreme_E6_axis_dominant_perspective_ccpdboard'


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


def load_meta():
    return json.loads((DATA_ROOT / 'generation_meta.json').read_text(encoding='utf-8'))['records']


def qvals(rec):
    vals = []
    for pt in rec['exact_quad']:
        vals += [str(int(round(float(pt[0])))), str(int(round(float(pt[1]))))]
    return vals


def pool_records(records, split):
    by = defaultdict(list)
    for r in records:
        if r['split'] == split:
            p = DATA_ROOT / r['file']
            if not p.exists():
                raise SystemExit(f'[FATAL] missing generated image {p}')
            by[r['province']].append(r)
    for k in by:
        by[k].sort(key=lambda r: (r['tier'], r['direction'], r['file']))
    return by


def fill_row(base, rec, fields):
    p = str(DATA_ROOT / rec['file'])
    rr = norm(base, fields)
    rr.update({
        'img_path': p,
        'img_rel_path': os.path.relpath(p, str(ROOT)),
        'text': rec['text'],
        'plate_len': str(len(rec['text'])),
        'family': 'green8',
        'sub_type': base.get('sub_type') or 'green8',
        'source': NEW_SOURCE,
        'source_family': f'{NEW_SOURCE}__green8',
        'semantic_group': 'green_support',
        'domain_role': 'support',
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
        'difficulty_tier': rec['tier'],
        'extreme_direction': rec['direction'],
    })
    for k, v in zip(QKEYS, qvals(rec)):
        rr[k] = v
    return rr


def copy_all_proxies(src, dst, fields):
    dst.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_PROXIES:
        p = src / name
        if not p.exists():
            raise SystemExit(f'[FATAL] missing proxy {p}')
        rows, _ = read_rows(p)
        write_csv(dst / name, [norm(r, fields) for r in rows], fields)


def pop_for_prov(pool, prov):
    arr = pool.get(prov, [])
    if not arr:
        raise SystemExit(f'[FATAL] generated candidate shortage for province {prov}')
    return arr.pop(0)


def validate_rows(rows, expected_count):
    bad = []
    for r in rows:
        if r.get('preprocess_group') != 'ccpd_board':
            bad.append(('preprocess_group', r.get('img_path')))
        for k in ['has_quad','can_parse_ccpd_geom','can_perspective']:
            if r.get(k) != '1':
                bad.append((k, r.get('img_path')))
        for k in QKEYS:
            if r.get(k, '') == '':
                bad.append((k, r.get('img_path')))
        for k, v in [('ocr_crop_mode','obb_warp'),('ocr_resize_mode','letterbox'),('ocr_resize_kernel','nn'),('ocr_preproc','gray3'),('ocr_channel_order','bgr'),('ocr_quad_pad_ratio','0.0')]:
            if r.get(k) != v:
                bad.append((k, r.get('img_path')))
        if not Path(r.get('img_path', '')).exists():
            bad.append(('missing_path', r.get('img_path')))
    if len(rows) != expected_count or bad:
        raise SystemExit(json.dumps({'fatal':'validate_rows','count':len(rows),'expected':expected_count,'bad':bad[:20]}, ensure_ascii=False, indent=2))


def main():
    train, base_fields = read_rows(ORIG_TRAIN)
    val, _ = read_rows(ORIG_VAL)
    fields = base_fields + [f for f in EXTRA_FIELDS if f not in base_fields]
    meta = load_meta()
    train_pool = pool_records(meta, 'train')
    proxy_pool = pool_records(meta, 'proxy')
    train_extreme = [r for r in train if r.get('source') == 'green_edgefit_extreme']
    orig_train_quota = Counter(r['text'][0] for r in train_extreme)

    out_train = []
    train_mapping = []
    new_train_rows = []
    for r in train:
        if r.get('source') == 'green_edgefit_extreme':
            rec = pop_for_prov(train_pool, r['text'][0])
            nr = fill_row(r, rec, fields)
            out_train.append(nr)
            new_train_rows.append(nr)
            train_mapping.append({'split':'train','province':r['text'][0],'tier':rec['tier'],'direction':rec['direction'],'old_text':r['text'],'new_text':rec['text'],'old_img_path':r['img_path'],'new_img_path':nr['img_path']})
        else:
            out_train.append(norm(r, fields))
    out_val = [norm(r, fields) for r in val]

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if NEW_PROXY_DIR.exists():
        shutil.rmtree(NEW_PROXY_DIR)
    OUT_DIR.mkdir(parents=True)
    NEW_PROXY_DIR.mkdir(parents=True)

    write_csv(OUT_DIR / 'train_B1A_E6_axis_dominant_perspective_eval_original.csv', out_train, fields)
    write_csv(OUT_DIR / 'val_B1A_E6_original_eval.csv', out_val, fields)
    copy_all_proxies(SRC_DIR, OUT_DIR, fields)
    with (OUT_DIR / 'extreme_train_swap_mapping.csv').open('w', encoding='utf-8', newline='') as f:
        mf = ['split','province','tier','direction','old_text','new_text','old_img_path','new_img_path']
        w = csv.DictWriter(f, fieldnames=mf)
        w.writeheader()
        w.writerows(train_mapping)

    copy_all_proxies(SRC_DIR, NEW_PROXY_DIR, fields)
    old_proxy, _ = read_rows(SRC_DIR / 'proxy_green_edgefit_extreme.csv')
    proxy_mapping = []
    new_proxy_rows = []
    for r in old_proxy:
        rec = pop_for_prov(proxy_pool, r['text'][0])
        nr = fill_row(r, rec, fields)
        new_proxy_rows.append(nr)
        proxy_mapping.append({'split':'proxy','province':r['text'][0],'tier':rec['tier'],'direction':rec['direction'],'old_text':r['text'],'new_text':rec['text'],'old_img_path':r['img_path'],'new_img_path':nr['img_path']})
    write_csv(NEW_PROXY_DIR / 'proxy_green_edgefit_extreme.csv', new_proxy_rows, fields)
    with (NEW_PROXY_DIR / 'extreme_proxy_swap_mapping.csv').open('w', encoding='utf-8', newline='') as f:
        mf = ['split','province','tier','direction','old_text','new_text','old_img_path','new_img_path']
        w = csv.DictWriter(f, fieldnames=mf)
        w.writeheader()
        w.writerows(proxy_mapping)

    non_extreme_orig = [r for r in train if r.get('source') != 'green_edgefit_extreme']
    non_extreme_new = [r for r in out_train if r.get('source') != NEW_SOURCE]
    non_extreme_same = len(non_extreme_orig) == len(non_extreme_new) and all((a.get('img_path'), a.get('text'), a.get('source'), a.get('preprocess_group')) == (b.get('img_path'), b.get('text'), b.get('source'), b.get('preprocess_group')) for a, b in zip(non_extreme_orig, non_extreme_new))
    old_proxy_original, _ = read_rows(SRC_DIR / 'proxy_green_edgefit_extreme.csv')
    old_proxy_copied, _ = read_rows(OUT_DIR / 'proxy_green_edgefit_extreme.csv')
    old_proxy_same = [(r.get('img_path'), r.get('text')) for r in old_proxy_original] == [(r.get('img_path'), r.get('text')) for r in old_proxy_copied]
    train_paths = {m['new_img_path'] for m in train_mapping}
    proxy_paths = {m['new_img_path'] for m in proxy_mapping}
    overlap = sorted(train_paths & proxy_paths)

    validate_rows(new_train_rows, 300)
    validate_rows(new_proxy_rows, 124)
    summary = {
        'experiment':'StageB1A-E6_axis_dominant_perspective_eval_original_plus_new_proxy',
        'definition_note':'Replace original StageB1A train green_edgefit_extreme rows with E6 axis-dominant perspective ccpd_board rows. Old proxy remains unchanged for main benchmark; separate new proxy replaces only extreme proxy for secondary evaluation.',
        'dataset_root':str(DATA_ROOT),
        'train_manifest':str(OUT_DIR / 'train_B1A_E6_axis_dominant_perspective_eval_original.csv'),
        'val_manifest':str(OUT_DIR / 'val_B1A_E6_original_eval.csv'),
        'old_proxy_manifest_dir':str(OUT_DIR),
        'new_proxy_manifest_dir':str(NEW_PROXY_DIR),
        'train_rows':len(out_train),
        'train_extreme_count':len(train_mapping),
        'train_extreme_by_province':dict(sorted(Counter(m['province'] for m in train_mapping).items())),
        'original_train_extreme_by_province':dict(sorted(orig_train_quota.items())),
        'train_extreme_by_tier':dict(sorted(Counter(m['tier'] for m in train_mapping).items())),
        'train_extreme_by_direction':dict(sorted(Counter(m['direction'] for m in train_mapping).items())),
        'non_extreme_unchanged_by_key':non_extreme_same,
        'old_eval_proxy_unchanged_path_text':old_proxy_same,
        'new_proxy_extreme_count':len(proxy_mapping),
        'new_proxy_extreme_by_province':dict(sorted(Counter(m['province'] for m in proxy_mapping).items())),
        'new_proxy_extreme_by_tier':dict(sorted(Counter(m['tier'] for m in proxy_mapping).items())),
        'new_proxy_extreme_by_direction':dict(sorted(Counter(m['direction'] for m in proxy_mapping).items())),
        'train_new_proxy_path_overlap_count':len(overlap),
        'train_new_proxy_path_overlap_examples':overlap[:10],
        'preprocess_assertion':'all E6 train/proxy replacement rows have ccpd_board + full quad + obb_warp/letterbox/nn/gray3/bgr/quad_pad=0.0',
    }
    if not non_extreme_same or not old_proxy_same or overlap:
        raise SystemExit(json.dumps({'fatal':'integrity','non_extreme_same':non_extreme_same,'old_proxy_same':old_proxy_same,'overlap':overlap[:10]}, ensure_ascii=False, indent=2))
    (OUT_DIR / 'summary_E6.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (NEW_PROXY_DIR / 'summary_E6_new_proxy.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
