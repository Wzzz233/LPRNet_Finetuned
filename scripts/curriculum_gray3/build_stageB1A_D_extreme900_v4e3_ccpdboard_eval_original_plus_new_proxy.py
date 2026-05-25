#!/usr/bin/env python3
"""Build StageB1A-D manifests.

D definition:
- Start from original StageB1A difficulty manifests.
- Remove original train source=green_edgefit_extreme rows (300 tier3/plain rows).
- Add 900 v4_e3 board_extreme_tail rows with ccpd_board/quad fields, using original per-province quota * 3.
- Keep non-extreme train rows unchanged.
- Keep original validation/proxy files unchanged for old benchmark.
- Also produce a new-proxy manifest dir with only proxy_green_edgefit_extreme replaced by non-overlapping v4_e3 rows.
- Fail hard on province quota shortage or train/proxy image overlap.
"""
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
SRC_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty'
OUT_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original'
NEW_PROXY_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy'
ORIG_TRAIN = SRC_DIR / 'train_B1A.csv'
ORIG_VAL = SRC_DIR / 'val_B1A.csv'
CAND_ROOT = ROOT / 'tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail'
SEED = 20260426
EXTREME_MULTIPLIER = 3
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
        w.writeheader(); w.writerows(rows)


def norm(row, fields):
    return {k: row.get(k, '') for k in fields}


def infer_text(path):
    stem = Path(path).stem
    m = re.search(r'([\u4e00-\u9fa5][A-Z0-9]{6,7})$', stem)
    if m:
        return m.group(1)
    m = re.search(r'-([\u4e00-\u9fa5][A-Z0-9]{6,7})-', stem)
    return m.group(1) if m else ''


def parse_quad(path):
    parts = Path(path).name.split('-')
    if len(parts) < 4:
        return None
    vals=[]
    try:
        for tok in parts[3].split('_'):
            x,y = tok.split('&')
            vals.extend([str(int(round(float(x)))), str(int(round(float(y))))])
    except Exception:
        return None
    return vals if len(vals) == 8 else None


def collect_candidates(root):
    by=defaultdict(list); total=0; skipped=[]
    for dp,_,files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() not in IMG_EXT:
                continue
            p=str(Path(dp)/fn); total += 1
            txt=infer_text(p); q=parse_quad(p)
            if not txt or q is None:
                skipped.append(p); continue
            by[txt[0]].append((p, txt, q))
    for k in by:
        by[k].sort(key=lambda x:x[0])
    return by,total,skipped


def fill_v4_row(template, new_path, new_text, q, fields):
    rr = norm(template, fields)
    rr.update({
        'img_path': new_path,
        'img_rel_path': os.path.relpath(new_path, str(ROOT)),
        'text': new_text,
        'plate_len': str(len(new_text)),
        'family': 'green8',
        'sub_type': template.get('sub_type') or 'green8',
        'source': 'green_edgefit_extreme900_v4e3_ccpdboard',
        'source_family': 'green_edgefit_extreme900_v4e3_ccpdboard__green8',
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
    qkeys=['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']
    for k,v in zip(qkeys,q): rr[k]=v
    return rr


def choose_by_quota(cand, quota, rng, used_paths):
    chosen={}; deficits={}
    for prov,cnt in quota.items():
        arr=[x for x in cand.get(prov, []) if x[0] not in used_paths]
        rng.shuffle(arr)
        if len(arr) < cnt:
            deficits[prov]={'need':cnt,'have':len(arr),'short':cnt-len(arr)}
        else:
            chosen[prov]=arr[:cnt]
            used_paths.update(x[0] for x in chosen[prov])
    if deficits:
        raise SystemExit(json.dumps({'fatal':'candidate deficits','deficits':deficits}, ensure_ascii=False, indent=2))
    return chosen


def copy_all_proxies(src, dst, fields):
    dst.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_PROXIES:
        p=src/name
        if not p.exists():
            raise SystemExit(f'[FATAL] missing proxy {p}')
        rows,_=read_rows(p)
        write_csv(dst/name, [norm(r, fields) for r in rows], fields)


def direction_from_q(qvals):
    import numpy as np
    q=np.array([float(x) for x in qvals], dtype=float).reshape(4,2)
    left=(q[0]+q[3])/2; right=(q[1]+q[2])/2; top=(q[0]+q[1])/2; bot=(q[3]+q[2])/2
    rise=float(right[1]-left[1]); skew=float(bot[0]-top[0])
    return ('rise_up' if rise < -1 else ('rise_down' if rise > 1 else 'rise_flat')), ('skew_left' if skew < -1 else ('skew_right' if skew > 1 else 'skew_flat'))


def main():
    rng=random.Random(SEED)
    train, base_fields = read_rows(ORIG_TRAIN)
    val, _ = read_rows(ORIG_VAL)
    fields = base_fields + [f for f in EXTRA_FIELDS if f not in base_fields]
    cand,total,skipped = collect_candidates(CAND_ROOT)

    orig_extreme=[r for r in train if r.get('source')=='green_edgefit_extreme']
    orig_quota=Counter(r['text'][0] for r in orig_extreme)
    train_quota=Counter({k:v*EXTREME_MULTIPLIER for k,v in orig_quota.items()})
    used=set()
    chosen=choose_by_quota(cand, train_quota, rng, used)

    # Non-extreme rows unchanged; append new 900 extreme rows in deterministic province/template order.
    out_train=[norm(r, fields) for r in train if r.get('source')!='green_edgefit_extreme']
    template_by_prov=defaultdict(list)
    for r in orig_extreme:
        template_by_prov[r['text'][0]].append(r)
    mapping=[]; dirs=Counter()
    for prov in sorted(train_quota):
        templates=template_by_prov[prov]
        arr=list(chosen[prov])
        for i,(new_path,new_text,q) in enumerate(arr):
            tmpl=templates[i % len(templates)]
            out_train.append(fill_v4_row(tmpl, new_path, new_text, q, fields))
            ru,sk=direction_from_q(q); dirs.update([ru,sk])
            mapping.append({'split':'train','province':prov,'old_text':tmpl['text'],'new_text':new_text,'old_img_path':tmpl['img_path'],'new_img_path':new_path,'sample_idx':i})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR/'train_B1A_D_extreme900_v4e3_ccpdboard_eval_original.csv', out_train, fields)
    write_csv(OUT_DIR/'val_B1A_D_original_eval.csv', [norm(r, fields) for r in val], fields)
    copy_all_proxies(SRC_DIR, OUT_DIR, fields)
    with (OUT_DIR/'extreme_train900_swap_mapping.csv').open('w',encoding='utf-8',newline='') as f:
        mf=['split','province','old_text','new_text','old_img_path','new_img_path','sample_idx']
        w=csv.DictWriter(f, fieldnames=mf); w.writeheader(); w.writerows(mapping)

    # New proxy, non-overlap.
    copy_all_proxies(SRC_DIR, NEW_PROXY_DIR, fields)
    old_proxy,_=read_rows(SRC_DIR/'proxy_green_edgefit_extreme.csv')
    proxy_quota=Counter(r['text'][0] for r in old_proxy)
    proxy_chosen=choose_by_quota(cand, proxy_quota, rng, used)
    proxy_rows=[]; proxy_map=[]; proxy_dirs=Counter()
    for r in old_proxy:
        prov=r['text'][0]
        new_path,new_text,q=proxy_chosen[prov].pop()
        proxy_rows.append(fill_v4_row(r, new_path, new_text, q, fields))
        ru,sk=direction_from_q(q); proxy_dirs.update([ru,sk])
        proxy_map.append({'split':'proxy','province':prov,'old_text':r['text'],'new_text':new_text,'old_img_path':r['img_path'],'new_img_path':new_path})
    write_csv(NEW_PROXY_DIR/'proxy_green_edgefit_extreme.csv', proxy_rows, fields)
    with (NEW_PROXY_DIR/'extreme_proxy_swap_mapping.csv').open('w',encoding='utf-8',newline='') as f:
        mf=['split','province','old_text','new_text','old_img_path','new_img_path']
        w=csv.DictWriter(f, fieldnames=mf); w.writeheader(); w.writerows(proxy_map)

    # Checks.
    non_extreme_orig=[r for r in train if r.get('source')!='green_edgefit_extreme']
    non_extreme_new=[r for r in out_train if r.get('source')!='green_edgefit_extreme900_v4e3_ccpdboard']
    non_extreme_same=len(non_extreme_orig)==len(non_extreme_new) and all(
        (a.get('img_path'),a.get('text'),a.get('source'),a.get('preprocess_group')) == (b.get('img_path'),b.get('text'),b.get('source'),b.get('preprocess_group'))
        for a,b in zip(non_extreme_orig, non_extreme_new)
    )
    old_proxy_orig,_=read_rows(SRC_DIR/'proxy_green_edgefit_extreme.csv')
    old_proxy_copied,_=read_rows(OUT_DIR/'proxy_green_edgefit_extreme.csv')
    old_proxy_same=[(r.get('img_path'),r.get('text')) for r in old_proxy_orig] == [(r.get('img_path'),r.get('text')) for r in old_proxy_copied]
    train_paths={m['new_img_path'] for m in mapping}; proxy_paths={m['new_img_path'] for m in proxy_map}
    overlap=sorted(train_paths & proxy_paths)
    summary={
        'experiment':'StageB1A-D_extreme900_v4e3_ccpdboard_eval_original_plus_new_proxy',
        'seed':SEED,
        'extreme_multiplier':EXTREME_MULTIPLIER,
        'candidate_root':str(CAND_ROOT),
        'candidate_total':total,
        'candidate_skipped_unparseable':len(skipped),
        'train_manifest':str(OUT_DIR/'train_B1A_D_extreme900_v4e3_ccpdboard_eval_original.csv'),
        'val_manifest':str(OUT_DIR/'val_B1A_D_original_eval.csv'),
        'old_proxy_manifest_dir':str(OUT_DIR),
        'new_proxy_manifest_dir':str(NEW_PROXY_DIR),
        'original_train_rows':len(train),
        'train_rows':len(out_train),
        'train_row_delta':len(out_train)-len(train),
        'original_train_extreme_count':len(orig_extreme),
        'train_extreme_count':len(mapping),
        'original_train_extreme_by_province':dict(sorted(orig_quota.items())),
        'train_extreme_by_province':dict(sorted(Counter(m['province'] for m in mapping).items())),
        'non_extreme_unchanged_by_key':non_extreme_same,
        'old_eval_proxy_unchanged_path_text':old_proxy_same,
        'new_proxy_extreme_count':len(proxy_map),
        'new_proxy_extreme_by_province':dict(sorted(Counter(m['province'] for m in proxy_map).items())),
        'original_proxy_extreme_by_province':dict(sorted(proxy_quota.items())),
        'train_new_proxy_path_overlap_count':len(overlap),
        'train_new_proxy_path_overlap_examples':overlap[:10],
        'direction_stats_train':dict(dirs),
        'direction_stats_new_proxy':dict(proxy_dirs),
        'definition_note':'D increases train extreme from 300 to 900 using v4_e3 + ccpd_board, preserving original per-province ratios x3; old proxy unchanged; new proxy non-overlap secondary eval.',
        'known_data_limits':'v4_e3 is accepted as medium-tilt extreme, but still has plate tight to black border and limited background clutter.',
    }
    if not non_extreme_same: raise SystemExit(json.dumps({'fatal':'non_extreme_changed'}, ensure_ascii=False, indent=2))
    if not old_proxy_same: raise SystemExit(json.dumps({'fatal':'old_proxy_changed'}, ensure_ascii=False, indent=2))
    if overlap: raise SystemExit(json.dumps({'fatal':'train_new_proxy_overlap','overlap':overlap[:10]}, ensure_ascii=False, indent=2))
    if len(mapping)!=900: raise SystemExit(json.dumps({'fatal':'train_extreme_count','got':len(mapping)}, ensure_ascii=False, indent=2))
    (OUT_DIR/'summary_D.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (NEW_PROXY_DIR/'summary_D_new_proxy.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__=='__main__':
    main()
