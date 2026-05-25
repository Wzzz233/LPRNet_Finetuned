#!/usr/bin/env python3
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
SRC_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty'
OUT_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_train_v4e3_ccpdboard_eval_original'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIG_TRAIN = SRC_DIR / 'train_B1A.csv'
ORIG_VAL = SRC_DIR / 'val_B1A.csv'
CAND_ROOT = ROOT / 'tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail'
SEED = 20260702
FIELD_EXTRA = ['img_rel_path','ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc','ocr_channel_order','ocr_min_occ_ratio']

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}

def read_csv(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f)), list(csv.DictReader(open(path, encoding='utf-8')).fieldnames or [])

def read_rows(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        rd = csv.DictReader(f)
        return list(rd), list(rd.fieldnames or [])

def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

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
    if len(parts) < 4:
        return None
    vals=[]
    for tok in parts[3].split('_'):
        try:
            x,y=tok.split('&')
            vals.extend([str(int(round(float(x)))), str(int(round(float(y))))])
        except Exception:
            return None
    return vals if len(vals)==8 else None

def collect_candidates(root):
    by=defaultdict(list)
    for dp,_,files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() not in IMG_EXT:
                continue
            p=str(Path(dp)/fn)
            txt=infer_text(p)
            q=parse_quad(p)
            if not txt or q is None:
                continue
            by[txt[0]].append((p,txt,q))
    for k in by:
        by[k].sort(key=lambda x:x[0])
    return by

def normalize(r, fields):
    return {k:r.get(k,'') for k in fields}

def main():
    rng=random.Random(SEED)
    train, base_fields = read_rows(ORIG_TRAIN)
    fields = base_fields + [f for f in FIELD_EXTRA if f not in base_fields]
    cand=collect_candidates(CAND_ROOT)
    need=Counter(r['text'][0] for r in train if r.get('source')=='green_edgefit_extreme')
    chosen={}
    deficits={}
    for prov,cnt in need.items():
        arr=cand.get(prov, [])[:]
        rng.shuffle(arr)
        if len(arr) < cnt:
            deficits[prov]={'need':cnt,'have':len(arr),'short':cnt-len(arr)}
        else:
            chosen[prov]=arr[:cnt]
    if deficits:
        raise SystemExit(json.dumps({'fatal':'candidate deficits','deficits':deficits}, ensure_ascii=False, indent=2))
    pools={k:list(v) for k,v in chosen.items()}
    out_train=[]; mapping=[]
    for r in train:
        rr=normalize(r, fields)
        if r.get('source')=='green_edgefit_extreme':
            prov=r['text'][0]
            new_path,new_text,q=pools[prov].pop()
            rr.update({
                'img_path': new_path,
                'img_rel_path': os.path.relpath(new_path, str(ROOT)),
                'text': new_text,
                'source': 'green_edgefit_extreme_v4e3_ccpdboard',
                'source_family': 'green_edgefit_extreme_v4e3_ccpdboard__green8',
                'preprocess_group': 'ccpd_board',
                'has_quad': '1',
                'can_parse_ccpd_geom': '1',
                'can_perspective': '1',
                'quad_source': 'ccpd_filename',
                'bbox_source': 'ccpd_filename',
                'ocr_crop_mode': 'obb_warp',
                'ocr_resize_mode': 'letterbox',
                'ocr_resize_kernel': 'nn',
                'ocr_preproc': 'gray3',
                'ocr_channel_order': 'bgr',
                'ocr_min_occ_ratio': '0.0',
                'ocr_quad_pad_ratio': '0.0',
            })
            rr['quad_1x'],rr['quad_1y'],rr['quad_2x'],rr['quad_2y'],rr['quad_3x'],rr['quad_3y'],rr['quad_4x'],rr['quad_4y']=q
            mapping.append({'province':prov,'old_text':r['text'],'new_text':new_text,'old_img_path':r['img_path'],'new_img_path':new_path})
        out_train.append(rr)
    # val/proxy全部保持原StageB1A，不替换评测目标；只补字段以兼容同目录评测。
    val, _ = read_rows(ORIG_VAL)
    out_val=[normalize(r, fields) for r in val]
    write_csv(OUT_DIR/'train_B1A_train_v4e3_ccpdboard_eval_original.csv', out_train, fields)
    write_csv(OUT_DIR/'val_B1A_original_eval.csv', out_val, fields)
    for p in SRC_DIR.glob('proxy_*.csv'):
        rows,_=read_rows(p)
        write_csv(OUT_DIR/p.name, [normalize(r, fields) for r in rows], fields)
    with (OUT_DIR/'extreme_train_swap_mapping.csv').open('w', encoding='utf-8', newline='') as f:
        mf=['province','old_text','new_text','old_img_path','new_img_path']
        w=csv.DictWriter(f, fieldnames=mf); w.writeheader(); w.writerows(mapping)
    summary={
        'experiment':'A_train_v4e3_ccpdboard_eval_original',
        'seed':SEED,
        'train_manifest':str(OUT_DIR/'train_B1A_train_v4e3_ccpdboard_eval_original.csv'),
        'val_manifest':str(OUT_DIR/'val_B1A_original_eval.csv'),
        'eval_manifest_dir':str(OUT_DIR),
        'train_rows':len(out_train),
        'train_non_extreme_unchanged': True,
        'train_extreme_count':len(mapping),
        'train_extreme_by_province':dict(Counter(m['province'] for m in mapping)),
        'eval_proxy_note':'All proxy_*.csv copied from original StageB1A; original proxy_green_edgefit_extreme is unchanged.',
    }
    (OUT_DIR/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__=='__main__':
    main()
