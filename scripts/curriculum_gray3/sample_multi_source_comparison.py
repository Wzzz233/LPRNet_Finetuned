#!/usr/bin/env python3
"""从各数据源抽图，走训练端正确处理链路，输出94x24 gray3对比图"""
import csv, sys
from pathlib import Path
import cv2, numpy as np

ROOT=Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT/'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

OUT=ROOT/'tmp/multi_source_board_gray3_comparison_20260428'
OUT.mkdir(parents=True, exist_ok=True)
ZOOM=8
QKEYS=['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']

def get_quad(row):
    vals=[]; 
    for k in QKEYS:
        v=row.get(k,'')
        if v=='': return None
        vals.append(float(v))
    return np.array(vals,dtype=np.float32).reshape(4,2)

def parse_ccpd_quad(name):
    parts=Path(name).stem.split('-')
    if len(parts)<4: return None
    try:
        pts=[]; 
        for t in parts[3].split('_'):
            x,y=t.split('&'); pts.append((float(x),float(y)))
        return np.array(pts,dtype=np.float32) if len(pts)==4 else None
    except: return None

# Use E6A manifest as it contains all sources
MANIFEST=ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/train_B1A_E6A_single_axis_visible_eval_original.csv'
rows=list(csv.DictReader(MANIFEST.open('r',encoding='utf-8-sig')))

# Define selections
sources={
    'ccpd2019_blue': {'filter': lambda r: r.get('source')=='ccpd2019', 'n': 4, 'label': 'CCPD2019 blue'},
    'ccpd2020_green': {'filter': lambda r: r.get('source')=='ccpd2020', 'n': 4, 'label': 'CCPD2020 green'},
    'E6_generated': {'filter': lambda r: r.get('source','').startswith('green_edgefit_extreme_E6'), 'n': 6, 'label': 'E6 generated extreme'},
    'cblprd': {'filter': lambda r: r.get('source')=='cblprd', 'n': 4, 'label': 'CBLPRD (plain)'},
    'crpd': {'filter': lambda r: r.get('source','').startswith('crpd_'), 'n': 4, 'label': 'CRPD blue'},
}

for src_key, cfg in sources.items():
    subset=[r for r in rows if cfg['filter'](r)]
    subset=sorted(subset, key=lambda r: (r.get('difficulty_tier',''), r['text']))[:cfg['n']]
    print(f'{src_key}: picked {len(subset)} from {sum(1 for r in rows if cfg["filter"](r))} total')

    for idx,r in enumerate(subset):
        img=cv2.imread(r['img_path'], cv2.IMREAD_COLOR)
        if img is None:
            print(f'  SKIP missing: {r["img_path"]}')
            continue

        preproc=r.get('preprocess_group','')
        is_plain=(preproc=='plain_plate' or src_key=='cblprd')

        if is_plain:
            # CBLPRD / plain: no perspective, just resize 94x24 then gray3
            warped=img  # no warp
            prep=cv2.resize(img, (94,24), interpolation=cv2.INTER_NEAREST)
            prep=prep.astype('float32')
            # apply gray3 manually
            gray=cv2.cvtColor(prep, cv2.COLOR_BGR2GRAY)
            prep=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            occ=-1.0
            has_q=False
        else:
            q=get_quad(r)
            if q is None:
                q=parse_ccpd_quad(r['img_path'])
            if q is None:
                print(f'  SKIP no quad: {r["img_path"]}')
                continue
            has_q=True
            prep,occ,warped,_,_ = prepare_board_ocr_input_from_quad_bgr888(img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)

        # raw + quad preview
        preview=img.copy()
        if has_q:
            cv2.polylines(preview, [q.astype(np.int32).reshape((-1,1,2))], True, (0,255,0), 2)

        # zoom
        gray_big=cv2.resize(prep, (94*ZOOM, 24*ZOOM), interpolation=cv2.INTER_NEAREST)
        raw_resized=cv2.resize(preview, (gray_big.shape[1], gray_big.shape[0]), interpolation=cv2.INTER_LINEAR)
        collage=np.concatenate([raw_resized, gray_big], axis=0)

        txt=r['text']
        label=f"{cfg['label']} | {txt} | {preproc}"
        cv2.putText(collage, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2, cv2.LINE_AA)

        fname=f"{src_key}_{idx:02d}_{txt}.png"
        safe=''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in fname)
        cv2.imwrite(str(OUT/safe), collage)

print(f'\nall saved to {OUT}')
print(f'files: {len(list(OUT.glob("*.png")))}')
