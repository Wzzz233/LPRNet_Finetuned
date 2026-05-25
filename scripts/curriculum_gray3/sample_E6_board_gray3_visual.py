#!/usr/bin/env python3
"""抽 E6A/E6B extreme 图走 ccpd_board 链路，输出 94x24 gray3 供肉眼检查。
复用项目已有 get_quad / prepare_board_ocr_input_from_quad_bgr888，不自己拼坐标。"""
import csv, sys
from pathlib import Path
import cv2, numpy as np

ROOT=Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT/'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

OUT=ROOT/'tmp/E6_board_gray3_visual_samples_20260428'
OUT.mkdir(parents=True, exist_ok=True)
ZOOM=8
QKEYS=['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']

def get_quad(row):
    vals=[]
    for k in QKEYS:
        v=row.get(k,'')
        if v=='': return None
        vals.append(float(v))
    return np.array(vals,dtype=np.float32).reshape(4,2)

MANIFESTS={
 'E6A': ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/train_B1A_E6A_single_axis_visible_eval_original.csv',
 'E6B': ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/train_B1A_E6B_compound_visible_eval_original.csv',
}
SOURCES={
 'E6A': 'green_edgefit_extreme_E6A_single_axis_visible_ccpdboard',
 'E6B': 'green_edgefit_extreme_E6B_compound_visible_ccpdboard',
}

for name,mpath in MANIFESTS.items():
    rows=list(csv.DictReader(mpath.open('r',encoding='utf-8-sig')))
    ext=[r for r in rows if r.get('source')==SOURCES[name]]
    # pick 8 per set: 3 low, 3 mid, 2 high (across varied directions)
    picked=[]
    for tier in ['low','mid','high']:
        tier_rows=sorted([r for r in ext if r.get('difficulty_tier')==tier],
                         key=lambda r: (r.get('extreme_direction',''), r['text']))
        n=3 if tier!='high' else 2
        picked.extend(tier_rows[:n])

    for idx,r in enumerate(picked):
        img=cv2.imread(r['img_path'], cv2.IMREAD_COLOR)
        q=get_quad(r)
        # ccpd_board pipeline
        prep,occ,warped,_,_ = prepare_board_ocr_input_from_quad_bgr888(img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)
        # raw + quad preview
        preview=img.copy()
        cv2.polylines(preview, [q.astype(np.int32).reshape((-1,1,2))], True, (0,255,0), 2)
        # zoom
        gray_big=cv2.resize(prep, (94*ZOOM, 24*ZOOM), interpolation=cv2.INTER_NEAREST)
        raw_resized=cv2.resize(preview, (gray_big.shape[1], gray_big.shape[0]), interpolation=cv2.INTER_LINEAR)
        collage=np.concatenate([raw_resized, gray_big], axis=0)
        label=f"{name} {r['text']} {r.get('difficulty_tier','')}/{r.get('extreme_direction','')}"
        cv2.putText(collage, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3, cv2.LINE_AA)
        fname=f"{name}_{idx:02d}_{r['text']}_{r.get('difficulty_tier','')}_{r.get('extreme_direction','')}.png"
        safe=''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in fname)
        cv2.imwrite(str(OUT/safe), collage)
    print(f'{name}: wrote {len(picked)} samples')

# Also pure 94x24 gray3 for direct inspection
RAW_OUT=OUT/'raw_94x24'
RAW_OUT.mkdir(exist_ok=True)
for name,mpath in MANIFESTS.items():
    rows=list(csv.DictReader(mpath.open('r',encoding='utf-8-sig')))
    ext=[r for r in rows if r.get('source')==SOURCES[name]]
    high=sorted([r for r in ext if r.get('difficulty_tier')=='high'],
                key=lambda r: (r.get('extreme_direction',''), r['text']))
    for idx,r in enumerate(high[:4]):
        img=cv2.imread(r['img_path'], cv2.IMREAD_COLOR)
        q=get_quad(r)
        prep,occ,warped,_,_ = prepare_board_ocr_input_from_quad_bgr888(img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)
        gray_big=cv2.resize(prep, (94*ZOOM, 24*ZOOM), interpolation=cv2.INTER_NEAREST)
        fname=f"{name}_high_{idx:02d}_{r['text']}_{r.get('extreme_direction','')}.png"
        safe=''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in fname)
        cv2.imwrite(str(RAW_OUT/safe), gray_big)
    print(f'{name} raw: wrote {min(4,len(high))} samples')
print(f'\nall saved to {OUT}')
