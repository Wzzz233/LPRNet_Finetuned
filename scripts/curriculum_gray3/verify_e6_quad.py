#!/usr/bin/env python3
"""验证 E6 extreme quad 是否正确：在图上画出 quad 看看四角是否真的框住车牌"""
import csv, sys
from pathlib import Path
import cv2, numpy as np

ROOT=Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT/'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

MANIFEST=ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/train_B1A_E6A_single_axis_visible_eval_original.csv'
QKEYS=['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']
SRC='green_edgefit_extreme_E6A_single_axis_visible_ccpdboard'
OUT=ROOT/'tmp/quad_verification'
OUT.mkdir(parents=True, exist_ok=True)

def get_quad(row):
    vals=[]; 
    for k in QKEYS:
        v=row.get(k,'')
        if v=='': return None
        vals.append(float(v))
    return np.array(vals,dtype=np.float32).reshape(4,2)

rows=list(csv.DictReader(MANIFEST.open('r',encoding='utf-8-sig')))
ext=[r for r in rows if r.get('source')==SRC]

# pick high tier from different directions
for tier in ['high','mid','low']:
    tier_rows=[r for r in ext if r.get('difficulty_tier')==tier]
    # sample 2 per tier
    for idx,r in enumerate(tier_rows[:2]):
        img=cv2.imread(r['img_path'], cv2.IMREAD_COLOR)
        q=get_quad(r)

        # draw quad with numbered corners
        vis=img.copy()
        pts=q.astype(np.int32)
        for ci,(x,y) in enumerate(pts):
            cv2.circle(vis, (x,y), 6, (0,0,255), -1)
            cv2.putText(vis, str(ci), (x+8,y-8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
        cv2.polylines(vis, [pts.reshape((-1,1,2))], True, (0,255,0), 2)

        # warp
        prep,occ,warped,_,_ = prepare_board_ocr_input_from_quad_bgr888(img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)
        gray_big=cv2.resize(prep, (94*8, 24*8), interpolation=cv2.INTER_NEAREST)
        warp_big=cv2.resize(warped, (warped.shape[1]*6, warped.shape[0]*6), interpolation=cv2.INTER_LINEAR)

        # collage: raw+quad | warp | gray3
        h=max(vis.shape[0], warp_big.shape[0], gray_big.shape[0])
        def pad(im,h):
            top=(h-im.shape[0])//2; bot=h-im.shape[0]-top
            return cv2.copyMakeBorder(im,top,bot,12,12,cv2.BORDER_CONSTANT,value=(245,245,245))
        a,b,c=pad(vis,h), pad(warp_big,h), pad(gray_big,h)
        collage=np.concatenate([a,b,c], axis=1)

        txt=r['text']; direc=r.get('extreme_direction','')
        cv2.putText(collage, f"{txt} {tier}/{direc}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(collage, f"quad: [{int(q[0,0])},{int(q[0,1])}] [{int(q[1,0])},{int(q[1,1])}] [{int(q[2,0])},{int(q[2,1])}] [{int(q[3,0])},{int(q[3,1])}]", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(collage, 'raw+quad', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,120,0), 2, cv2.LINE_AA)
        cv2.putText(collage, 'warp', (a.shape[1]+20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(collage, 'gray3 94x24', (a.shape[1]+b.shape[1]+20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

        fname=f"quad_check_{tier}_{idx}_{txt}.png"
        safe=''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in fname)
        cv2.imwrite(str(OUT/safe), collage)
print(f'wrote to {OUT}')
print('files:', list(OUT.glob('*.png')))
