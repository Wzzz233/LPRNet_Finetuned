#!/usr/bin/env python3
"""QA for rebuilt E6A/E6B manifests with corrected JPEG quality."""
import csv, json, math
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/wzzz/LPRNet')
OUT=ROOT/'reports/stageB1A_E6AB_manifest_QA_20260428'
OUT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT/'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

FONT_PATH=next(p for p in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc','/usr/share/fonts/opentype/unifont/unifont.otf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists())
FONT=ImageFont.truetype(FONT_PATH, 18); SMALL=ImageFont.truetype(FONT_PATH, 12); TINY=ImageFont.truetype(FONT_PATH, 10)
PROVS=['沪','苏','浙','粤','皖','京','湘','冀','陕','鄂','鲁','川','闽','赣','豫','渝']
QKEYS=['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']

def read_rows(path): return list(csv.DictReader(path.open('r',encoding='utf-8',newline='')))

def get_quad(row):
    vals=[]; 
    for k in QKEYS:
        v=row.get(k,'')
        if v=='': return None
        vals.append(float(v))
    return np.array(vals,dtype=np.float32).reshape(4,2)

def pil_bgr(img): return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def fit(im,size,fill=(245,245,245)):
    if isinstance(im,np.ndarray): im=pil_bgr(im)
    w,h=im.size; s=min(size[0]/max(1,w),size[1]/max(1,h))
    nw,nh=max(1,int(w*s)),max(1,int(h*s))
    rs=im.resize((nw,nh),Image.Resampling.BILINEAR)
    can=Image.new('RGB',size,fill); can.paste(rs,((size[0]-nw)//2,(size[1]-nh)//2))
    return can

def draw_quad(draw,q,img_shape,origin,box_size):
    if q is None: return
    h,w=img_shape[:2]; s=min(box_size[0]/w,box_size[1]/h)
    offx=origin[0]+(box_size[0]-int(w*s))//2; offy=origin[1]+(box_size[1]-int(h*s))//2
    pts=[(offx+x*s,offy+y*s) for x,y in q]
    draw.line(pts+[pts[0]],fill=(255,0,0),width=2)

def pick_by_prov(rows,n=12):
    by=defaultdict(list); 
    for r in rows: by[(r.get('text') or '')[:1]].append(r)
    for k in by: by[k].sort(key=lambda r:(r.get('difficulty_tier',''),r.get('extreme_direction','')))
    out=[]; keys=[k for k in PROVS if k in by]+[k for k in sorted(by) if k not in PROVS]
    while len(out)<n and any(by.values()):
        for k in keys:
            if by[k] and len(out)<n: out.append(by[k].pop(0))
    return out

def render_rows(title,subtitle,rows,out_path):
    cell_w,cell_h,title_h,cols=430,258,90,3
    can=Image.new('RGB',(cols*cell_w,title_h+math.ceil(len(rows)/cols)*cell_h),(255,255,255))
    d=ImageDraw.Draw(can)
    d.text((10,8),title,font=FONT,fill=(0,0,0))
    d.text((10,34),subtitle,font=SMALL,fill=(50,50,50))
    for i,r in enumerate(rows):
        x=(i%cols)*cell_w; y=title_h+(i//cols)*cell_h
        d.rectangle([x,y,x+cell_w-1,y+cell_h-1],outline=(205,205,205))
        p=Path(r['img_path']); img=cv2.imread(str(p))
        if img is None: d.text((x+8,y+8),f'cv2 failed {p}',font=SMALL,fill=(200,0,0)); continue
        q=get_quad(r)
        if q is None:
            prep=cv2.resize(img,(94,24),interpolation=cv2.INTER_NEAREST); warped=img
        else:
            prep,occ,warped,_,_=prepare_board_ocr_input_from_quad_bgr888(img,q,94,24,'letterbox','nn','gray3','bgr',quad_pad_ratio=0.0)
        can.paste(fit(img,(190,86)),(x+8,y+8))
        draw_quad(d,q,img.shape,(x+8,y+8),(190,86))
        can.paste(fit(warped,(190,86)),(x+216,y+8))
        prep_big=cv2.resize(prep,(388,96),interpolation=cv2.INTER_NEAREST)
        can.paste(pil_bgr(prep_big),(x+8,y+106))
        txt=r.get('text',''); tier=r.get('difficulty_tier',''); direc=r.get('extreme_direction','')
        d.text((x+8,y+208),f'{i:02d} {txt} {tier}/{direc}',font=TINY,fill=(0,0,0))
        d.text((x+8,y+224),f'q100 JPEG | {p.parent.name}/{p.name[:40]}',font=TINY,fill=(0,0,120))
    can.save(out_path,quality=92)

E6A_SRC='green_edgefit_extreme_E6A_single_axis_visible_ccpdboard'
E6B_SRC='green_edgefit_extreme_E6B_compound_visible_ccpdboard'
e6a_train=read_rows(ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/train_B1A_E6A_single_axis_visible_eval_original.csv')
e6b_train=read_rows(ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/train_B1A_E6B_compound_visible_eval_original.csv')
e6a=[r for r in e6a_train if r.get('source')==E6A_SRC]
e6b=[r for r in e6b_train if r.get('source')==E6B_SRC]

for tier in ['low','mid','high']:
    rows_a=pick_by_prov([r for r in e6a if r.get('difficulty_tier')==tier],12)
    rows_b=pick_by_prov([r for r in e6b if r.get('difficulty_tier')==tier],12)
    render_rows(f'E6A train {tier.upper()}: q100 JPEG rebuilt',f'single-axis visible; 每格: raw+quad / warp / final94 gray3',rows_a,OUT/f'E6A_train_{tier}_final94.jpg')
    render_rows(f'E6B train {tier.upper()}: q100 JPEG rebuilt',f'compound visible; 每格: raw+quad / warp / final94 gray3',rows_b,OUT/f'E6B_train_{tier}_final94.jpg')

print(f'QA saved to {OUT}')
for f in sorted(OUT.glob('*.jpg')): print(f'  {f.name}')
