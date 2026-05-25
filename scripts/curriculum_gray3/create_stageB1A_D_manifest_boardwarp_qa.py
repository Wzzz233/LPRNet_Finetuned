#!/usr/bin/env python3
"""QA final StageB1A-D train/proxy manifests using actual ccpd_board final94 path."""
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
SRC_MANIFEST_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_difficulty'
D_MANIFEST_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_D_extreme900_v4e3_ccpdboard_eval_original'
NEW_PROXY_DIR = ROOT / 'manifests/curriculum_gray3_stageb_v1_B1A_D_new_v4e3_ccpdboard_proxy'
OUT = ROOT / 'reports/stageB1A_D_extreme900_v4e3_ccpdboard_manifest_QA_20260425'
WIN = Path('/mnt/c/Users/Wzzz2/Desktop/stageB1A_D_extreme900_v4e3_ccpdboard_manifest_QA_20260425')
OUT.mkdir(parents=True, exist_ok=True)
WIN.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / 'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888  # noqa: E402

FONT_PATH = next(p for p in [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
] if Path(p).exists())
FONT = ImageFont.truetype(FONT_PATH, 18)
SMALL = ImageFont.truetype(FONT_PATH, 12)
TINY = ImageFont.truetype(FONT_PATH, 10)
PROVS = ['沪','苏','浙','粤','皖','京','湘','冀','陕','鄂','鲁','川','闽','赣','豫','渝']


def read_rows(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def abs_path(row):
    p=row.get('img_path') or ''
    return Path(p) if p else ROOT/(row.get('img_rel_path') or '')


def get_quad(row):
    vals=[]
    for k in ['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']:
        v=row.get(k,'')
        if v=='': return None
        vals.append(float(v))
    return np.array(vals,dtype=np.float32).reshape(4,2)


def infer_quad_from_filename(path):
    parts=path.name.split('-')
    if len(parts)<4: return None
    pts=[]
    try:
        for tok in parts[3].split('_'):
            x,y=tok.split('&'); pts.append((float(x),float(y)))
    except Exception: return None
    return np.array(pts,dtype=np.float32) if len(pts)==4 else None


def bbox_nonblack(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys,xs=np.where(gray>8)
    if len(xs)==0: return None
    x1,y1,x2,y2=int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
    w,h=gray.shape[1],gray.shape[0]
    return {'bbox':[x1,y1,x2,y2],'cx_norm':(x1+x2+1)/2/w,'cy_norm':(y1+y2+1)/2/h,'occ_w':(x2-x1+1)/w,'occ_h':(y2-y1+1)/h,'margin_l':x1/w,'margin_r':(w-1-x2)/w,'margin_t':y1/h,'margin_b':(h-1-y2)/h}


def pil_bgr(img): return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def fit(im,size,fill=(245,245,245)):
    if isinstance(im,np.ndarray): im=pil_bgr(im)
    w,h=im.size; scale=min(size[0]/max(1,w), size[1]/max(1,h)); nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
    rs=im.resize((nw,nh), Image.Resampling.BILINEAR); can=Image.new('RGB',size,fill); can.paste(rs,((size[0]-nw)//2,(size[1]-nh)//2)); return can

def draw_quad(draw,q,img_shape,origin,box_size):
    if q is None: return
    h,w=img_shape[:2]; scale=min(box_size[0]/w,box_size[1]/h); offx=origin[0]+(box_size[0]-int(w*scale))//2; offy=origin[1]+(box_size[1]-int(h*scale))//2
    pts=[(offx+x*scale,offy+y*scale) for x,y in q]; draw.line(pts+[pts[0]],fill=(255,0,0),width=2)

def pick_by_prov(rows,n=48):
    by=defaultdict(list)
    for r in rows: by[(r.get('text') or '')[:1]].append(r)
    for k in by: by[k].sort(key=lambda r: abs_path(r).name)
    out=[]; keys=[k for k in PROVS if k in by]+[k for k in sorted(by) if k not in PROVS]
    while len(out)<n and any(by.values()):
        for k in keys:
            if by[k] and len(out)<n: out.append(by[k].pop(0))
    return out


def render_rows(title, rows, out_path, mode):
    cell_w,cell_h,title_h,cols=430,250,82,3
    can=Image.new('RGB',(cols*cell_w,title_h+math.ceil(len(rows)/cols)*cell_h),(255,255,255)); d=ImageDraw.Draw(can)
    d.text((10,8),title,font=FONT,fill=(0,0,0)); d.text((10,34),'每格: 原图+quad / warp / final94 gray3；D为manifest真实训练输入。',font=SMALL,fill=(50,50,50)); d.text((10,54),f'font={FONT_PATH}',font=TINY,fill=(90,90,90))
    recs=[]
    for i,r in enumerate(rows):
        x=(i%cols)*cell_w; y=title_h+(i//cols)*cell_h; d.rectangle([x,y,x+cell_w-1,y+cell_h-1],outline=(205,205,205))
        p=abs_path(r); img=cv2.imread(str(p))
        if img is None: d.text((x+8,y+8),f'cv2 failed {p}',font=SMALL,fill=(200,0,0)); continue
        q=get_quad(r) if mode=='manifest_quad' else infer_quad_from_filename(p)
        if q is None:
            prep=cv2.resize(img,(94,24),interpolation=cv2.INTER_NEAREST); warped=img; occ=-1.0
        else:
            prep,occ,warped,_,_=prepare_board_ocr_input_from_quad_bgr888(img,q,94,24,'letterbox','nn','gray3','bgr',quad_pad_ratio=0.0)
        can.paste(fit(img,(190,86)),(x+8,y+8)); draw_quad(d,q,img.shape,(x+8,y+8),(190,86)); can.paste(fit(warped,(190,86)),(x+216,y+8))
        prep_big=cv2.resize(prep,(388,96),interpolation=cv2.INTER_NEAREST); can.paste(pil_bgr(prep_big),(x+8,y+106))
        bb=bbox_nonblack(prep); txt=r.get('text','')
        d.text((x+8,y+208),f'{i:02d} {txt} {r.get("source","")[:34]}',font=TINY,fill=(0,0,0)); d.text((x+8,y+223),f'{p.parent.name}/{p.name[:42]}',font=TINY,fill=(0,0,120))
        rec={'title':title,'idx':i,'text':txt,'path':str(p),'source':r.get('source',''),'preprocess_group':r.get('preprocess_group',''),'occ':occ,'has_quad':q is not None}
        if bb:
            rec.update({k:v for k,v in bb.items() if k!='bbox'}); rec['bbox']=bb['bbox']
            color=(180,0,0) if bb['margin_l']<0.02 or bb['margin_r']<0.02 or bb['margin_t']<0.02 or bb['margin_b']<0.02 else (0,120,0)
            d.text((x+178,y+208),f"bbox={bb['bbox']} ow/oh={bb['occ_w']:.2f}/{bb['occ_h']:.2f}",font=TINY,fill=color); d.text((x+178,y+223),f"mLRTB={bb['margin_l']:.2f},{bb['margin_r']:.2f},{bb['margin_t']:.2f},{bb['margin_b']:.2f}",font=TINY,fill=color)
        recs.append(rec)
    can.save(out_path,quality=92); return recs


def quad_direction_stats(rows):
    stats=Counter()
    for r in rows:
        q=get_quad(r)
        if q is None: stats['missing_quad']+=1; continue
        left=(q[0]+q[3])/2; right=(q[1]+q[2])/2; top=(q[0]+q[1])/2; bot=(q[3]+q[2])/2
        rise=float(right[1]-left[1]); skew=float(bot[0]-top[0])
        stats['rise_up' if rise<-1 else ('rise_down' if rise>1 else 'rise_flat')]+=1
        stats['skew_left' if skew<-1 else ('skew_right' if skew>1 else 'skew_flat')]+=1
    return dict(stats)


def summarize_numeric(records):
    out={'n':len(records)}
    for k in ['occ','cx_norm','cy_norm','occ_w','occ_h','margin_l','margin_r','margin_t','margin_b']:
        xs=[float(r[k]) for r in records if k in r and r[k] is not None and float(r[k])>=0]
        if xs:
            arr=np.array(xs,dtype=float); out[k]={'mean':float(arr.mean()),'p50':float(np.percentile(arr,50)),'p05':float(np.percentile(arr,5)),'p95':float(np.percentile(arr,95)),'min':float(arr.min()),'max':float(arr.max())}
    return out


def main():
    orig_train=[r for r in read_rows(SRC_MANIFEST_DIR/'train_B1A.csv') if r.get('source')=='green_edgefit_extreme']
    d_train=[r for r in read_rows(D_MANIFEST_DIR/'train_B1A_D_extreme900_v4e3_ccpdboard_eval_original.csv') if r.get('source')=='green_edgefit_extreme900_v4e3_ccpdboard']
    old_proxy=read_rows(D_MANIFEST_DIR/'proxy_green_edgefit_extreme.csv')
    new_proxy=read_rows(NEW_PROXY_DIR/'proxy_green_edgefit_extreme.csv')
    recs=[]
    recs+=render_rows('OLD B1A train extreme: tier3 black-border preview',pick_by_prov(orig_train,48),OUT/'01_old_B1A_train_extreme_boardwarp.jpg','filename_quad')
    recs+=render_rows('D train extreme900: v4_e3 ccpd_board final training input',pick_by_prov(d_train,60),OUT/'02_D_train_extreme900_v4e3_ccpdboard_final94.jpg','manifest_quad')
    recs+=render_rows('D old proxy unchanged: original StageB1A benchmark',pick_by_prov(old_proxy,48),OUT/'03_old_proxy_unchanged_preview.jpg','filename_quad')
    recs+=render_rows('D new proxy: non-overlap v4_e3 ccpd_board difficulty eval',pick_by_prov(new_proxy,48),OUT/'04_new_proxy_v4e3_ccpdboard_final94.jpg','manifest_quad')
    train_paths={str(abs_path(r)) for r in d_train}; proxy_paths={str(abs_path(r)) for r in new_proxy}
    summary={'out_dir':str(OUT),'windows_out_dir':str(WIN),'inputs':{'orig_train_extreme_count':len(orig_train),'d_train_extreme_count':len(d_train),'old_proxy_count':len(old_proxy),'new_proxy_count':len(new_proxy),'train_new_proxy_overlap':len(train_paths & proxy_paths)},'province_counts':{'orig_train':dict(sorted(Counter(r['text'][0] for r in orig_train).items())),'d_train':dict(sorted(Counter(r['text'][0] for r in d_train).items())),'old_proxy':dict(sorted(Counter(r['text'][0] for r in old_proxy).items())),'new_proxy':dict(sorted(Counter(r['text'][0] for r in new_proxy).items()))},'direction_stats_d_train':quad_direction_stats(d_train),'direction_stats_new_proxy':quad_direction_stats(new_proxy),'sample_numeric_summary':summarize_numeric(recs),'sheets':[str(OUT/'01_old_B1A_train_extreme_boardwarp.jpg'),str(OUT/'02_D_train_extreme900_v4e3_ccpdboard_final94.jpg'),str(OUT/'03_old_proxy_unchanged_preview.jpg'),str(OUT/'04_new_proxy_v4e3_ccpdboard_final94.jpg')],'known_limit':'v4_e3 is accepted as medium-tilt extreme, but samples remain tight to black border with limited background clutter.'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'sample_records.csv').open('w',encoding='utf-8',newline='') as f:
        fields=sorted({k for r in recs for k in r.keys()}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(recs)
    for p in OUT.iterdir():
        if p.is_file(): shutil.copy2(p, WIN/p.name)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
