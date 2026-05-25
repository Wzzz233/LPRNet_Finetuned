#!/usr/bin/env python3
import csv
import json
import math
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
OUT = ROOT / 'reports' / 'stageB1A_extreme_source_QA_boardwarp_20260425'
WIN = Path('/mnt/c/Users/Wzzz2/Desktop/stageB1A_extreme_source_QA_boardwarp_20260425')
OUT.mkdir(parents=True, exist_ok=True)
WIN.mkdir(parents=True, exist_ok=True)

FONT_PATH = next(p for p in [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
] if Path(p).exists())
FONT = ImageFont.truetype(FONT_PATH, 18)
SMALL = ImageFont.truetype(FONT_PATH, 12)
TINY = ImageFont.truetype(FONT_PATH, 11)

import sys
sys.path.insert(0, str(ROOT/'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name  # noqa: E402

candidates = [
    ('B1A_USED_tier3_full_v2_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v2/images/train/extreme'),
    ('tier3_full_v1_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v1/images/train/extreme'),
    ('tier3_full_v3_su_conservative_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v3_su_conservative/images/train/extreme'),
    ('v4_e2_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e2_20260411/images/train/board_extreme_tail'),
    ('v4_e3_equalprov_a_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail'),
    ('v4_e4_extreme_append10_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e4_extreme_append10_20260412/images/train/board_extreme_tail'),
]
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
PROVS = ['沪','苏','浙','粤','皖','京','湘','冀','陕','鄂','鲁','川']

quad_re = re.compile(r'([\u4e00-\u9fa5][A-Z0-9]{6,7})')

def infer_text(path):
    ms = quad_re.findall(path.stem)
    return ms[-1] if ms else ''

def collect_images(d):
    return [p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT]

def pick_samples(paths, n=18):
    by = {}
    for p in sorted(paths):
        txt = infer_text(p)
        prov = txt[:1] if txt else p.parent.name
        by.setdefault(prov, []).append(p)
    out = []
    keys = [k for k in PROVS if k in by] + [k for k in sorted(by) if k not in PROVS]
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop(0))
    return out

def pil_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def fit(im, size, fill=(245,245,245)):
    if isinstance(im, np.ndarray):
        im = pil_bgr(im)
    w,h = im.size
    scale = min(size[0]/max(1,w), size[1]/max(1,h))
    nw,nh = max(1,int(w*scale)), max(1,int(h*scale))
    rs = im.resize((nw,nh), Image.Resampling.BILINEAR)
    can = Image.new('RGB', size, fill)
    can.paste(rs, ((size[0]-nw)//2, (size[1]-nh)//2))
    return can

def draw_quad(draw, q, img_shape, origin, box_size):
    if q is None:
        return
    h,w = img_shape[:2]
    scale = min(box_size[0]/w, box_size[1]/h)
    offx = origin[0] + (box_size[0]-int(w*scale))//2
    offy = origin[1] + (box_size[1]-int(h*scale))//2
    pts = [(offx+x*scale, offy+y*scale) for x,y in q]
    draw.line(pts+[pts[0]], fill=(255,0,0), width=2)

def bbox_nonblack(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 8)
    if len(xs) == 0:
        return None
    x1,y1,x2,y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return [x1,y1,x2,y2], (x1+x2+1)/2/gray.shape[1], (y1+y2+1)/2/gray.shape[0], (x2-x1+1)/gray.shape[1], (y2-y1+1)/gray.shape[0]

def render_sheet(name, paths, out_path):
    cell_w, cell_h, title_h, cols = 420, 250, 76, 3
    rows = math.ceil(len(paths)/cols)
    can = Image.new('RGB', (cols*cell_w, title_h+rows*cell_h), (255,255,255))
    d = ImageDraw.Draw(can)
    d.text((10,8), name, font=FONT, fill=(0,0,0))
    d.text((10,34), '上左: 原图+quad  上右: warp后图  下: 按训练/板端 ccpd_board 走 prepare_board_ocr_input_from_quad_bgr888 后的94x24 gray3放大', font=SMALL, fill=(60,60,60))
    d.text((10,52), f'真实链路: warpPerspective -> letterbox(94x24) -> gray3, font={FONT_PATH}', font=TINY, fill=(90,90,90))
    rows_out = []
    for i,p in enumerate(paths):
        x=(i%3)*cell_w; y=title_h+(i//3)*cell_h
        d.rectangle([x,y,x+cell_w-1,y+cell_h-1], outline=(200,200,200))
        img = cv2.imread(str(p))
        if img is None:
            continue
        q = parse_ccpd_quad_from_name(p.name)
        if q is None:
            d.text((x+8,y+8), f'quad parse failed: {p.name}', font=SMALL, fill=(180,0,0))
            continue
        prep, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
            img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0
        )
        full_box = (190, 86)
        warp_box = (190, 86)
        can.paste(fit(img, full_box), (x+8, y+8))
        draw_quad(d, q, img.shape, (x+8,y+8), full_box)
        can.paste(fit(warped, warp_box), (x+215, y+8))
        prep_big = cv2.resize(prep, (388, 96), interpolation=cv2.INTER_NEAREST)
        can.paste(pil_bgr(prep_big), (x+8, y+106))
        bb = bbox_nonblack(prep)
        txt = infer_text(p)
        d.text((x+8,y+208), f'{i:02d} {txt} {p.parent.name}', font=TINY, fill=(0,0,0))
        d.text((x+8,y+223), f'img={img.shape[1]}x{img.shape[0]} warp={warped.shape[1]}x{warped.shape[0]} occ={occ:.3f}', font=TINY, fill=(0,0,120))
        if bb:
            bbox, cx, cy, ow, oh = bb
            d.text((x+180,y+208), f'94bbox={bbox} cx={cx:.2f},{cy:.2f} occwh={ow:.2f},{oh:.2f}', font=TINY, fill=(180,0,0) if abs(cx-0.5)>0.06 else (0,120,0))
            rows_out.append({'dir_name':name,'path':str(p),'text':txt,'occ':occ,'cx_norm':cx,'cy_norm':cy,'occ_w':ow,'occ_h':oh,'warp_w':int(warped.shape[1]),'warp_h':int(warped.shape[0])})
    can.save(out_path, quality=92)
    return rows_out

def summarize(rows):
    out = {'n': len(rows)}
    for k in ['occ','cx_norm','cy_norm','occ_w','occ_h','warp_w','warp_h']:
        vals = np.array([float(r[k]) for r in rows], dtype=float) if rows else np.array([])
        if vals.size:
            out[k] = {'mean': float(vals.mean()), 'p50': float(np.percentile(vals,50)), 'p95': float(np.percentile(vals,95)), 'min': float(vals.min()), 'max': float(vals.max())}
    return out

def main():
    all_summary=[]
    all_rows=[]
    mixed=[]
    for name, d in candidates:
        paths = collect_images(d)
        samples = pick_samples(paths, 18)
        out_img = OUT/f'{name}_boardwarp.jpg'
        rows = render_sheet(f'{name} dir={d}', samples, out_img)
        for r in rows:
            r['source_dir'] = str(d)
        all_rows.extend(rows)
        all_summary.append({'name':name,'dir':str(d),'count':len(paths),'sheet':str(out_img),'stats_on_sample':summarize(rows)})
        mixed.extend(samples[:3])
    if mixed:
        render_sheet('MIXED boardwarp overview', mixed, OUT/'00_mixed_boardwarp_overview.jpg')
    with (OUT/'sample_records.csv').open('w', encoding='utf-8', newline='') as f:
        fields=['dir_name','source_dir','path','text','occ','cx_norm','cy_norm','occ_w','occ_h','warp_w','warp_h']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); [w.writerow(r) for r in all_rows]
    (OUT/'summary.json').write_text(json.dumps({'out_dir':str(OUT),'windows_out_dir':str(WIN),'candidates':all_summary}, ensure_ascii=False, indent=2), encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file(): shutil.copy2(p, WIN/p.name)
    print(json.dumps({'out_dir':str(OUT),'windows_out_dir':str(WIN),'n_rows':len(all_rows)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
