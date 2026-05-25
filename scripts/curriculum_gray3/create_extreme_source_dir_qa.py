#!/usr/bin/env python3
"""Create QA sheets comparing multiple green extreme data directories.

Outputs per-directory contact sheets and a mixed overview:
- full crop before 94x24 resize
- parsed filename quad overlay when available
- actual plain_plate-style 94x24 gray3/direct-resize preview enlarged
- simple foreground centering metrics
"""
import csv
import json
import math
import random
import re
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
OUT = ROOT / 'reports' / 'stageB1A_extreme_source_QA_20260425'
WIN = Path('/mnt/c/Users/Wzzz2/Desktop/stageB1A_extreme_source_QA_20260425')
OUT.mkdir(parents=True, exist_ok=True)
WIN.mkdir(parents=True, exist_ok=True)

FONT_PATHS = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
FONT_PATH = next(p for p in FONT_PATHS if Path(p).exists())
FONT = ImageFont.truetype(FONT_PATH, 18)
SMALL = ImageFont.truetype(FONT_PATH, 13)
TINY = ImageFont.truetype(FONT_PATH, 11)

CANDIDATES = [
    ('B1A_USED_tier3_full_v2_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v2/images/train/extreme'),
    ('tier3_full_v1_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v1/images/train/extreme'),
    ('tier3_full_v3_su_conservative_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v3_su_conservative/images/train/extreme'),
    ('old_green_edgefit_v3_allprov_train_extreme', ROOT/'datasets/green_edgefit_v3_allprov/images/train/extreme'),
    ('v4_e2_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e2_20260411/images/train/board_extreme_tail'),
    ('v4_e3_equalprov_a_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail'),
    ('v4_e4_extreme_append10_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e4_extreme_append10_20260412/images/train/board_extreme_tail'),
    ('v4_boardlike_a3000_board_extreme_tail', ROOT/'green_edgefit_v4_boardlike_a3000/images/train/board_extreme_tail'),
    ('e16a_board_extreme_tail', ROOT/'generated/green_e16a_nonanhui_ad_balance_12k_std_v1/images/train/board_extreme_tail'),
]
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
PROV_ORDER = ['沪','苏','浙','粤','皖','京','湘','冀','陕','鄂','鲁','川']

quad_re = re.compile(r'(?P<bbox>\d+&\d+_\d+&\d+)-(?P<quad>\d+&\d+_\d+&\d+_\d+&\d+_\d+&\d+)')

def parse_quad_from_name(name):
    m = quad_re.search(name)
    if not m:
        return None
    pts = []
    for token in m.group('quad').split('_'):
        x, y = token.split('&')
        pts.append((float(x), float(y)))
    return np.array(pts, dtype=np.float32)

def infer_text(path):
    stem = path.stem
    # Common tail: ...-沪AD12345.jpg or ...-沪-0034-沪HD19700.jpg
    m = re.search(r'([\u4e00-\u9fa5][A-Z0-9]{6,7})$', stem)
    if m:
        return m.group(1)
    # e20a-冀AD06028-board_extreme_tail-029
    m = re.search(r'-([\u4e00-\u9fa5][A-Z0-9]{6,7})-', stem)
    if m:
        return m.group(1)
    return ''

def collect_images(d):
    if not d.exists():
        return []
    return [p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT]

def pick_samples(paths, n=24, seed=20260425):
    rng = random.Random(seed)
    by_prov = defaultdict(list)
    for p in paths:
        txt = infer_text(p)
        prov = txt[:1] if txt else p.parent.name
        by_prov[prov].append(p)
    for v in by_prov.values():
        v.sort(key=lambda x: str(x))
        rng.shuffle(v)
    out = []
    keys = [k for k in PROV_ORDER if k in by_prov] + sorted(k for k in by_prov if k not in PROV_ORDER)
    while len(out) < n and any(by_prov.values()):
        for k in keys:
            if by_prov[k] and len(out) < n:
                out.append(by_prov[k].pop())
    return out

def pil_from_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def fit_pil(img, size, fill=(245,245,245)):
    if isinstance(img, np.ndarray):
        img = pil_from_bgr(img)
    w, h = img.size
    scale = min(size[0]/max(1,w), size[1]/max(1,h))
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    im = img.resize((nw, nh), Image.Resampling.BILINEAR)
    can = Image.new('RGB', size, fill)
    can.paste(im, ((size[0]-nw)//2, (size[1]-nh)//2))
    return can

def direct_94_gray3(img_bgr):
    # Mirrors plain_plate visual semantics for synthetic support: direct resize to 94x24 then gray3.
    r = cv2.resize(img_bgr, (94,24), interpolation=cv2.INTER_NEAREST)
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def fg_metrics(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Green synthetic backgrounds are not pure black; use non-near-white/non-near-black robustly.
    # For center QA, report bbox over pixels differing from border median.
    border = np.concatenate([gray[0,:], gray[-1,:], gray[:,0], gray[:,-1]])
    med = float(np.median(border))
    mask = np.abs(gray.astype(np.float32) - med) > 10
    # fallback: non-black
    if mask.sum() < 10:
        mask = gray > 8
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1,y1,x2,y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    cx = (x1+x2+1)/2.0
    cy = (y1+y2+1)/2.0
    h,w = gray.shape[:2]
    return {
        'bbox':[x1,y1,x2,y2], 'bbox_w':x2-x1+1, 'bbox_h':y2-y1+1,
        'cx_norm':cx/w, 'cy_norm':cy/h, 'occ_w':(x2-x1+1)/w, 'occ_h':(y2-y1+1)/h,
    }

def quad_metrics(q):
    if q is None:
        return {}
    def dist(a,b): return float(np.linalg.norm(a-b))
    top, right, bottom, left = dist(q[0],q[1]), dist(q[1],q[2]), dist(q[2],q[3]), dist(q[3],q[0])
    def angle(a,b):
        dx,dy = b[0]-a[0], b[1]-a[1]
        return math.degrees(math.atan2(dy, dx))
    horiz = max(abs(angle(q[0],q[1])), abs(angle(q[3],q[2])))
    # deviation from vertical for left/right edges
    vert = max(abs(abs(angle(q[0],q[3]))-90), abs(abs(angle(q[1],q[2]))-90))
    return {
        'top':top, 'bottom':bottom, 'left':left, 'right':right,
        'width_ratio': max(top,bottom)/max(1e-6,min(top,bottom)),
        'height_ratio': max(left,right)/max(1e-6,min(left,right)),
        'horiz_tilt_deg': horiz, 'vert_tilt_deg': vert,
    }

def draw_quad_scaled(draw, q, src_shape, origin, box_size):
    if q is None:
        return
    h,w = src_shape[:2]
    scale = min(box_size[0]/max(1,w), box_size[1]/max(1,h))
    offx = origin[0] + (box_size[0]-int(w*scale))//2
    offy = origin[1] + (box_size[1]-int(h*scale))//2
    pts = [(offx + float(x)*scale, offy + float(y)*scale) for x,y in q]
    draw.line(pts + [pts[0]], fill=(255,0,0), width=2)
    for i,(x,y) in enumerate(pts):
        draw.ellipse([x-3,y-3,x+3,y+3], fill=(255,0,0))
        draw.text((x+3,y+2), str(i+1), font=TINY, fill=(255,0,0))

def render_sheet(name, paths, out_path, cols=3):
    cell_w, cell_h = 350, 242
    title_h = 78
    rows = math.ceil(len(paths)/cols)
    can = Image.new('RGB', (cols*cell_w, title_h+rows*cell_h), (255,255,255))
    d = ImageDraw.Draw(can)
    d.text((12,8), name, font=FONT, fill=(0,0,0))
    d.text((12,34), '上: full crop before 94x24 resize + parsed quad(red if available); 下: plain_plate actual 94x24 gray3/direct-resize enlarged', font=SMALL, fill=(60,60,60))
    d.text((12,54), f'font={FONT_PATH}', font=TINY, fill=(90,90,90))
    records = []
    for i,p in enumerate(paths):
        x = (i%cols)*cell_w
        y = title_h + (i//cols)*cell_h
        d.rectangle([x,y,x+cell_w-1,y+cell_h-1], outline=(200,200,200))
        img = cv2.imread(str(p))
        if img is None:
            continue
        q = parse_quad_from_name(p.name)
        qmet = quad_metrics(q)
        crop_box = (330, 96)
        crop = fit_pil(img, crop_box)
        can.paste(crop, (x+8, y+8))
        draw_quad_scaled(d, q, img.shape, (x+8,y+8), crop_box)
        inp = direct_94_gray3(img)
        inp_big = pil_from_bgr(cv2.resize(inp, (282,72), interpolation=cv2.INTER_NEAREST))
        can.paste(inp_big, (x+8, y+111))
        met = fg_metrics(inp)
        text = infer_text(p)
        d.text((x+8,y+186), f'{i:02d} {text} {p.parent.name} size={img.shape[1]}x{img.shape[0]}', font=TINY, fill=(0,0,0))
        if met:
            d.text((x+8,y+201), f'94bbox={met["bbox"]} cx={met["cx_norm"]:.2f},{met["cy_norm"]:.2f} occ={met["occ_w"]:.2f},{met["occ_h"]:.2f}', font=TINY, fill=(120,0,0) if abs(met['cx_norm']-0.5)>0.08 else (0,90,0))
        if qmet:
            d.text((x+8,y+216), f'quad tilt H/V={qmet["horiz_tilt_deg"]:.1f}/{qmet["vert_tilt_deg"]:.1f} wr/hr={qmet["width_ratio"]:.2f}/{qmet["height_ratio"]:.2f}', font=TINY, fill=(0,0,120))
        else:
            d.text((x+8,y+216), 'quad: not parsed from filename', font=TINY, fill=(120,120,120))
        rec = {'dir_name':name, 'path':str(p), 'text':text, 'image_w':int(img.shape[1]), 'image_h':int(img.shape[0]), 'plain94_metrics':met, 'quad_metrics':qmet}
        records.append(rec)
    can.save(out_path, quality=92)
    return records

def summarize(records):
    vals = defaultdict(list)
    for r in records:
        met = r.get('plain94_metrics') or {}
        qmet = r.get('quad_metrics') or {}
        for k in ['cx_norm','cy_norm','occ_w','occ_h']:
            if k in met: vals[k].append(float(met[k]))
        for k in ['horiz_tilt_deg','vert_tilt_deg','width_ratio','height_ratio']:
            if k in qmet and math.isfinite(qmet[k]): vals[k].append(float(qmet[k]))
    out = {'n':len(records)}
    for k,v in vals.items():
        if not v: continue
        a = np.array(v, dtype=float)
        out[k] = {'mean':float(a.mean()), 'p50':float(np.percentile(a,50)), 'p95':float(np.percentile(a,95)), 'min':float(a.min()), 'max':float(a.max())}
    return out

def main():
    all_summary = []
    all_records = []
    overview_paths = []
    for name,dpath in CANDIDATES:
        paths = collect_images(dpath)
        if not paths:
            all_summary.append({'name':name, 'dir':str(dpath), 'exists':dpath.exists(), 'count':0, 'note':'no images'})
            continue
        samples = pick_samples(paths, 24, seed=abs(hash(name)) & 0xffffffff)
        sheet = OUT / f'{name}.jpg'
        recs = render_sheet(f'{name}  count={len(paths)}  dir={dpath}', samples, sheet)
        for r in recs:
            r['source_dir'] = str(dpath)
        all_records.extend(recs)
        all_summary.append({'name':name, 'dir':str(dpath), 'exists':True, 'count':len(paths), 'sheet':str(sheet), 'stats_on_sample':summarize(recs)})
        overview_paths.extend(samples[:3])
    if overview_paths:
        recs = render_sheet('MIXED OVERVIEW: first 3 samples from each extreme candidate directory', overview_paths, OUT/'00_mixed_overview.jpg', cols=3)
        all_records.extend(recs)
    # CSV records
    with (OUT/'sample_records.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['dir_name','source_dir','path','text','image_w','image_h','cx_norm','cy_norm','occ_w','occ_h','horiz_tilt_deg','vert_tilt_deg','width_ratio','height_ratio']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_records:
            met = r.get('plain94_metrics') or {}; q = r.get('quad_metrics') or {}
            row = {k:r.get(k,'') for k in ['dir_name','source_dir','path','text','image_w','image_h']}
            for k in ['cx_norm','cy_norm','occ_w','occ_h']:
                row[k] = met.get(k,'')
            for k in ['horiz_tilt_deg','vert_tilt_deg','width_ratio','height_ratio']:
                row[k] = q.get(k,'')
            w.writerow(row)
    (OUT/'summary.json').write_text(json.dumps({'out_dir':str(OUT),'windows_out_dir':str(WIN),'font_path':FONT_PATH,'candidates':all_summary}, ensure_ascii=False, indent=2), encoding='utf-8')
    # HTML index
    imgs = ['00_mixed_overview.jpg'] + [Path(s['sheet']).name for s in all_summary if s.get('sheet')]
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>Extreme source QA</title><style>body{font-family:sans-serif;background:#eee} img{display:block;max-width:100%;margin:18px auto;border:1px solid #999} pre{background:white;padding:12px}</style></head><body>']
    html.append('<h1>StageB1A extreme source directory QA</h1>')
    html.append('<pre>'+json.dumps(all_summary, ensure_ascii=False, indent=2)+'</pre>')
    for img in imgs:
        if (OUT/img).exists():
            html.append(f'<h2>{img}</h2><img src="{img}">')
    html.append('</body></html>')
    (OUT/'index.html').write_text('\n'.join(html), encoding='utf-8')
    # copy deliverables
    for p in OUT.iterdir():
        if p.is_file():
            shutil.copy2(p, WIN/p.name)
    print(json.dumps({'out_dir':str(OUT),'windows_out_dir':str(WIN),'num_candidates':len(CANDIDATES),'num_records':len(all_records)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
