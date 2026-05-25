#!/usr/bin/env python3
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
OUT = ROOT / 'reports' / 'stageB1A_next_extreme_candidate_audit_20260425'
WIN = Path('/mnt/c/Users/Wzzz2/Desktop/stageB1A_next_extreme_candidate_audit_20260425')
OUT.mkdir(parents=True, exist_ok=True)
WIN.mkdir(parents=True, exist_ok=True)

FONT_PATH = next(p for p in [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
] if Path(p).exists())
FONT = ImageFont.truetype(FONT_PATH, 18)
SMALL = ImageFont.truetype(FONT_PATH, 12)
TINY = ImageFont.truetype(FONT_PATH, 10)

import sys
sys.path.insert(0, str(ROOT/'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name  # noqa: E402

CANDIDATES = [
    ('B1A_USED_tier3_full_v2_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v2/images/train/extreme'),
    ('v4_e2_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e2_20260411/images/train/board_extreme_tail'),
    ('v4_e3_equalprov_a_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail'),
    ('v4_e4_extreme_append10_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e4_extreme_append10_20260412/images/train/board_extreme_tail'),
]
IMG_EXT = {'.jpg','.jpeg','.png','.bmp','.ppm'}
PROVS = ['沪','苏','浙','粤','皖','京','湘','冀','陕','鄂','鲁','川']


def infer_text(path):
    stem = path.stem
    for i in range(len(stem)-1):
        ch = stem[i]
        if '\u4e00' <= ch <= '\u9fff':
            tail = stem[i:i+8]
            if len(tail) >= 7:
                return tail[:8]
    return ''


def collect_images(d):
    return [p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT]


def quad_direction_metrics(q):
    q = np.asarray(q, dtype=np.float32)
    tl,tr,br,bl = q
    top = tr - tl
    left = bl - tl
    horiz_deg = math.degrees(math.atan2(float(top[1]), float(top[0])))
    vert_dev_deg = math.degrees(math.atan2(float(left[0]), float(left[1])))
    center = q.mean(axis=0)
    top_mid = (tl+tr)/2.0
    bot_mid = (bl+br)/2.0
    rise = float(top_mid[1]-bot_mid[1])
    # sign of left/right edge lean and top edge rise
    return {
        'horiz_deg': horiz_deg,
        'vert_dev_deg': vert_dev_deg,
        'horiz_abs': abs(horiz_deg),
        'vert_abs': abs(vert_dev_deg),
        'rise': rise,
        'rise_sign': 'up' if rise < -1e-3 else ('down' if rise > 1e-3 else 'flat'),
        'horiz_sign': 'ccw' if horiz_deg < -1e-3 else ('cw' if horiz_deg > 1e-3 else 'flat'),
        'center_x': float(center[0]), 'center_y': float(center[1]),
    }


def nonblack_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys,xs = np.where(gray>8)
    if len(xs)==0:
        return None
    x1,y1,x2,y2 = int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
    return x1,y1,x2,y2


def boardwarp_metrics(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    q = parse_ccpd_quad_from_name(path.name)
    if q is None:
        return None
    prep, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0
    )
    bb = nonblack_bbox(prep)
    if bb is None:
        return None
    x1,y1,x2,y2 = bb
    margin_l = x1
    margin_r = 93 - x2
    margin_t = y1
    margin_b = 23 - y2
    return {
        'path': str(path),
        'text': infer_text(path),
        'quad': q.tolist(),
        'occ': float(occ),
        'warp_w': int(warped.shape[1]),
        'warp_h': int(warped.shape[0]),
        'bbox': [x1,y1,x2,y2],
        'cx_norm': float((x1+x2+1)/2/94),
        'cy_norm': float((y1+y2+1)/2/24),
        'occ_w': float((x2-x1+1)/94),
        'occ_h': float((y2-y1+1)/24),
        'margin_l': int(margin_l),
        'margin_r': int(margin_r),
        'margin_t': int(margin_t),
        'margin_b': int(margin_b),
        'margin_lr_diff': int(abs(margin_l-margin_r)),
        'margin_tb_diff': int(abs(margin_t-margin_b)),
        **quad_direction_metrics(q),
    }


def summarize_numeric(vals):
    a = np.array(vals, dtype=float)
    return {'mean': float(a.mean()), 'p50': float(np.percentile(a,50)), 'p95': float(np.percentile(a,95)), 'min': float(a.min()), 'max': float(a.max())}


def pick_by_sign(rows, n_per_bucket=6):
    buckets = defaultdict(list)
    for r in rows:
        key = (r['rise_sign'], r['horiz_sign'])
        buckets[key].append(r)
    for k in buckets:
        buckets[k].sort(key=lambda x: x['path'])
    out=[]
    order=[('up','ccw'),('up','cw'),('down','ccw'),('down','cw'),('up','flat'),('down','flat'),('flat','ccw'),('flat','cw'),('flat','flat')]
    while len(out) < n_per_bucket*4 and any(buckets.values()):
        progressed=False
        for k in order:
            if buckets[k] and len([x for x in out if (x['rise_sign'],x['horiz_sign'])==k]) < n_per_bucket:
                out.append(buckets[k].pop(0)); progressed=True
        if not progressed:
            break
    return out[:24]


def pil_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def render_candidate_sheet(name, rows, out_path):
    cell_w, cell_h, cols, title_h = 330, 170, 3, 76
    canvas = Image.new('RGB', (cols*cell_w, title_h+math.ceil(len(rows)/cols)*cell_h), (255,255,255))
    d = ImageDraw.Draw(canvas)
    d.text((10,8), name, font=FONT, fill=(0,0,0))
    d.text((10,34), '只看 boardwarp 最终 94x24 gray3 输入；按倾斜方向桶抽样。红字=贴边/方向偏置线索', font=SMALL, fill=(60,60,60))
    d.text((10,52), f'font={FONT_PATH}', font=TINY, fill=(90,90,90))
    for i,r in enumerate(rows):
        x=(i%cols)*cell_w; y=title_h+(i//cols)*cell_h
        d.rectangle([x,y,x+cell_w-1,y+cell_h-1], outline=(200,200,200))
        img = cv2.imread(r['path'])
        q = np.asarray(r['quad'], dtype=np.float32)
        prep, _, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)
        prep_big = cv2.resize(prep, (220, 60), interpolation=cv2.INTER_NEAREST)
        warp_big = cv2.resize(warped, (220, 60), interpolation=cv2.INTER_NEAREST)
        canvas.paste(pil_bgr(prep_big), (x+8,y+8))
        canvas.paste(pil_bgr(warp_big), (x+8,y+74))
        d.text((x+235,y+8), f"{r['text']}", font=SMALL, fill=(0,0,0))
        d.text((x+235,y+26), f"rise={r['rise_sign']} {r['horiz_sign']}", font=TINY, fill=(180,0,0) if r['rise_sign']=='up' else (0,0,0))
        d.text((x+235,y+40), f"h={r['horiz_deg']:.1f} v={r['vert_dev_deg']:.1f}", font=TINY, fill=(0,0,120))
        d.text((x+235,y+54), f"occ={r['occ']:.3f} cx={r['cx_norm']:.2f}", font=TINY, fill=(0,120,0))
        d.text((x+235,y+68), f"L/R={r['margin_l']}/{r['margin_r']}", font=TINY, fill=(180,0,0) if min(r['margin_l'],r['margin_r'])<=1 else (0,0,0))
        d.text((x+235,y+82), f"T/B={r['margin_t']}/{r['margin_b']}", font=TINY, fill=(180,0,0) if min(r['margin_t'],r['margin_b'])<=0 else (0,0,0))
        d.text((x+235,y+96), f"occwh={r['occ_w']:.2f}/{r['occ_h']:.2f}", font=TINY, fill=(0,0,0))
        d.text((x+8,y+140), Path(r['path']).name[:48], font=TINY, fill=(90,90,90))
    canvas.save(out_path, quality=92)


def main():
    summaries=[]
    all_csv=[]
    for name, d in CANDIDATES:
        rows=[]
        for p in collect_images(d):
            m=boardwarp_metrics(p)
            if m is not None:
                rows.append(m)
        stats={
            'count_total': len(rows),
            'direction_counts': {},
            'numeric': {},
        }
        dir_counts=defaultdict(int)
        for r in rows:
            dir_counts[f"rise:{r['rise_sign']}|rot:{r['horiz_sign']}"] += 1
        stats['direction_counts']=dict(sorted(dir_counts.items()))
        for k in ['occ','cx_norm','cy_norm','occ_w','occ_h','margin_l','margin_r','margin_t','margin_b','margin_lr_diff','margin_tb_diff','horiz_abs','vert_abs','warp_w','warp_h']:
            vals=[r[k] for r in rows]
            if vals:
                stats['numeric'][k]=summarize_numeric(vals)
        sampled=pick_by_sign(rows, n_per_bucket=6)
        out_img=OUT/f'{name}_final94_boardwarp_only.jpg'
        render_candidate_sheet(name, sampled, out_img)
        summaries.append({'name':name,'dir':str(d),'stats':stats,'sheet':str(out_img)})
        for r in rows:
            r['candidate']=name
            r['source_dir']=str(d)
            all_csv.append(r)
    # mixed compare by first 6 from each candidate
    mix=[]
    for name, d in CANDIDATES:
        subset=[r for r in all_csv if r['candidate']==name][:6]
        mix.extend(subset)
    render_candidate_sheet('MIXED final94 boardwarp compare', mix, OUT/'00_mixed_final94_boardwarp_compare.jpg')
    with (OUT/'audit_rows.csv').open('w', encoding='utf-8', newline='') as f:
        fields=['candidate','source_dir','path','text','occ','cx_norm','cy_norm','occ_w','occ_h','margin_l','margin_r','margin_t','margin_b','margin_lr_diff','margin_tb_diff','horiz_deg','vert_dev_deg','horiz_abs','vert_abs','rise_sign','horiz_sign','warp_w','warp_h']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader();
        for r in all_csv:
            w.writerow({k:r.get(k,'') for k in fields})
    summary={'out_dir':str(OUT),'windows_out_dir':str(WIN),'candidates':summaries}
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    html=['<!doctype html><html><head><meta charset="utf-8"><title>StageB1A next extreme candidate audit</title><style>body{font-family:sans-serif;background:#eee} img{display:block;max-width:100%;margin:18px auto;border:1px solid #999} pre{background:white;padding:12px}</style></head><body>']
    html.append('<h1>StageB1A next extreme candidate audit</h1>')
    html.append('<pre>'+json.dumps(summary, ensure_ascii=False, indent=2)+'</pre>')
    for p in ['00_mixed_final94_boardwarp_compare.jpg']+[Path(x['sheet']).name for x in summaries]:
        html.append(f'<h2>{p}</h2><img src="{p}">')
    html.append('</body></html>')
    (OUT/'index.html').write_text('\n'.join(html), encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file(): shutil.copy2(p, WIN/p.name)
    print(json.dumps({'out_dir':str(OUT),'windows_out_dir':str(WIN),'rows':len(all_csv)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
