#!/usr/bin/env python3
"""Audit multiple green extreme candidate directories with board/ccpd warp.

This is a 2026-04-26 refresh focused on the user's question:
StageB1A extreme was off-center / not the newest best; compare older tier3/v4
and the later accepted moderate LMH source in the actual boardwarp final94 view.
"""
import csv
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
OUT = ROOT / 'reports' / 'stageB1A_extreme_multi_source_boardwarp_QA_20260426'
WIN = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/stageB1A_extreme_multi_source_boardwarp_QA_20260426')
OUT.mkdir(parents=True, exist_ok=True)
WIN.mkdir(parents=True, exist_ok=True)

for p in [ROOT / 'src']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name  # noqa: E402

FONT_PATH = next(p for p in [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
] if Path(p).exists())
FONT = ImageFont.truetype(FONT_PATH, 18)
SMALL = ImageFont.truetype(FONT_PATH, 13)
TINY = ImageFont.truetype(FONT_PATH, 10)

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
PROV_ORDER = ['京','津','沪','渝','冀','晋','蒙','辽','吉','黑','苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新']

CANDIDATES = [
    ('01_B1A_USED_tier3_full_v2_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v2/images/train/extreme'),
    ('02_tier3_full_v1_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v1/images/train/extreme'),
    ('03_tier3_full_v3_su_conservative_train_extreme', ROOT/'datasets/green_edgefit_tier3_full_v3_su_conservative/images/train/extreme'),
    ('04_v4_e2_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e2_20260411/images/train/board_extreme_tail'),
    ('05_v4_e3_equalprov_a_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e3_equalprov_a_20260412/images/train/board_extreme_tail'),
    ('06_v4_e4_extreme_append10_board_extreme_tail', ROOT/'tmp/green_edgefit_v4_e4_extreme_append10_20260412/images/train/board_extreme_tail'),
    ('07_E1_ACCEPTED_moderate_lmh_train_low', ROOT/'tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/train/low'),
    ('08_E1_ACCEPTED_moderate_lmh_train_mid', ROOT/'tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/train/mid'),
    ('09_E1_ACCEPTED_moderate_lmh_train_high', ROOT/'tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/train/high'),
    ('10_E1_ACCEPTED_moderate_lmh_proxy_all', ROOT/'tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426/images/proxy'),
]

TEXT_RE = re.compile(r'([\u4e00-\u9fa5][A-Z0-9]{6,7})')

def collect_images(d):
    if not d.exists():
        return []
    return sorted([p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT], key=lambda x: str(x))

def infer_text(path):
    stem = path.stem
    matches = TEXT_RE.findall(stem)
    return matches[-1] if matches else ''

def parse_quad(path):
    q = parse_ccpd_quad_from_name(path.name)
    if q is not None:
        return np.asarray(q, dtype=np.float32)
    # E1mod filenames use: prefix-bbox-quad-split-tier-dir-prov-id-text.jpg
    m = re.search(r'\d+&\d+_\d+&\d+-(\d+&\d+_\d+&\d+_\d+&\d+_\d+&\d+)', path.name)
    if m:
        pts = []
        for tok in m.group(1).split('_'):
            x, y = tok.split('&')
            pts.append((float(x), float(y)))
        return np.asarray(pts, dtype=np.float32)
    return None

def nonblack_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 8)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def shape_direction(path):
    s = path.name
    for token in ['left_up','left_mid','left_down','right_up','right_mid','right_down','mid_up','mid_down']:
        if token in s:
            return token
    return ''

def quad_metrics(q):
    tl, tr, br, bl = q
    top = tr - tl
    bottom = br - bl
    left = bl - tl
    right = br - tr
    horiz_deg = math.degrees(math.atan2(float(top[1]), float(top[0])))
    bottom_deg = math.degrees(math.atan2(float(bottom[1]), float(bottom[0])))
    left_dev = math.degrees(math.atan2(float(left[0]), float(left[1])))
    right_dev = math.degrees(math.atan2(float(right[0]), float(right[1])))
    top_mid = (tl + tr) / 2.0
    bot_mid = (bl + br) / 2.0
    rise = float(top_mid[1] - bot_mid[1])
    return {
        'horiz_deg': horiz_deg,
        'bottom_deg': bottom_deg,
        'vert_dev_deg': left_dev,
        'right_dev_deg': right_dev,
        'horiz_abs': abs(horiz_deg),
        'vert_abs': abs(left_dev),
        'rise': rise,
        'rise_sign': 'up' if rise < -1e-3 else ('down' if rise > 1e-3 else 'flat'),
        'horiz_sign': 'ccw' if horiz_deg < -1e-3 else ('cw' if horiz_deg > 1e-3 else 'flat'),
        'quad_cx': float(q[:,0].mean()),
        'quad_cy': float(q[:,1].mean()),
    }

def measure(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    q = parse_quad(path)
    if q is None:
        return None
    prep, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)
    bb = nonblack_bbox(prep)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    return {
        'path': str(path),
        'file': path.name,
        'text': infer_text(path),
        'province': infer_text(path)[:1],
        'shape_dir': shape_direction(path),
        'image_w': int(img.shape[1]),
        'image_h': int(img.shape[0]),
        'quad': q.tolist(),
        'occ': float(occ),
        'warp_w': int(warped.shape[1]),
        'warp_h': int(warped.shape[0]),
        'bbox': [x1, y1, x2, y2],
        'cx_norm': float((x1+x2+1)/2/94),
        'cy_norm': float((y1+y2+1)/2/24),
        'occ_w': float((x2-x1+1)/94),
        'occ_h': float((y2-y1+1)/24),
        'margin_l': int(x1),
        'margin_r': int(93-x2),
        'margin_t': int(y1),
        'margin_b': int(23-y2),
        'margin_lr_diff': int(abs(x1-(93-x2))),
        'margin_tb_diff': int(abs(y1-(23-y2))),
        **quad_metrics(q),
    }

def pick_samples(rows, n=24, seed=20260426):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        # Prefer spread by tier/direction/province. Filename contains low/mid/high and left/right/up/down.
        key = (r['rise_sign'], r['horiz_sign'], r.get('province') or '')
        buckets[key].append(r)
    for arr in buckets.values():
        arr.sort(key=lambda r: r['path'])
        rng.shuffle(arr)
    keys = sorted(buckets.keys(), key=lambda k: (k[0], k[1], PROV_ORDER.index(k[2]) if k[2] in PROV_ORDER else 99, k[2]))
    out = []
    while len(out) < n and any(buckets.values()):
        progressed = False
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
                progressed = True
        if not progressed:
            break
    return out

def pil_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def fit_pil_bgr(img, size):
    im = pil_bgr(img)
    w, h = im.size
    scale = min(size[0]/max(1,w), size[1]/max(1,h))
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    can = Image.new('RGB', size, (245,245,245))
    can.paste(im, ((size[0]-nw)//2, (size[1]-nh)//2))
    return can, scale, ((size[0]-nw)//2, (size[1]-nh)//2)

def draw_quad(draw, q, src_shape, origin, box_size):
    h, w = src_shape[:2]
    scale = min(box_size[0]/max(1,w), box_size[1]/max(1,h))
    offx = origin[0] + (box_size[0]-int(w*scale))//2
    offy = origin[1] + (box_size[1]-int(h*scale))//2
    pts = [(offx + float(x)*scale, offy + float(y)*scale) for x, y in q]
    draw.line(pts + [pts[0]], fill=(255,0,0), width=2)
    for i, (x,y) in enumerate(pts):
        draw.ellipse([x-3,y-3,x+3,y+3], fill=(255,0,0))
        draw.text((x+3,y+1), str(i+1), font=TINY, fill=(255,0,0))

def render_sheet(name, rows, out_path, mode='three_panel'):
    cols = 3
    if mode == 'final94_only':
        cell_w, cell_h, title_h = 340, 148, 78
    else:
        cell_w, cell_h, title_h = 360, 270, 84
    can = Image.new('RGB', (cols*cell_w, title_h + math.ceil(len(rows)/cols)*cell_h), (255,255,255))
    d = ImageDraw.Draw(can)
    d.text((10,8), name, font=FONT, fill=(0,0,0))
    if mode == 'final94_only':
        d.text((10,34), '最终板端/训练口径 final94 gray3 输入；红字提示贴边/偏心/方向。', font=SMALL, fill=(60,60,60))
    else:
        d.text((10,34), '上: 原图+quad；中: warpPerspective 后；下: prepare_board_ocr_input_from_quad_bgr888 final94 gray3。', font=SMALL, fill=(60,60,60))
    d.text((10,56), f'font={FONT_PATH}', font=TINY, fill=(90,90,90))
    for i, r in enumerate(rows):
        x = (i % cols) * cell_w
        y = title_h + (i // cols) * cell_h
        d.rectangle([x,y,x+cell_w-1,y+cell_h-1], outline=(200,200,200))
        img = cv2.imread(r['path'])
        q = np.asarray(r['quad'], dtype=np.float32)
        prep, _, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
            img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)
        if mode == 'final94_only':
            prep_big = cv2.resize(prep, (235, 60), interpolation=cv2.INTER_NEAREST)
            can.paste(pil_bgr(prep_big), (x+8, y+8))
            tx = x + 250
            ty = y + 8
        else:
            crop_box = (340, 76)
            crop, _, _ = fit_pil_bgr(img, crop_box)
            can.paste(crop, (x+8,y+8))
            draw_quad(d, q, img.shape, (x+8,y+8), crop_box)
            warp_big = cv2.resize(warped, (250, 58), interpolation=cv2.INTER_NEAREST)
            prep_big = cv2.resize(prep, (250, 64), interpolation=cv2.INTER_NEAREST)
            can.paste(pil_bgr(warp_big), (x+8, y+90))
            can.paste(pil_bgr(prep_big), (x+8, y+158))
            tx = x + 265
            ty = y + 90
        offcenter = abs(r['cx_norm'] - 0.5) > 0.08 or min(r['margin_l'], r['margin_r']) <= 1
        d.text((tx,ty), r['text'], font=SMALL, fill=(0,0,0))
        d.text((tx,ty+18), f"occ={r['occ']:.2f}", font=TINY, fill=(0,100,0))
        d.text((tx,ty+32), f"cx={r['cx_norm']:.2f} cy={r['cy_norm']:.2f}", font=TINY, fill=(180,0,0) if offcenter else (0,0,0))
        d.text((tx,ty+46), f"L/R={r['margin_l']}/{r['margin_r']}", font=TINY, fill=(180,0,0) if min(r['margin_l'], r['margin_r'])<=1 else (0,0,0))
        d.text((tx,ty+60), f"h={r['horiz_deg']:.1f} v={r['vert_dev_deg']:.1f}", font=TINY, fill=(0,0,150))
        d.text((tx,ty+74), f"{r['rise_sign']}/{r['horiz_sign']}", font=TINY, fill=(120,0,120))
        if mode != 'final94_only':
            d.text((x+8,y+230), Path(r['path']).name[:56], font=TINY, fill=(80,80,80))
        else:
            d.text((x+8,y+108), Path(r['path']).name[:54], font=TINY, fill=(80,80,80))
    can.save(out_path, quality=92)

def summarize_numeric(vals):
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        'mean': float(a.mean()),
        'p50': float(np.percentile(a,50)),
        'p90': float(np.percentile(a,90)),
        'p95': float(np.percentile(a,95)),
        'min': float(a.min()),
        'max': float(a.max()),
    }

def summarize_rows(rows):
    numeric_keys = ['occ','cx_norm','cy_norm','occ_w','occ_h','margin_l','margin_r','margin_t','margin_b','margin_lr_diff','margin_tb_diff','horiz_abs','vert_abs','warp_w','warp_h']
    out = {'count_measured': len(rows)}
    out['direction_counts'] = dict(Counter(f"rise:{r['rise_sign']}|rot:{r['horiz_sign']}" for r in rows).most_common())
    out['shape_dir_counts'] = dict(Counter(r.get('shape_dir') or 'unknown' for r in rows).most_common())
    out['province_counts_top'] = dict(Counter(r.get('province') or '' for r in rows).most_common(12))
    out['numeric'] = {k: summarize_numeric([r[k] for r in rows if k in r]) for k in numeric_keys}
    out['offcenter_ratio_abs_cx_gt_008'] = sum(abs(r['cx_norm']-0.5)>0.08 for r in rows)/len(rows) if rows else 0
    out['touch_lr_ratio'] = sum(min(r['margin_l'], r['margin_r'])<=1 for r in rows)/len(rows) if rows else 0
    return out

def write_report(summary):
    lines = []
    lines.append('# GREEN StageB1A Extreme Multi-source Boardwarp QA Report')
    lines.append('')
    lines.append('日期：2026-04-26')
    lines.append('')
    lines.append('## 目的')
    lines.append('StageB1A 已训练完成，但 extreme 表现不佳；用户指出 extreme 不在中心，并怀疑训练/评估使用的 extreme 不是最新最好版本。本报告只做数据源 QA，不启动训练。')
    lines.append('')
    lines.append('## 口径')
    lines.append('- 对每个候选目录解析文件名 quad。')
    lines.append("- 使用 `prepare_board_ocr_input_from_quad_bgr888(img, quad, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)` 生成与训练/板端一致的 final94 gray3 预览。")
    lines.append('- 同时输出三栏图：原图+quad / warp 后 / final94，以及 final94-only 对照图。')
    lines.append('- QA 图已复制到 Windows OneDrive QA 目录，供人工肉眼审查。')
    lines.append('')
    lines.append('## 交付路径')
    lines.append(f"- WSL: `{summary['out_dir']}`")
    lines.append(f"- Windows: `{summary['windows_out_dir']}`")
    lines.append(f"- HTML: `{summary['windows_out_dir']}/index.html`")
    lines.append('')
    lines.append('## 候选目录与统计')
    for c in summary['candidates']:
        lines.append(f"### {c['name']}")
        lines.append(f"- dir: `{c['dir']}`")
        lines.append(f"- exists: {c['exists']}  images: {c['count_images']}  measured: {c['stats']['count_measured'] if c.get('stats') else 0}")
        if c.get('stats') and c['stats'].get('count_measured', 0) > 0:
            st = c['stats']
            lines.append(f"- direction_counts: `{st.get('direction_counts', {})}`")
            lines.append(f"- shape_dir_counts: `{st.get('shape_dir_counts', {})}`")
            for k in ['cx_norm','occ_w','margin_l','margin_r','horiz_abs','vert_abs']:
                lines.append(f"- {k}: `{json.dumps(st['numeric'].get(k), ensure_ascii=False)}`")
            lines.append(f"- offcenter_ratio_abs_cx_gt_0.08: {st['offcenter_ratio_abs_cx_gt_008']:.4f}")
            lines.append(f"- touch_lr_ratio: {st['touch_lr_ratio']:.4f}")
            lines.append(f"- sheet_three_panel: `{c['sheet_three_panel']}`")
            lines.append(f"- sheet_final94: `{c['sheet_final94']}`")
        lines.append('')
    lines.append('## 初步判断')
    lines.append('- 这份 QA 的有效结论只限于数据源几何/最终输入形态；不对训练效果下结论。')
    lines.append('- 后续是否替换训练源，需要以用户人工确认的 QA 图为准，并继续保持 old proxy / new proxy 口径分离。')
    lines.append('')
    p = ROOT / 'reports' / 'GREEN_STAGEB1A_EXTREME_MULTI_SOURCE_BOARDWARP_QA_REPORT.md'
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copy2(p, WIN / p.name)
    return str(p)

def main():
    all_rows = []
    candidates = []
    mixed = []
    for idx, (name, d) in enumerate(CANDIDATES):
        paths = collect_images(d)
        rows = []
        for p in paths:
            m = measure(p)
            if m is not None:
                m['candidate'] = name
                m['source_dir'] = str(d)
                rows.append(m)
                all_rows.append(m)
        sampled = pick_samples(rows, 24, seed=20260426+idx)
        mixed.extend(sampled[:3])
        sheet3 = OUT / f'{name}_three_panel_boardwarp.jpg'
        sheet94 = OUT / f'{name}_final94_only.jpg'
        if sampled:
            render_sheet(name, sampled, sheet3, mode='three_panel')
            render_sheet(name, sampled, sheet94, mode='final94_only')
        candidates.append({
            'name': name,
            'dir': str(d),
            'exists': d.exists(),
            'count_images': len(paths),
            'stats': summarize_rows(rows) if rows else {'count_measured': 0},
            'sheet_three_panel': str(sheet3) if sampled else '',
            'sheet_final94': str(sheet94) if sampled else '',
        })
    if mixed:
        render_sheet('00_MIXED_three_samples_each_candidate_boardwarp', mixed, OUT/'00_mixed_three_panel_boardwarp.jpg', mode='three_panel')
        render_sheet('00_MIXED_final94_only_compare', mixed, OUT/'00_mixed_final94_only_compare.jpg', mode='final94_only')
    with (OUT/'audit_rows.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['candidate','source_dir','path','file','text','province','shape_dir','image_w','image_h','occ','cx_norm','cy_norm','occ_w','occ_h','margin_l','margin_r','margin_t','margin_b','margin_lr_diff','margin_tb_diff','horiz_deg','vert_dev_deg','horiz_abs','vert_abs','rise_sign','horiz_sign','warp_w','warp_h']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, '') for k in fields})
    summary = {'out_dir': str(OUT), 'windows_out_dir': str(WIN), 'font_path': FONT_PATH, 'candidates': candidates, 'total_rows': len(all_rows)}
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = write_report(summary)
    summary['report'] = report
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>StageB1A extreme multi-source boardwarp QA</title><style>body{font-family:sans-serif;background:#eee} img{display:block;max-width:100%;margin:18px auto;border:1px solid #999} pre{background:white;padding:12px;white-space:pre-wrap}</style></head><body>']
    html.append('<h1>StageB1A extreme multi-source boardwarp QA 20260426</h1>')
    html.append('<pre>'+json.dumps(summary, ensure_ascii=False, indent=2)+'</pre>')
    for img in ['00_mixed_three_panel_boardwarp.jpg','00_mixed_final94_only_compare.jpg']:
        if (OUT/img).exists():
            html.append(f'<h2>{img}</h2><img src="{img}">')
    for c in candidates:
        for key in ['sheet_three_panel','sheet_final94']:
            if c.get(key):
                img = Path(c[key]).name
                html.append(f"<h2>{img}</h2><img src=\"{img}\">")
    html.append('</body></html>')
    (OUT/'index.html').write_text('\n'.join(html), encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():
            shutil.copy2(p, WIN/p.name)
    print(json.dumps({'out_dir': str(OUT), 'windows_out_dir': str(WIN), 'report': report, 'total_rows': len(all_rows), 'num_candidates': len(candidates)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
