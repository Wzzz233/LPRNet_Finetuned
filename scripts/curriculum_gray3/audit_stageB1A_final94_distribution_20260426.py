#!/usr/bin/env python3
"""Final94 distribution audit for StageB1A accepted moderate domain.

No training. Compare final94 board/manifest input distributions:
- old proxy extreme
- E1/E3 train accepted low/mid/high (same accepted manifest if E3 did not create a new manifest)
- new accepted proxy low/mid/high
- green_edgefit_hard
- real green CCPD2020
"""
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
OUT = ROOT / 'reports' / 'stageB1A_final94_distribution_audit_20260426'
WIN = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/stageB1A_final94_distribution_audit_20260426')
OUT.mkdir(parents=True, exist_ok=True)
WIN.mkdir(parents=True, exist_ok=True)

for p in [ROOT/'src']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from load_data import prepare_board_ocr_input_from_quad_bgr888  # noqa: E402

FONT_PATH = next(p for p in [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
] if Path(p).exists())
FONT = ImageFont.truetype(FONT_PATH, 18)
SMALL = ImageFont.truetype(FONT_PATH, 13)
TINY = ImageFont.truetype(FONT_PATH, 10)

MAN_E1 = ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original'
MAN_NEW = ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy'

DATASETS = [
    {
        'name': 'old_proxy_extreme',
        'kind': 'manifest',
        'manifest': MAN_E1/'proxy_green_edgefit_extreme.csv',
        'filter': lambda r: True,
        'note': 'Original StageB1A old proxy benchmark extreme, unchanged in E1/E3 eval-original dir.',
    },
    {
        'name': 'train_accepted_low',
        'kind': 'manifest',
        'manifest': MAN_E1/'train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv',
        'filter': lambda r: r.get('source') == 'green_edgefit_extreme_E1_moderate_lmh_ccpdboard' and r.get('difficulty_tier') == 'low',
        'note': 'E1/E3 train accepted low tier. E3 reuses E1 accepted manifest if no separate E3 manifest exists.',
    },
    {
        'name': 'train_accepted_mid',
        'kind': 'manifest',
        'manifest': MAN_E1/'train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv',
        'filter': lambda r: r.get('source') == 'green_edgefit_extreme_E1_moderate_lmh_ccpdboard' and r.get('difficulty_tier') == 'mid',
        'note': 'E1/E3 train accepted mid tier.',
    },
    {
        'name': 'train_accepted_high',
        'kind': 'manifest',
        'manifest': MAN_E1/'train_B1A_E1_moderate_lmh_ccpdboard_eval_original.csv',
        'filter': lambda r: r.get('source') == 'green_edgefit_extreme_E1_moderate_lmh_ccpdboard' and r.get('difficulty_tier') == 'high',
        'note': 'E1/E3 train accepted high tier.',
    },
    {
        'name': 'new_proxy_accepted_low',
        'kind': 'manifest',
        'manifest': MAN_NEW/'proxy_green_edgefit_extreme.csv',
        'filter': lambda r: r.get('difficulty_tier') == 'low',
        'note': 'New accepted proxy low tier.',
    },
    {
        'name': 'new_proxy_accepted_mid',
        'kind': 'manifest',
        'manifest': MAN_NEW/'proxy_green_edgefit_extreme.csv',
        'filter': lambda r: r.get('difficulty_tier') == 'mid',
        'note': 'New accepted proxy mid tier.',
    },
    {
        'name': 'new_proxy_accepted_high',
        'kind': 'manifest',
        'manifest': MAN_NEW/'proxy_green_edgefit_extreme.csv',
        'filter': lambda r: r.get('difficulty_tier') == 'high',
        'note': 'New accepted proxy high tier.',
    },
    {
        'name': 'green_edgefit_hard',
        'kind': 'manifest',
        'manifest': MAN_E1/'proxy_green_edgefit_hard.csv',
        'filter': lambda r: True,
        'note': 'Synthetic hard proxy from the same eval manifest dir.',
    },
    {
        'name': 'real_green_CCPD2020',
        'kind': 'manifest',
        'manifest': MAN_E1/'proxy_green_ccpd2020_real.csv',
        'filter': lambda r: True,
        'note': 'Real CCPD2020 green proxy from the same eval manifest dir.',
    },
]

METRIC_KEYS = [
    'mean','std','min','p05','p10','p25','p50','p75','p90','p95','max'
]
NUMERIC = [
    'fg_mean','fg_std','fg_p10','fg_p50','fg_p90','fg_contrast','edge_density','sobel_mean',
    'cx_norm','cy_norm','occ_w','occ_h','margin_l','margin_r','margin_t','margin_b','margin_lr_diff','margin_tb_diff',
    'black_ratio','white_ratio','mid_ratio','dark_border_ratio','warp_w','warp_h','quad_horiz_abs','quad_vert_abs','quad_area_norm'
]


def read_rows(manifest, filter_fn):
    rows = []
    with Path(manifest).open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if filter_fn(r):
                rows.append(r)
    return rows


def fnum(v, default=None):
    try:
        if v is None or v == '':
            return default
        return float(v)
    except Exception:
        return default


def get_quad(row):
    pts = []
    for i in range(1,5):
        x = fnum(row.get(f'quad_{i}x'))
        y = fnum(row.get(f'quad_{i}y'))
        if x is None or y is None:
            return None
        pts.append([x,y])
    return np.asarray(pts, dtype=np.float32)


def final94(row):
    path = row.get('img_path') or row.get('path')
    if not path:
        return None, None, 'missing_path'
    img = cv2.imread(path)
    if img is None:
        return None, None, 'imread_failed'
    q = get_quad(row)
    # Most rows here are ccpd_board/obb_warp. Fallback to direct resize only if no quad exists.
    if q is not None:
        try:
            prep, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
                img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=fnum(row.get('ocr_quad_pad_ratio'), 0.0) or 0.0)
            meta = {'occ': float(occ), 'warp_w': int(warped.shape[1]), 'warp_h': int(warped.shape[0]), **quad_shape_metrics(q, img.shape)}
            return prep, meta, ''
        except Exception as e:
            return None, None, f'warp_failed:{type(e).__name__}'
    # direct fallback, labeled; should not dominate this audit.
    r = cv2.resize(img, (94,24), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    prep = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return prep, {'occ': None, 'warp_w': 94, 'warp_h': 24, 'quad_horiz_abs': None, 'quad_vert_abs': None, 'quad_area_norm': None}, 'no_quad_direct_resize'


def quad_shape_metrics(q, shape):
    tl,tr,br,bl = q
    top = tr - tl
    left = bl - tl
    horiz = abs(math.degrees(math.atan2(float(top[1]), float(top[0]))))
    vert = abs(math.degrees(math.atan2(float(left[0]), float(left[1]))))
    area = abs(cv2.contourArea(q.astype(np.float32)))
    h,w = shape[:2]
    return {'quad_horiz_abs': horiz, 'quad_vert_abs': vert, 'quad_area_norm': float(area / max(1.0, w*h))}


def measure_image(prep, meta):
    gray = cv2.cvtColor(prep, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    ys, xs = np.where(gray > 8)
    if len(xs) == 0:
        x1=y1=0; x2=w-1; y2=h-1
        roi = gray
    else:
        x1,y1,x2,y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        roi = gray[y1:y2+1, x1:x2+1]
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    edges = cv2.Canny(gray, 40, 120)
    border = np.concatenate([gray[0,:], gray[-1,:], gray[:,0], gray[:,-1]])
    out = {
        'fg_mean': float(roi.mean()),
        'fg_std': float(roi.std()),
        'fg_p10': float(np.percentile(roi,10)),
        'fg_p50': float(np.percentile(roi,50)),
        'fg_p90': float(np.percentile(roi,90)),
        'fg_contrast': float(np.percentile(roi,90) - np.percentile(roi,10)),
        'edge_density': float((edges > 0).mean()),
        'sobel_mean': float(mag.mean()),
        'cx_norm': float((x1+x2+1)/2/w),
        'cy_norm': float((y1+y2+1)/2/h),
        'occ_w': float((x2-x1+1)/w),
        'occ_h': float((y2-y1+1)/h),
        'margin_l': float(x1),
        'margin_r': float((w-1)-x2),
        'margin_t': float(y1),
        'margin_b': float((h-1)-y2),
        'margin_lr_diff': float(abs(x1-((w-1)-x2))),
        'margin_tb_diff': float(abs(y1-((h-1)-y2))),
        'black_ratio': float((gray <= 8).mean()),
        'white_ratio': float((gray >= 245).mean()),
        'mid_ratio': float(((gray > 50) & (gray < 210)).mean()),
        'dark_border_ratio': float((border <= 8).mean()),
        'warp_w': float(meta.get('warp_w') or 0),
        'warp_h': float(meta.get('warp_h') or 0),
        'quad_horiz_abs': meta.get('quad_horiz_abs'),
        'quad_vert_abs': meta.get('quad_vert_abs'),
        'quad_area_norm': meta.get('quad_area_norm'),
    }
    return out


def summarize_values(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return {k: None for k in METRIC_KEYS}
    a = np.asarray(vals, dtype=np.float64)
    return {
        'mean': float(a.mean()),
        'std': float(a.std()),
        'min': float(a.min()),
        'p05': float(np.percentile(a,5)),
        'p10': float(np.percentile(a,10)),
        'p25': float(np.percentile(a,25)),
        'p50': float(np.percentile(a,50)),
        'p75': float(np.percentile(a,75)),
        'p90': float(np.percentile(a,90)),
        'p95': float(np.percentile(a,95)),
        'max': float(a.max()),
    }


def wasserstein_1d(a, b):
    a = np.asarray([x for x in a if x is not None and np.isfinite(float(x))], dtype=np.float64)
    b = np.asarray([x for x in b if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return None
    n = max(len(a), len(b))
    qa = np.quantile(a, np.linspace(0,1,n))
    qb = np.quantile(b, np.linspace(0,1,n))
    return float(np.mean(np.abs(qa - qb)))


def z_gap(group_mean, ref_mean, ref_std):
    if group_mean is None or ref_mean is None or ref_std is None or ref_std < 1e-9:
        return None
    return float((group_mean - ref_mean) / ref_std)


def pil_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def render_sheet(name, samples, out_path):
    cols, cell_w, cell_h, title_h = 4, 300, 132, 72
    can = Image.new('RGB', (cols*cell_w, title_h+math.ceil(len(samples)/cols)*cell_h), (255,255,255))
    d = ImageDraw.Draw(can)
    d.text((10,8), name, font=FONT, fill=(0,0,0))
    d.text((10,34), 'final94 gray3 only; metrics: brightness/contrast/occ/margins/edge density', font=SMALL, fill=(60,60,60))
    d.text((10,52), f'font={FONT_PATH}', font=TINY, fill=(80,80,80))
    for i, rec in enumerate(samples):
        x=(i%cols)*cell_w; y=title_h+(i//cols)*cell_h
        d.rectangle([x,y,x+cell_w-1,y+cell_h-1], outline=(200,200,200))
        big = cv2.resize(rec['prep'], (220,56), interpolation=cv2.INTER_NEAREST)
        can.paste(pil_bgr(big), (x+8,y+8))
        m=rec['metrics']
        d.text((x+8,y+68), f"{rec.get('text','')} {rec.get('tier','')}", font=TINY, fill=(0,0,0))
        d.text((x+8,y+82), f"mean={m['fg_mean']:.1f} std={m['fg_std']:.1f} ed={m['edge_density']:.3f}", font=TINY, fill=(0,0,120))
        d.text((x+8,y+96), f"occ={m['occ_w']:.2f}/{m['occ_h']:.2f} cx={m['cx_norm']:.2f}", font=TINY, fill=(0,100,0))
        d.text((x+8,y+110), f"L/R/T/B={m['margin_l']:.0f}/{m['margin_r']:.0f}/{m['margin_t']:.0f}/{m['margin_b']:.0f}", font=TINY, fill=(120,0,0))
    can.save(out_path, quality=92)


def render_boxplot(summary, out_path):
    # Simple visual: selected metrics, p10-p90 box with p50 line, mean dot. No matplotlib dependency.
    metrics = ['fg_mean','fg_std','fg_contrast','edge_density','occ_w','occ_h','margin_l','margin_r','quad_horiz_abs','quad_vert_abs']
    groups = list(summary['groups'].keys())
    left_w, row_h, col_w, top_h = 230, 30, 175, 80
    can = Image.new('RGB', (left_w+len(metrics)*col_w, top_h+len(groups)*row_h), (255,255,255))
    d = ImageDraw.Draw(can)
    d.text((10,8), 'final94 distribution audit: p10-p90 box, p50 tick, mean dot', font=FONT, fill=(0,0,0))
    d.text((10,34), 'Each metric scaled independently across groups. See summary.json for exact values.', font=SMALL, fill=(60,60,60))
    for j,m in enumerate(metrics):
        d.text((left_w+j*col_w+4,56), m, font=TINY, fill=(0,0,0))
        vals=[]
        for g in groups:
            s=summary['groups'][g]['metrics'].get(m,{})
            for k in ['p10','p90','mean','p50']:
                if s.get(k) is not None: vals.append(float(s[k]))
        mn=min(vals) if vals else 0; mx=max(vals) if vals else 1
        if mx-mn < 1e-9: mx=mn+1
        for i,g in enumerate(groups):
            y=top_h+i*row_h
            s=summary['groups'][g]['metrics'].get(m,{})
            def sx(v): return left_w+j*col_w+10+int((float(v)-mn)/(mx-mn)*(col_w-28))
            d.line([left_w+j*col_w+10,y+row_h//2,left_w+(j+1)*col_w-18,y+row_h//2], fill=(230,230,230))
            if s.get('p10') is not None:
                x1=sx(s['p10']); x2=sx(s['p90']); xm=sx(s['p50']); xmean=sx(s['mean'])
                d.rectangle([x1,y+8,x2,y+row_h-8], outline=(0,90,180), fill=(210,235,255))
                d.line([xm,y+6,xm,y+row_h-6], fill=(0,0,0), width=1)
                d.ellipse([xmean-3,y+row_h//2-3,xmean+3,y+row_h//2+3], fill=(180,0,0))
    for i,g in enumerate(groups):
        y=top_h+i*row_h
        d.text((8,y+8), g[:35], font=TINY, fill=(0,0,0))
        d.line([0,y,left_w+len(metrics)*col_w,y], fill=(235,235,235))
    can.save(out_path, quality=92)


def pick_samples(records, n=24, seed=20260426):
    rng = random.Random(seed)
    arr = records[:]
    rng.shuffle(arr)
    return arr[:n]


def main():
    all_records = []
    group_records = {}
    errors = {}
    for ds in DATASETS:
        rows = read_rows(ds['manifest'], ds['filter'])
        recs = []
        err = Counter()
        for row in rows:
            prep, meta, e = final94(row)
            if prep is None:
                err[e or 'unknown'] += 1
                continue
            metrics = measure_image(prep, meta or {})
            rec = {
                'group': ds['name'],
                'img_path': row.get('img_path',''),
                'text': row.get('text',''),
                'tier': row.get('difficulty_tier',''),
                'source': row.get('source',''),
                'preprocess_group': row.get('preprocess_group',''),
                'error_note': e,
                'metrics': metrics,
                'prep': prep,
            }
            recs.append(rec)
            all_records.append(rec)
        group_records[ds['name']] = recs
        errors[ds['name']] = {'input_rows': len(rows), 'usable': len(recs), 'errors': dict(err), 'manifest': str(ds['manifest']), 'note': ds['note']}

    groups_summary = {}
    for name, recs in group_records.items():
        groups_summary[name] = {
            **errors[name],
            'metrics': {k: summarize_values([r['metrics'].get(k) for r in recs]) for k in NUMERIC},
            'source_counts_top': dict(Counter(r['source'] for r in recs).most_common(8)),
            'preprocess_counts': dict(Counter(r['preprocess_group'] for r in recs)),
        }

    ref_hard = group_records.get('green_edgefit_hard', [])
    ref_real = group_records.get('real_green_CCPD2020', [])
    distances = {}
    for name, recs in group_records.items():
        distances[name] = {'to_green_edgefit_hard': {}, 'to_real_green_CCPD2020': {}}
        for k in NUMERIC:
            vals = [r['metrics'].get(k) for r in recs]
            hv = [r['metrics'].get(k) for r in ref_hard]
            rv = [r['metrics'].get(k) for r in ref_real]
            distances[name]['to_green_edgefit_hard'][k] = {
                'wasserstein': wasserstein_1d(vals, hv),
                'mean_gap': None,
                'z_gap_vs_ref': None,
            }
            distances[name]['to_real_green_CCPD2020'][k] = {
                'wasserstein': wasserstein_1d(vals, rv),
                'mean_gap': None,
                'z_gap_vs_ref': None,
            }
            gm = groups_summary[name]['metrics'][k]['mean']
            hm = groups_summary['green_edgefit_hard']['metrics'][k]['mean'] if 'green_edgefit_hard' in groups_summary else None
            hs = groups_summary['green_edgefit_hard']['metrics'][k]['std'] if 'green_edgefit_hard' in groups_summary else None
            rm = groups_summary['real_green_CCPD2020']['metrics'][k]['mean'] if 'real_green_CCPD2020' in groups_summary else None
            rs = groups_summary['real_green_CCPD2020']['metrics'][k]['std'] if 'real_green_CCPD2020' in groups_summary else None
            if gm is not None and hm is not None:
                distances[name]['to_green_edgefit_hard'][k]['mean_gap'] = float(gm-hm)
                distances[name]['to_green_edgefit_hard'][k]['z_gap_vs_ref'] = z_gap(gm, hm, hs)
            if gm is not None and rm is not None:
                distances[name]['to_real_green_CCPD2020'][k]['mean_gap'] = float(gm-rm)
                distances[name]['to_real_green_CCPD2020'][k]['z_gap_vs_ref'] = z_gap(gm, rm, rs)

    summary = {
        'out_dir': str(OUT),
        'windows_out_dir': str(WIN),
        'font_path': FONT_PATH,
        'groups': groups_summary,
        'distances': distances,
        'metric_definitions': {
            'fg_*': 'stats over non-black bbox in final94 gray image',
            'edge_density': 'Canny(40,120) edge pixel ratio over final94',
            'occ/margins/cx': 'non-black bbox occupancy and placement in 94x24 final input',
            'quad_*': 'source quad geometry before warp, when available',
            'wasserstein': 'mean absolute quantile distance, same metric units',
            'z_gap_vs_ref': 'mean gap divided by reference std',
        },
    }

    # CSV: one row per sample metrics.
    with (OUT/'final94_sample_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['group','img_path','text','tier','source','preprocess_group','error_note'] + NUMERIC
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_records:
            row = {k: r.get(k,'') for k in ['group','img_path','text','tier','source','preprocess_group','error_note']}
            for k in NUMERIC:
                row[k] = r['metrics'].get(k)
            w.writerow(row)

    # CSV: compact group summary.
    with (OUT/'final94_group_summary.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['group','input_rows','usable','metric','mean','std','p10','p50','p90','min','max','gap_to_hard_mean','gap_to_real_mean','z_to_hard','z_to_real','w1_to_hard','w1_to_real']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g, gs in groups_summary.items():
            for k in NUMERIC:
                m = gs['metrics'][k]
                row = {'group': g, 'input_rows': gs['input_rows'], 'usable': gs['usable'], 'metric': k,
                       'mean': m['mean'], 'std': m['std'], 'p10': m['p10'], 'p50': m['p50'], 'p90': m['p90'], 'min': m['min'], 'max': m['max']}
                row['gap_to_hard_mean'] = distances[g]['to_green_edgefit_hard'][k]['mean_gap']
                row['gap_to_real_mean'] = distances[g]['to_real_green_CCPD2020'][k]['mean_gap']
                row['z_to_hard'] = distances[g]['to_green_edgefit_hard'][k]['z_gap_vs_ref']
                row['z_to_real'] = distances[g]['to_real_green_CCPD2020'][k]['z_gap_vs_ref']
                row['w1_to_hard'] = distances[g]['to_green_edgefit_hard'][k]['wasserstein']
                row['w1_to_real'] = distances[g]['to_real_green_CCPD2020'][k]['wasserstein']
                w.writerow(row)

    # Images.
    mixed=[]
    for i,(name,recs) in enumerate(group_records.items()):
        samples=pick_samples(recs, 24, seed=20260426+i)
        render_sheet(name, samples, OUT/f'{name}_final94_samples.jpg')
        mixed.extend(samples[:4])
    render_sheet('00_mixed_final94_4_each_group', mixed, OUT/'00_mixed_final94_4_each_group.jpg')
    render_boxplot(summary, OUT/'00_distribution_boxplot_selected_metrics.jpg')

    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    write_report(summary)

    html = ['<!doctype html><html><head><meta charset="utf-8"><title>Final94 distribution audit</title><style>body{font-family:sans-serif;background:#eee} img{display:block;max-width:100%;margin:18px auto;border:1px solid #999} pre{background:white;padding:12px;white-space:pre-wrap}</style></head><body>']
    html.append('<h1>StageB1A final94 distribution audit 20260426</h1>')
    html.append('<pre>'+json.dumps({k: summary[k] for k in ['out_dir','windows_out_dir','font_path','metric_definitions']}, ensure_ascii=False, indent=2)+'</pre>')
    for img in ['00_distribution_boxplot_selected_metrics.jpg','00_mixed_final94_4_each_group.jpg'] + [f'{name}_final94_samples.jpg' for name in group_records]:
        if (OUT/img).exists():
            html.append(f'<h2>{img}</h2><img src="{img}">')
    html.append('</body></html>')
    (OUT/'index.html').write_text('\n'.join(html), encoding='utf-8')

    for p in OUT.iterdir():
        if p.is_file():
            shutil.copy2(p, WIN/p.name)
    report = ROOT/'reports/GREEN_STAGEB1A_FINAL94_DISTRIBUTION_AUDIT_REPORT.md'
    if report.exists():
        shutil.copy2(report, WIN/report.name)

    print(json.dumps({'out_dir': str(OUT), 'windows_out_dir': str(WIN), 'groups': {k: len(v) for k,v in group_records.items()}, 'report': str(report)}, ensure_ascii=False, indent=2))


def fmt_metric(summary, group, metric):
    m = summary['groups'][group]['metrics'][metric]
    if m['mean'] is None:
        return 'NA'
    return f"mean={m['mean']:.4f}, p10={m['p10']:.4f}, p50={m['p50']:.4f}, p90={m['p90']:.4f}"


def fmt_gap(summary, group, metric):
    dh = summary['distances'][group]['to_green_edgefit_hard'][metric]
    dr = summary['distances'][group]['to_real_green_CCPD2020'][metric]
    def f(x): return 'NA' if x is None else f'{x:.4f}'
    return f"gap_hard={f(dh['mean_gap'])} z_hard={f(dh['z_gap_vs_ref'])} w1_hard={f(dh['wasserstein'])}; gap_real={f(dr['mean_gap'])} z_real={f(dr['z_gap_vs_ref'])} w1_real={f(dr['wasserstein'])}"


def write_report(summary):
    p = ROOT/'reports/GREEN_STAGEB1A_FINAL94_DISTRIBUTION_AUDIT_REPORT.md'
    lines=[]
    lines.append('# GREEN StageB1A Final94 Distribution Audit Report')
    lines.append('')
    lines.append('日期：2026-04-26')
    lines.append('')
    lines.append('## 目的')
    lines.append('不开训练，只审计 final94 输入分布，比较 accepted moderate 与 hard/real 的距离，判断 accepted moderate 是否在模型实际输入口径上偏离 hard/real。')
    lines.append('')
    lines.append('## 数据组')
    for g, gs in summary['groups'].items():
        lines.append(f"- {g}: input_rows={gs['input_rows']}, usable={gs['usable']}, manifest={gs['manifest']}")
    lines.append('')
    lines.append('## 输出')
    lines.append(f"- WSL: `{summary['out_dir']}`")
    lines.append(f"- Windows QA: `{summary['windows_out_dir']}`")
    lines.append(f"- HTML: `{summary['windows_out_dir']}/index.html`")
    lines.append('- `final94_group_summary.csv`: 每组每指标 summary/gap/z/wasserstein')
    lines.append('- `final94_sample_metrics.csv`: 每样本 final94 指标')
    lines.append('- `00_distribution_boxplot_selected_metrics.jpg`: 关键指标分布图')
    lines.append('- `*_final94_samples.jpg`: 每组 final94 样本图')
    lines.append('')
    lines.append('## 关键指标摘要')
    key_metrics=['fg_mean','fg_std','fg_contrast','edge_density','occ_w','occ_h','margin_l','margin_r','black_ratio','dark_border_ratio','quad_horiz_abs','quad_vert_abs']
    for g in summary['groups']:
        lines.append(f'### {g}')
        for m in key_metrics:
            lines.append(f"- {m}: {fmt_metric(summary,g,m)} | {fmt_gap(summary,g,m)}")
        lines.append('')
    lines.append('## 初步可读结论')
    lines.append('- 本报告只基于 final94 分布统计与抽样图，不包含训练/识别效果结论。')
    lines.append('- 若 accepted low/mid/high 在亮度、边缘密度、黑边、占宽、quad 角度上相对 hard/real 的 z_gap 或 wasserstein 明显大，应优先修生成分布，而不是立即开新训练。')
    lines.append('- old proxy / accepted proxy / train accepted 需分开看：old proxy 是历史 benchmark，new accepted proxy 是目标域 probe，hard/real 是参照域。')
    p.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return p

if __name__ == '__main__':
    main()
