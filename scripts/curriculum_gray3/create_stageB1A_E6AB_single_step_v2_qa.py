#!/usr/bin/env python3
"""QA for single-step-warp E6A/E6B (v2) vs old two-step-warp."""
import csv, json, math
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/wzzz/LPRNet')
OUT=ROOT/'reports/stageB1A_E6AB_single_step_v2_QA_20260428'
OUT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT/'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

FONT_PATH=next(p for p in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc','/usr/share/fonts/opentype/unifont/unifont.otf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists())
FONT=ImageFont.truetype(FONT_PATH, 18); SMALL=ImageFont.truetype(FONT_PATH, 12); TINY=ImageFont.truetype(FONT_PATH, 10)
PROVS=['沪','苏','浙','粤','皖','京','湘','冀','陕','鄂','鲁','川','闽','赣','豫','渝']

# ── New preblur single-step data (v3) ──
NEW_E6A_DIR = ROOT / 'tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428'
NEW_E6B_DIR = ROOT / 'tmp/green_extreme_stageB1A_E6B_preblur_v3_20260428'

# ── Old two-step data (baseline) ──
OLD_E6A_DIR = ROOT / 'tmp/green_extreme_stageB1A_E6A_single_axis_visible_20260427'
OLD_E6B_DIR = ROOT / 'tmp/green_extreme_stageB1A_E6B_compound_visible_20260427'

def load_records(data_dir):
    meta = data_dir / 'generation_meta.json'
    if not meta.exists():
        print(f'  [WARN] missing {meta}')
        return []
    d = json.loads(meta.read_text(encoding='utf-8'))
    records = d.get('records', [])
    for r in records:
        r['img_path'] = str(data_dir / r['file'])
    return records

def get_quad(rec):
    pts = rec.get('exact_quad', None)
    if pts is None:
        return None
    return np.array(pts, dtype=np.float32).reshape(4, 2)

def pil_bgr(img): return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def fit(im, size, fill=(245, 245, 245)):
    if isinstance(im, np.ndarray): im = pil_bgr(im)
    w, h = im.size
    s = min(size[0] / max(1, w), size[1] / max(1, h))
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    rs = im.resize((nw, nh), Image.Resampling.BILINEAR)
    can = Image.new('RGB', size, fill)
    can.paste(rs, ((size[0] - nw) // 2, (size[1] - nh) // 2))
    return can

def draw_quad(draw, q, img_shape, origin, box_size):
    if q is None: return
    h, w = img_shape[:2]
    s = min(box_size[0] / w, box_size[1] / h)
    offx = origin[0] + (box_size[0] - int(w * s)) // 2
    offy = origin[1] + (box_size[1] - int(h * s)) // 2
    pts = [(offx + x * s, offy + y * s) for x, y in q]
    draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)

def pick_by_prov(records, n=12):
    by = defaultdict(list)
    for r in records:
        by[(r.get('province') or r.get('text', ''))[:1]].append(r)
    for k in by:
        by[k].sort(key=lambda r: (r.get('tier', ''), r.get('direction', '')))
    out = []
    keys = [k for k in PROVS if k in by] + [k for k in sorted(by) if k not in PROVS]
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop(0))
    return out


def render_rows(title, subtitle, records, out_path):
    cell_w, cell_h, title_h, cols = 430, 258, 90, 3
    can = Image.new('RGB', (cols * cell_w, title_h + math.ceil(len(records) / cols) * cell_h), (255, 255, 255))
    d = ImageDraw.Draw(can)
    d.text((10, 8), title, font=FONT, fill=(0, 0, 0))
    d.text((10, 34), subtitle, font=SMALL, fill=(50, 50, 50))
    for i, r in enumerate(records):
        x = (i % cols) * cell_w
        y = title_h + (i // cols) * cell_h
        d.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(205, 205, 205))
        p = Path(r['img_path'])
        img = cv2.imread(str(p))
        if img is None:
            d.text((x + 8, y + 8), f'cv2 failed {p}', font=SMALL, fill=(200, 0, 0))
            continue
        q = get_quad(r)
        if q is None:
            prep = cv2.resize(img, (94, 24), interpolation=cv2.INTER_NEAREST)
            warped = img
        else:
            prep, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
                img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)
        can.paste(fit(img, (190, 86)), (x + 8, y + 8))
        draw_quad(d, q, img.shape, (x + 8, y + 8), (190, 86))
        can.paste(fit(warped, (190, 86)), (x + 216, y + 8))
        prep_big = cv2.resize(prep, (388, 96), interpolation=cv2.INTER_NEAREST)
        can.paste(pil_bgr(prep_big), (x + 8, y + 106))
        txt = r.get('province', '') + (r.get('text', '') or '')[-7:]
        tier = r.get('tier', '')
        direc = r.get('direction', '')
        d.text((x + 8, y + 208), f'{i:02d} {txt} {tier}/{direc}', font=TINY, fill=(0, 0, 0))
        d.text((x + 8, y + 224), f'preblur v3 | {p.parent.name}/{p.name[:40]}', font=TINY, fill=(0, 0, 120))
    can.save(out_path, quality=92)


def main():
    print('Loading new E6A (single-step v2)...')
    new_e6a = load_records(NEW_E6A_DIR)
    print(f'  {len(new_e6a)} records')
    print('Loading new E6B (single-step v2)...')
    new_e6b = load_records(NEW_E6B_DIR)
    print(f'  {len(new_e6b)} records')

    print('Loading OLD E6A (two-step)...')
    old_e6a = load_records(OLD_E6A_DIR)
    print(f'  {len(old_e6a)} records')
    print('Loading OLD E6B (two-step)...')
    old_e6b = load_records(OLD_E6B_DIR)
    print(f'  {len(old_e6b)} records')

    for tier in ['low', 'mid', 'high']:
        # New E6A
        rows_a = pick_by_prov([r for r in new_e6a if r.get('tier') == tier], 12)
        render_rows(f'NEW E6A train {tier.upper()}: preblur single-step v3',
                    f'single-axis visible; 每格: raw+quad / warp / final94 gray3',
                    rows_a, OUT / f'NEW_E6A_train_{tier}_final94.jpg')

        # Old E6A
        rows_oa = pick_by_prov([r for r in old_e6a if r.get('tier') == tier], 12)
        render_rows(f'OLD E6A train {tier.upper()}: two-step warp (baseline)',
                    f'single-axis visible; 每格: raw+quad / warp / final94 gray3',
                    rows_oa, OUT / f'OLD_E6A_train_{tier}_final94.jpg')

        # New E6B
        rows_b = pick_by_prov([r for r in new_e6b if r.get('tier') == tier], 12)
        render_rows(f'NEW E6B train {tier.upper()}: preblur single-step v3',
                    f'compound visible; 每格: raw+quad / warp / final94 gray3',
                    rows_b, OUT / f'NEW_E6B_train_{tier}_final94.jpg')

        # Old E6B
        rows_ob = pick_by_prov([r for r in old_e6b if r.get('tier') == tier], 12)
        render_rows(f'OLD E6B train {tier.upper()}: two-step warp (baseline)',
                    f'compound visible; 每格: raw+quad / warp / final94 gray3',
                    rows_ob, OUT / f'OLD_E6B_train_{tier}_final94.jpg')

    print(f'\nQA saved to {OUT}')
    for f in sorted(OUT.glob('*.jpg')):
        print(f'  {f.name}  ({f.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
