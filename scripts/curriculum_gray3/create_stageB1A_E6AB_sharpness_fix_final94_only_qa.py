#!/usr/bin/env python3
"""Render pure final94-only QA sheets for the sharpness-fixed E6A/E6B path."""

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
OUT = ROOT / 'reports/stageB1A_E6AB_sharpness_fix_final94_only_QA_20260428'
OUT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / 'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

FONT_PATH = next(
    p
    for p in [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/unifont/unifont.otf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    if Path(p).exists()
)
FONT = ImageFont.truetype(FONT_PATH, 20)
SMALL = ImageFont.truetype(FONT_PATH, 14)
TINY = ImageFont.truetype(FONT_PATH, 11)

NEW_E6A_META = ROOT / 'tmp/green_extreme_stageB1A_E6A_preblur_v3_20260428/generation_meta.json'
NEW_E6B_META = ROOT / 'tmp/green_extreme_stageB1A_E6B_preblur_v3_20260428/generation_meta.json'
E6A_SCRIPT = ROOT / 'scripts/curriculum_gray3/generate_stageB1A_E6A_single_axis_visible_dataset.py'
E6B_SCRIPT = ROOT / 'scripts/curriculum_gray3/generate_stageB1A_E6B_compound_visible_dataset.py'
ZOOM = 5
PER_TIER = 4


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(ang)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[3, 0]:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], np.float32)
    return ordered.astype(np.float32)


def parse_quad(name: str):
    parts = Path(name).stem.split('-')
    pts = []
    for pair in parts[3].split('_'):
        x, y = pair.split('&')
        pts.append([float(x), float(y)])
    return order_quad(pts)


def pil_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def choose_rows(meta_path: Path, tiers, per_tier):
    records = json.loads(meta_path.read_text(encoding='utf-8')).get('records', [])
    chosen = []
    for tier in tiers:
        pool = [r for r in records if r.get('split') == 'train' and r.get('tier') == tier]
        by_direction = defaultdict(list)
        for r in pool:
            by_direction[r.get('direction', '')].append(r)
        for key in by_direction:
            by_direction[key].sort(key=lambda r: (r.get('province', ''), r.get('text', '')))
        keys = sorted(by_direction)
        while len([r for r in chosen if r.get('tier') == tier]) < per_tier and any(by_direction.values()):
            for key in keys:
                if by_direction[key] and len([r for r in chosen if r.get('tier') == tier]) < per_tier:
                    chosen.append(by_direction[key].pop(0))
    return chosen


def render_final94(module, rec):
    src = Path(rec['source_exact'])
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(src)
    srcq = parse_quad(src.name)
    dstq = np.array(rec['exact_quad'], dtype=np.float32)
    rendered = module.render_plate_to_canvas(img, srcq, dstq)
    ok, encoded = cv2.imencode(
        '.png',
        rendered,
        [cv2.IMWRITE_PNG_COMPRESSION, int(getattr(module, 'PNG_COMPRESSION', 3))],
    )
    if not ok:
        raise RuntimeError('png encode failed during QA simulation')
    rendered = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    prep, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        rendered, dstq, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0
    )
    return prep


def render_sheet(title, subtitle, rows, module, out_path: Path):
    cols = 3
    cell_w = 500
    cell_h = 190
    title_h = 84
    canvas = Image.new(
        'RGB',
        (cols * cell_w, title_h + math.ceil(len(rows) / cols) * cell_h),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, font=FONT, fill=(0, 0, 0))
    draw.text((12, 40), subtitle, font=SMALL, fill=(60, 60, 60))

    for i, rec in enumerate(rows):
        x = (i % cols) * cell_w
        y = title_h + (i // cols) * cell_h
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(205, 205, 205))

        prep = render_final94(module, rec)
        prep_big = cv2.resize(
            prep,
            (prep.shape[1] * ZOOM, prep.shape[0] * ZOOM),
            interpolation=cv2.INTER_NEAREST,
        )
        panel = Image.new('RGB', (470, 120), (0, 0, 0))
        panel.paste(pil_bgr(prep_big), (0, 0))
        canvas.paste(panel, (x + 14, y + 14))

        draw.text(
            (x + 14, y + 144),
            f"{i:02d} {rec['tier']} {rec['direction']} {rec['text']}",
            font=SMALL,
            fill=(0, 0, 0),
        )
        draw.text(
            (x + 14, y + 165),
            '新链路最终 94x24 gray3 放大图',
            font=TINY,
            fill=(0, 90, 0),
        )

    canvas.save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiers', default='low,mid,high')
    ap.add_argument('--per-tier', type=int, default=PER_TIER)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    tiers = [x.strip() for x in args.tiers.split(',') if x.strip()]
    if not tiers:
        raise SystemExit('empty --tiers')

    e6a = load_module(E6A_SCRIPT, 'e6a_sharp_final94_qa')
    e6b = load_module(E6B_SCRIPT, 'e6b_sharp_final94_qa')

    rows_a = choose_rows(NEW_E6A_META, tiers, args.per_tier)
    rows_b = choose_rows(NEW_E6B_META, tiers, args.per_tier)
    tag = f"_{args.tag}" if args.tag else ''
    tier_label = '/'.join(tiers)

    render_sheet(
        'E6A Final94 Only QA',
        f'只看新的最终 94x24 gray3 放大图；tier={tier_label}；几何沿用当前 E6A 采样。',
        rows_a,
        e6a,
        OUT / f'E6A_final94_only{tag}.jpg',
    )
    render_sheet(
        'E6B Final94 Only QA',
        f'只看新的最终 94x24 gray3 放大图；tier={tier_label}；几何沿用当前 E6B 采样。',
        rows_b,
        e6b,
        OUT / f'E6B_final94_only{tag}.jpg',
    )

    print(f'QA saved to {OUT}')
    for path in sorted(OUT.glob('*.jpg')):
        print(path.name, path.stat().st_size)


if __name__ == '__main__':
    main()
