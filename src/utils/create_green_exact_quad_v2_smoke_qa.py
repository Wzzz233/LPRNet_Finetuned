#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')

import sys
sys.path.insert(0, str(ROOT / 'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888


def parse_args():
    ap = argparse.ArgumentParser(description='QA review for rebuilt green exact quad synthetic sources.')
    ap.add_argument('--old-root', default='/home/wzzz/LPRNet/datasets/green_exact_quad_synthetic_v1')
    ap.add_argument('--new-root', default='/home/wzzz/LPRNet/tmp/green_exact_quad_synthetic_v2_smoke_20260428')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/reports/green_exact_quad_v2_smoke_QA_20260428')
    return ap.parse_args()


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


def parse_quad_from_name(path: Path):
    parts = path.stem.split('-')
    pts = []
    for pair in parts[3].split('_'):
        x, y = pair.split('&')
        pts.append([float(x), float(y)])
    return order_quad(pts)


def fit_pil(img, size, fill=(245, 245, 245)):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    w, h = img.size
    scale = min(size[0] / max(1, w), size[1] / max(1, h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    rs = img.resize((nw, nh), Image.Resampling.BILINEAR)
    can = Image.new('RGB', size, fill)
    can.paste(rs, ((size[0] - nw) // 2, (size[1] - nh) // 2))
    return can


def lap(x):
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def load_rows(root: Path):
    rows = []
    for split in ['train', 'val', 'test']:
        label_path = root / 'manifests' / f'{split}_synthetic_labels.txt'
        if not label_path.exists():
            continue
        for idx, line in enumerate(label_path.read_text(encoding='utf-8').splitlines()):
            line = line.strip()
            if not line:
                continue
            rel_path, text = line.split(maxsplit=1)
            rows.append({
                'split': split,
                'index': idx,
                'rel_path': rel_path,
                'text': text,
                'img_path': root / rel_path,
            })
    return rows


def build_keyed(rows):
    return {(r['split'], r['index'], r['text']): r for r in rows}


def render_views(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    quad = parse_quad_from_name(path)
    prep, _, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img, quad, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0
    )
    return {
        'img': img,
        'warped': warped,
        'prep': prep,
        'prep_lap': lap(prep),
        'warp_lap': lap(warped),
    }


def draw_sheet(pairs, out_path: Path):
    cols = 2
    cell_w = 720
    cell_h = 270
    title_h = 96
    rows = math.ceil(len(pairs) / cols)
    can = Image.new('RGB', (cols * cell_w, title_h + rows * cell_h), (255, 255, 255))
    d = ImageDraw.Draw(can)
    d.text((12, 10), 'Green Exact Quad V2 Smoke QA', font=FONT, fill=(0, 0, 0))
    d.text((12, 42), '同文本样本：旧 exact 源图 vs 新重建 exact 源图；每格下排是同链路 final94 gray3。', font=SMALL, fill=(60, 60, 60))
    d.text((12, 66), '看点：首字是否更完整，笔画是否更干净，是否还带旧版那种脏纹理/碎边。', font=SMALL, fill=(60, 60, 60))

    for i, item in enumerate(pairs):
        x = (i % cols) * cell_w
        y = title_h + (i // cols) * cell_h
        d.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(205, 205, 205))

        old_panel = fit_pil(item['old']['warped'], (320, 96))
        new_panel = fit_pil(item['new']['warped'], (320, 96))
        can.paste(old_panel, (x + 12, y + 12))
        can.paste(new_panel, (x + 388, y + 12))

        old_big = cv2.resize(item['old']['prep'], (94 * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
        new_big = cv2.resize(item['new']['prep'], (94 * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
        can.paste(fit_pil(old_big, (320, 96), fill=(0, 0, 0)), (x + 12, y + 120))
        can.paste(fit_pil(new_big, (320, 96), fill=(0, 0, 0)), (x + 388, y + 120))

        d.text((x + 12, y + 220), f"{item['split']} #{item['index']:05d} {item['text']}", font=SMALL, fill=(0, 0, 0))
        d.text((x + 12, y + 242), f"旧 warp/final94 lap: {item['old']['warp_lap']:.1f} / {item['old']['prep_lap']:.1f}", font=TINY, fill=(130, 0, 0))
        d.text((x + 388, y + 242), f"新 warp/final94 lap: {item['new']['warp_lap']:.1f} / {item['new']['prep_lap']:.1f}", font=TINY, fill=(0, 100, 0))

    can.save(out_path, quality=92)


def draw_new_final94_only_sheet(pairs, out_path: Path):
    cols = 2
    cell_w = 720
    cell_h = 180
    title_h = 96
    rows = math.ceil(len(pairs) / cols)
    can = Image.new('RGB', (cols * cell_w, title_h + rows * cell_h), (255, 255, 255))
    d = ImageDraw.Draw(can)
    d.text((12, 10), 'Green Exact Quad V2 Smoke QA: New Final94 Only', font=FONT, fill=(0, 0, 0))
    d.text((12, 42), '只看新方案最终 94x24 gray3 放大图，不混旧图，不混中间图。', font=SMALL, fill=(60, 60, 60))
    d.text((12, 66), '看点：首字是否还能辨认，字符是否还会断裂，是否仍有大面积脏边/锯齿。', font=SMALL, fill=(60, 60, 60))

    for i, item in enumerate(pairs):
        x = (i % cols) * cell_w
        y = title_h + (i // cols) * cell_h
        d.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(205, 205, 205))

        new_big = cv2.resize(item['new']['prep'], (94 * 6, 24 * 6), interpolation=cv2.INTER_NEAREST)
        can.paste(fit_pil(new_big, (640, 110), fill=(0, 0, 0)), (x + 40, y + 18))

        d.text((x + 40, y + 138), f"{item['split']} #{item['index']:05d} {item['text']}", font=SMALL, fill=(0, 0, 0))
        d.text((x + 40, y + 158), f"新 final94 lap: {item['new']['prep_lap']:.1f}", font=TINY, fill=(0, 100, 0))

    can.save(out_path, quality=94)


def write_report(pairs, out_path: Path):
    summary = {
        'sample_count': len(pairs),
        'samples': [
            {
                'split': x['split'],
                'index': x['index'],
                'text': x['text'],
                'old_rel_path': x['old_rel_path'],
                'new_rel_path': x['new_rel_path'],
                'old_warp_lap': round(x['old']['warp_lap'], 3),
                'new_warp_lap': round(x['new']['warp_lap'], 3),
                'old_final94_lap': round(x['old']['prep_lap'], 3),
                'new_final94_lap': round(x['new']['prep_lap'], 3),
            }
            for x in pairs
        ],
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    args = parse_args()
    old_root = Path(args.old_root)
    new_root = Path(args.new_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_rows = build_keyed(load_rows(old_root))
    new_rows = build_keyed(load_rows(new_root))
    keys = [k for k in sorted(new_rows) if k in old_rows]
    if not keys:
        raise SystemExit('no matched smoke rows between old and new roots')

    pairs = []
    for key in keys:
        old = old_rows[key]
        new = new_rows[key]
        pairs.append({
            'split': key[0],
            'index': key[1],
            'text': key[2],
            'old_rel_path': old['rel_path'],
            'new_rel_path': new['rel_path'],
            'old': render_views(old['img_path']),
            'new': render_views(new['img_path']),
        })

    draw_sheet(pairs, out_dir / 'green_exact_quad_v2_smoke_sheet.jpg')
    draw_new_final94_only_sheet(pairs, out_dir / 'green_exact_quad_v2_smoke_final94_new_only.jpg')
    write_report(pairs, out_dir / 'green_exact_quad_v2_smoke_report.json')
    print(json.dumps({
        'out_dir': str(out_dir),
        'sheet': str(out_dir / 'green_exact_quad_v2_smoke_sheet.jpg'),
        'final94_new_only_sheet': str(out_dir / 'green_exact_quad_v2_smoke_final94_new_only.jpg'),
        'report': str(out_dir / 'green_exact_quad_v2_smoke_report.json'),
        'sample_count': len(pairs),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
