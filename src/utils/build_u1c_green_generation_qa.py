#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
MANIFEST = ROOT / 'manifests' / 'unified_manifest_official_gray3_bluegreen_u1c.csv'
OUT_DIR = ROOT / 'qa_u1c_green_generation_difficulty'
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/opentype/unifont/unifont_jp.otf',
]

TARGET_GROUPS = [
    ('e2_simple', {'source_root_tag': 'e2_green', 'difficulty_bucket': 'simple'}),
    ('e2_hard', {'source_root_tag': 'e2_green', 'difficulty_bucket': 'hard'}),
    ('e4_extreme_append', {'source_root_tag': 'u1_extra_extreme_manifest_1', 'difficulty_bucket': 'extreme'}),
    ('tier3_v3_simple', {'source_root_tag': 'u1_tier3_v3_pool', 'difficulty_bucket': 'simple'}),
    ('tier3_v3_hard', {'source_root_tag': 'u1_tier3_v3_pool', 'difficulty_bucket': 'hard'}),
    ('tier3_v3_extreme', {'source_root_tag': 'u1_tier3_v3_pool', 'difficulty_bucket': 'extreme'}),
    ('a3000_simple', {'source_root_tag': 'u1_a3000_boardlike_pool', 'difficulty_bucket': 'simple'}),
    ('a3000_hard', {'source_root_tag': 'u1_a3000_boardlike_pool', 'difficulty_bucket': 'hard'}),
    ('a3000_extreme', {'source_root_tag': 'u1_a3000_boardlike_pool', 'difficulty_bucket': 'extreme'}),
]
SAMPLES_PER_GROUP = 6
CARD_W = 980
CARD_H = 300
THUMB_W = 420
THUMB_H = 108
PAD = 24


def pick_font(size: int):
    for path in FONT_CANDIDATES:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    raise RuntimeError('No usable CJK font found')


def load_rows():
    with MANIFEST.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def filter_rows(rows, conds):
    out = []
    for r in rows:
        if r.get('family') != 'green8':
            continue
        if r.get('split') != 'train':
            continue
        ok = True
        for k, v in conds.items():
            if (r.get(k) or '') != v:
                ok = False
                break
        if ok and r.get('img_path') and Path(r['img_path']).exists():
            out.append(r)
    return out


def read_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f'Failed to read {path}')
    return img


def fit_thumb(img):
    h, w = img.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((THUMB_H, THUMB_W, 3), 255, dtype=np.uint8)
    x = (THUMB_W - new_w) // 2
    y = (THUMB_H - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def make_card(row, title, font_title, font_body):
    img = read_image(row['img_path'])
    thumb = fit_thumb(img)
    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    canvas = Image.new('RGB', (CARD_W, CARD_H), (255, 255, 255))
    canvas.paste(Image.fromarray(thumb_rgb), (PAD, PAD))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, CARD_W - 1, CARD_H - 1), outline=(180, 180, 180), width=2)
    text_x = PAD + THUMB_W + 24
    lines = [
        f'组别: {title}',
        f'text: {row.get("text", "")}',
        f'source_root_tag: {row.get("source_root_tag", "")}',
        f'source: {row.get("source", "")}',
        f'difficulty_bucket: {row.get("difficulty_bucket", "")}',
        f'img_rel_path: {row.get("img_rel_path", "")[:85]}',
    ]
    y = PAD
    draw.text((text_x, y), title, font=font_title, fill=(0, 0, 0))
    y += 44
    for line in lines[1:]:
        draw.text((text_x, y), line, font=font_body, fill=(20, 20, 20))
        y += 34
    return np.array(canvas)


def build_sheet(cards, cols=2):
    rows = math.ceil(len(cards) / cols)
    sheet = np.full((rows * (CARD_H + PAD) + PAD, cols * (CARD_W + PAD) + PAD, 3), 245, dtype=np.uint8)
    for idx, card in enumerate(cards):
        r = idx // cols
        c = idx % cols
        y = PAD + r * (CARD_H + PAD)
        x = PAD + c * (CARD_W + PAD)
        sheet[y:y + CARD_H, x:x + CARD_W] = card[:, :, ::-1]
    return sheet


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260422)
    rows = load_rows()
    font_title = pick_font(30)
    font_body = pick_font(22)
    picked_rows = []
    cards = []
    for title, conds in TARGET_GROUPS:
        pool = filter_rows(rows, conds)
        if not pool:
            continue
        rng.shuffle(pool)
        chosen = pool[:SAMPLES_PER_GROUP]
        for row in chosen:
            picked_rows.append({
                'group': title,
                'text': row.get('text', ''),
                'source_root_tag': row.get('source_root_tag', ''),
                'source': row.get('source', ''),
                'difficulty_bucket': row.get('difficulty_bucket', ''),
                'img_path': row.get('img_path', ''),
                'img_rel_path': row.get('img_rel_path', ''),
            })
            cards.append(make_card(row, title, font_title, font_body))
    sheet = build_sheet(cards, cols=2)
    sheet_path = OUT_DIR / 'u1c_green_generation_difficulty_contact_sheet.jpg'
    cv2.imwrite(str(sheet_path), sheet)
    csv_path = OUT_DIR / 'u1c_green_generation_difficulty_samples.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'text', 'source_root_tag', 'source', 'difficulty_bucket', 'img_path', 'img_rel_path'])
        writer.writeheader()
        writer.writerows(picked_rows)
    readme = OUT_DIR / 'README.txt'
    readme.write_text(
        '该 QA 联系表按 U1C manifest 抽样 green8 各生成集/难度桶。\n'
        '重点看 e2 simple/hard、e4 extreme append、tier3 v3 simple/hard/extreme、a3000 simple/hard/extreme。\n'
        f'字体: {next((p for p in FONT_CANDIDATES if Path(p).exists()), "N/A")}\n',
        encoding='utf-8'
    )
    print(str(sheet_path))
    print(str(csv_path))


if __name__ == '__main__':
    main()
