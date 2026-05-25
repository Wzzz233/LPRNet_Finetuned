#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

SRC_ROOT = Path('/home/wzzz/LPRNet/green_edgefit_tier3_full_v2')
OUT_ROOT = Path('/home/wzzz/LPRNet/green_edgefit_tier3_full_v3_su_conservative')
REPO_ROOT = '/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/repo_license_plate_generator'
SEED = 20260406

ALL_PROVINCES = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
]
PROV_DIR = {p: f'p{i:02d}_u{ord(p):04x}' for i, p in enumerate(ALL_PROVINCES)}
LETTERS_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')
CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0,0],[245,0],[245,71],[0,71]])


def ensure_repo_imports(repo_root: str):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    old = os.getcwd()
    os.chdir(repo_root)
    try:
        from generate_chars_image import CharsImageGenerator
        from generate_plate_template import LicensePlateImageGenerator
        from augment_image import ImageAugmentation
        chars_gen = CharsImageGenerator('small_new_energy')
        template_gen = LicensePlateImageGenerator('small_new_energy')
        template = template_gen.generate_template_image(chars_gen.plate_width, chars_gen.plate_height)
        augmenter = ImageAugmentation('small_new_energy', template)
        augmenter.env_data_paths = [os.path.abspath(os.path.join(repo_root, p)) for p in augmenter.env_data_paths]
        augmenter.smu = cv2.imread(os.path.abspath(os.path.join(repo_root, 'images', 'smu.jpg')))
    finally:
        os.chdir(old)
    return chars_gen, augmenter


def load_used_texts_from_tsv(tsv_path):
    used = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            text = (r.get('text') or '').strip().upper()
            if text:
                used.add(text)
    return used


def make_random_green_plate(province, used_texts, rng):
    while True:
        text = province + rng.choice(LETTERS_NO_IO) + rng.choice(['D', 'F']) + rng.choice(ALNUM_NO_IO) + ''.join(rng.choice(DIGITS) for _ in range(4))
        if text not in used_texts:
            used_texts.add(text)
            return text


def build_base_plate(text, chars_gen, augmenter):
    char_img = chars_gen.generate_images([text])[0]
    img = augmenter.augment(char_img, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(img, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)


def warp_with_quad(img, quad):
    M = cv2.getPerspectiveTransform(SRC_RECT, quad.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped, quad.astype(np.float32)


def warp_simple(img, rng):
    quad = np.float32([
        [1.5, 1.5],
        [CANVAS_W-2.0, 1.5],
        [CANVAS_W-2.0, CANVAS_H-2.0],
        [1.5, CANVAS_H-2.0],
    ])
    return warp_with_quad(img, quad)


def realism_simple(img, rng):
    return img.copy()


def save_ccpd_style(img, quad, province, split, difficulty, uid, out_root):
    pdir = PROV_DIR[province]
    folder = out_root / 'images' / split / difficulty / pdir
    folder.mkdir(parents=True, exist_ok=True)
    q = np.asarray(quad, dtype=np.float32)
    x1 = int(np.floor(q[:, 0].min()))
    y1 = int(np.floor(q[:, 1].min()))
    x2 = int(np.ceil(q[:, 0].max()))
    y2 = int(np.ceil(q[:, 1].max()))
    bbox_str = f"{x1}&{y1}_{x2}&{y2}"
    qstr = '_'.join(f'{int(x)}&{int(y)}' for x, y in quad)
    filename = f"edgefit-tier3-{bbox_str}-{qstr}-{uid}.jpg"
    path = folder / filename
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return f"images/{split}/{difficulty}/{pdir}/{filename}"


def main():
    rng = random.Random(SEED)
    chars_gen, augmenter = ensure_repo_imports(REPO_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / 'images').mkdir(exist_ok=True)
    (OUT_ROOT / 'manifests').mkdir(exist_ok=True)
    (OUT_ROOT / 'details').mkdir(exist_ok=True)

    src_tsv = SRC_ROOT / 'details' / 'accepted.tsv'
    used_texts = load_used_texts_from_tsv(src_tsv)

    rows = []
    with open(src_tsv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)

    # 保留规则：非苏全部保留；苏 train simple 全保留；苏 train hard 取前36；苏 train extreme 取前12；苏 val/test 全保留
    selected = []
    su_hard_count = 0
    su_extreme_count = 0
    for r in rows:
        if r['province'] != '苏':
            selected.append(r)
            continue
        if r['split'] != 'train':
            selected.append(r)
            continue
        if r['difficulty'] == 'simple':
            selected.append(r)
        elif r['difficulty'] == 'hard':
            if su_hard_count < 36:
                selected.append(r)
                su_hard_count += 1
        elif r['difficulty'] == 'extreme':
            if su_extreme_count < 12:
                selected.append(r)
                su_extreme_count += 1

    # 复制选中旧图到新目录
    for r in selected:
        src = SRC_ROOT / r['rel_path']
        dst = OUT_ROOT / r['rel_path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    # 补24张苏 simple train
    extra_rows = []
    existing_su_simple = sum(1 for r in selected if r['province']=='苏' and r['split']=='train' and r['difficulty']=='simple')
    for idx in range(existing_su_simple, 192):
        text = make_random_green_plate('苏', used_texts, rng)
        base = build_base_plate(text, chars_gen, augmenter)
        aug, quad = warp_simple(base, rng)
        aug = realism_simple(aug, rng)
        uid = f'train-simple-苏-extra-{idx:04d}-{text}'
        rel_path = save_ccpd_style(aug, quad, '苏', 'train', 'simple', uid, OUT_ROOT)
        extra_rows.append({
            'split': 'train',
            'difficulty': 'simple',
            'province': '苏',
            'text': text,
            'rel_path': rel_path,
            'quad': json.dumps(quad.tolist(), ensure_ascii=False),
        })

    all_rows = selected + extra_rows

    # 重新写 labels.txt
    for split in ['train','val','test']:
        items = [r for r in all_rows if r['split']==split]
        with open(OUT_ROOT / 'manifests' / f'{split}_labels.txt', 'w', encoding='utf-8') as f:
            for r in items:
                f.write(f"{r['rel_path']} {r['text']}\n")

    # 写 accepted.tsv
    with open(OUT_ROOT / 'details' / 'accepted.tsv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split','difficulty','province','text','rel_path','quad'], delimiter='\t')
        w.writeheader()
        for r in all_rows:
            row = dict(r)
            if not isinstance(row['quad'], str):
                row['quad'] = json.dumps(row['quad'], ensure_ascii=False)
            w.writerow(row)

    # build report
    from collections import Counter
    summary = Counter((r['split'], r['province'], r['difficulty']) for r in all_rows)
    report = {
        'selected_total': len(selected),
        'extra_generated': len(extra_rows),
        'su_train_simple': summary[('train','苏','simple')],
        'su_train_hard': summary[('train','苏','hard')],
        'su_train_extreme': summary[('train','苏','extreme')],
        'out_dir': str(OUT_ROOT),
    }
    (OUT_ROOT / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
