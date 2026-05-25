#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

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


def load_used_texts(extra_files):
    used = set()
    for path in extra_files:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix.lower() == '.csv':
            with p.open('r', encoding='utf-8', newline='') as f:
                for row in csv.DictReader(f):
                    text = (row.get('text') or '').strip().upper()
                    if text:
                        used.add(text)
        else:
            for line in p.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                if ' ' in line:
                    _, text = line.split(maxsplit=1)
                    used.add(text.strip().upper())
                else:
                    used.add(line.upper())
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


def quad_bbox(quad):
    q = np.asarray(quad, dtype=np.float32)
    x1 = int(np.floor(q[:,0].min())); y1 = int(np.floor(q[:,1].min()))
    x2 = int(np.ceil(q[:,0].max())); y2 = int(np.ceil(q[:,1].max()))
    x1 = max(0, min(CANVAS_W-1, x1)); x2 = max(0, min(CANVAS_W-1, x2))
    y1 = max(0, min(CANVAS_H-1, y1)); y2 = max(0, min(CANVAS_H-1, y2))
    return x1, y1, x2, y2


def warp_edgefit_simple(img, rng):
    dst = np.float32([
        [rng.uniform(0, 20), rng.uniform(0, 10)],
        [CANVAS_W-1-rng.uniform(0, 8), rng.uniform(2, 18)],
        [CANVAS_W-1-rng.uniform(0, 2), CANVAS_H-1-rng.uniform(0, 8)],
        [rng.uniform(0, 25), CANVAS_H-1-rng.uniform(2, 14)],
    ])
    if rng.random() < 0.5:
        dst[[0,3],1] += rng.uniform(4, 10)
    else:
        dst[[1,2],1] += rng.uniform(4, 10)
    dst[:,0] = np.clip(dst[:,0], 0, CANVAS_W-1)
    dst[:,1] = np.clip(dst[:,1], 0, CANVAS_H-1)
    M = cv2.getPerspectiveTransform(np.float32([[0,0],[245,0],[245,71],[0,71]]), dst.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped, dst.astype(np.float32)


def realism_simple(img, rng):
    out = img.copy()
    if rng.random() < 0.8:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.5, 1.4))
    if rng.random() < 0.7:
        q = rng.randint(32, 60)
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    if rng.random() < 0.85:
        alpha = rng.uniform(0.86, 0.99)
        beta = rng.uniform(-15, -2)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        noise = np.random.normal(0.0, rng.uniform(1.2, 4.0), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def warp_edgefit_harder(img, rng):
    left_pull = rng.uniform(4, 28)
    right_pull = rng.uniform(0, 10)
    top_drop_l = rng.uniform(0, 14)
    top_drop_r = rng.uniform(4, 24)
    bot_raise_l = rng.uniform(0, 8)
    bot_raise_r = rng.uniform(0, 12)
    dst = np.float32([
        [left_pull, top_drop_l],
        [CANVAS_W - 1 - right_pull, top_drop_r],
        [CANVAS_W - 1 - rng.uniform(0, 4), CANVAS_H - 1 - bot_raise_r],
        [rng.uniform(0, 32), CANVAS_H - 1 - bot_raise_l],
    ])
    if rng.random() < 0.5:
        dst[[0,3],1] += rng.uniform(5, 12)
    else:
        dst[[1,2],1] += rng.uniform(5, 12)
    dst[:,0] = np.clip(dst[:,0], 0, CANVAS_W-1)
    dst[:,1] = np.clip(dst[:,1], 0, CANVAS_H-1)
    M = cv2.getPerspectiveTransform(np.float32([[0,0],[245,0],[245,71],[0,71]]), dst.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped, dst.astype(np.float32)


def realism_harder(img, rng):
    out = img.copy()
    if rng.random() < 0.95:
        if rng.random() < 0.45:
            k = rng.choice([5, 7])
            kernel = np.zeros((k, k), dtype=np.float32)
            if rng.random() < 0.5:
                kernel[k // 2, :] = 1.0 / k
            else:
                kernel[:, k // 2] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)
        else:
            k = rng.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.9, 2.2))
    if rng.random() < 0.9:
        q = rng.randint(20, 45)
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    if rng.random() < 0.95:
        alpha = rng.uniform(0.74, 0.93)
        beta = rng.uniform(-24, -6)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.65:
        noise = np.random.normal(0.0, rng.uniform(2.0, 6.5), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        h, w = out.shape[:2]
        band = rng.randint(2, 6)
        shade = rng.randint(10, 35)
        out[:band, :, :] = np.clip(out[:band, :, :].astype(np.int16) - shade, 0, 255).astype(np.uint8)
        out[h-band:, :, :] = np.clip(out[h-band:, :, :].astype(np.int16) - shade, 0, 255).astype(np.uint8)
    return out


def save_ccpd_style(img, quad, province, split, difficulty, uid, out_root):
    x1, y1, x2, y2 = quad_bbox(quad)
    q = np.rint(np.asarray(quad, dtype=np.float32)).astype(np.int32)
    quad_part = '_'.join(f'{int(x)}&{int(y)}' for x, y in q)
    name = f'edgefit-0-{x1}&{y1}_{x2}&{y2}-{quad_part}-{uid}.jpg'
    out_dir = Path(out_root) / 'images' / split / difficulty / PROV_DIR[province]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f'write failed: {out_path}')
    return str(out_path.relative_to(out_root)).replace('\\', '/')


def generate_dataset(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    chars_gen, augmenter = ensure_repo_imports(args.repo_root)
    used_texts = load_used_texts(args.avoid_text_files)
    out_root = Path(args.out_dir)
    (out_root / 'details').mkdir(parents=True, exist_ok=True)
    (out_root / 'manifests').mkdir(parents=True, exist_ok=True)

    quotas = {
        'train': {'simple': args.train_simple, 'harder': args.train_harder},
        'val': {'simple': args.val_simple, 'harder': args.val_harder},
        'test': {'simple': args.test_simple, 'harder': args.test_harder},
    }
    train_overrides = {
        '皖': {'simple': args.anhui_train_simple, 'harder': args.anhui_train_harder},
        '浙': {'simple': args.zhe_train_simple, 'harder': args.zhe_train_harder},
    }
    rows = []
    split_texts = defaultdict(set)
    for split in ['train', 'val', 'test']:
        for province in ALL_PROVINCES:
            for difficulty in ['simple', 'harder']:
                if split == 'train' and province in train_overrides:
                    target = train_overrides[province][difficulty]
                else:
                    target = quotas[split][difficulty]
                for idx in range(target):
                    text = make_random_green_plate(province, used_texts, rng)
                    split_texts[split].add(text)
                    base = build_base_plate(text, chars_gen, augmenter)
                    if difficulty == 'simple':
                        aug, quad = warp_edgefit_simple(base, rng)
                        aug = realism_simple(aug, rng)
                    else:
                        aug, quad = warp_edgefit_harder(base, rng)
                        aug = realism_harder(aug, rng)
                    uid = f'{split}-{difficulty}-{province}-{idx:04d}-{text}'
                    rel_path = save_ccpd_style(aug, quad, province, split, difficulty, uid, out_root)
                    rows.append({
                        'split': split,
                        'difficulty': difficulty,
                        'province': province,
                        'text': text,
                        'rel_path': rel_path,
                        'quad': quad.tolist(),
                    })
    return rows, split_texts


def write_outputs(rows, split_texts, out_dir):
    out_root = Path(out_dir)
    by_split = defaultdict(list)
    by_pair = Counter((r['split'], r['difficulty'], r['province']) for r in rows)
    for r in rows:
        by_split[r['split']].append(r)
    for split, items in by_split.items():
        txt_path = out_root / 'manifests' / f'{split}_labels.txt'
        with txt_path.open('w', encoding='utf-8') as f:
            for r in items:
                f.write(f"{r['rel_path']} {r['text']}\n")
    tsv_path = out_root / 'details' / 'accepted.tsv'
    with tsv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split','difficulty','province','text','rel_path','quad'], delimiter='\t')
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr['quad'] = json.dumps(rr['quad'], ensure_ascii=False)
            w.writerow(rr)
    report = {
        'total': len(rows),
        'split_counts': {k: len(v) for k, v in by_split.items()},
        'pair_counts_example': {f'{s}/{d}/{p}': c for (s,d,p), c in list(sorted(by_pair.items()))[:12]},
        'split_text_overlap': {
            'train_val': len(split_texts['train'] & split_texts['val']),
            'train_test': len(split_texts['train'] & split_texts['test']),
            'val_test': len(split_texts['val'] & split_texts['test']),
        },
        'manifests': {split: str((out_root / 'manifests' / f'{split}_labels.txt')) for split in ['train','val','test']},
        'accepted_tsv': str(tsv_path),
    }
    (out_root / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=20260403)
    ap.add_argument('--train_simple', type=int, default=120)
    ap.add_argument('--train_harder', type=int, default=120)
    ap.add_argument('--anhui_train_simple', type=int, default=40)
    ap.add_argument('--anhui_train_harder', type=int, default=40)
    ap.add_argument('--zhe_train_simple', type=int, default=180)
    ap.add_argument('--zhe_train_harder', type=int, default=60)
    ap.add_argument('--val_simple', type=int, default=24)
    ap.add_argument('--val_harder', type=int, default=24)
    ap.add_argument('--test_simple', type=int, default=24)
    ap.add_argument('--test_harder', type=int, default=24)
    ap.add_argument('--avoid_text_files', nargs='*', default=[])
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rows, split_texts = generate_dataset(args)
    report = write_outputs(rows, split_texts, args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
