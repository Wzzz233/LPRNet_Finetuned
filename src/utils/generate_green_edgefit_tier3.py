#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三档 edgefit 生成器（按用户最新定义）
- simple: 生成器原始输出，基本不做额外处理
- hard: 轻度倾斜 + 轻度模糊，板端后不应出现大量黑边
- extreme: 大倾斜，板端后应出现明显黑边
"""

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


def edge_lengths(quad):
    q = np.asarray(quad, dtype=np.float32)
    top = float(np.linalg.norm(q[1] - q[0]))
    bottom = float(np.linalg.norm(q[2] - q[3]))
    right = float(np.linalg.norm(q[2] - q[1]))
    left = float(np.linalg.norm(q[3] - q[0]))
    return top, bottom, right, left


def warped_ratio(quad):
    top, bottom, right, left = edge_lengths(quad)
    w = max(top, bottom)
    h = max(right, left)
    return w / max(h, 1e-6)


def make_rectish_quad(rng, margin_x=(8,18), margin_y=(4,10), y_skew=(1,6), x_skew=(0,6)):
    left = rng.uniform(*margin_x)
    right = CANVAS_W - 1 - rng.uniform(*margin_x)
    top = rng.uniform(*margin_y)
    bottom = CANVAS_H - 1 - rng.uniform(*margin_y)
    quad = np.float32([
        [left + rng.uniform(0, x_skew[1]), top + rng.uniform(0, y_skew[1])],
        [right - rng.uniform(0, x_skew[1]), top + rng.uniform(0, y_skew[1])],
        [right - rng.uniform(0, x_skew[1]), bottom - rng.uniform(0, y_skew[1])],
        [left + rng.uniform(0, x_skew[1]), bottom - rng.uniform(0, y_skew[1])],
    ])
    quad[:,0] = np.clip(quad[:,0], 0, CANVAS_W-1)
    quad[:,1] = np.clip(quad[:,1], 0, CANVAS_H-1)
    return quad


def warp_with_quad(img, quad):
    M = cv2.getPerspectiveTransform(SRC_RECT, quad.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped, quad.astype(np.float32)


def warp_simple(img, rng):
    # 用户定义：原始生成器模样，尽量不额外处理；这里仅给一个几乎贴满画布的轻微规整 quad
    quad = np.float32([
        [1.5, 1.5],
        [CANVAS_W-2.0, 1.5],
        [CANVAS_W-2.0, CANVAS_H-2.0],
        [1.5, CANVAS_H-2.0],
    ])
    return warp_with_quad(img, quad)


def warp_hard(img, rng):
    # 轻度倾斜：宽高比仍接近 94x24，对板端而言黑边应很少
    for _ in range(20):
        quad = make_rectish_quad(
            rng,
            margin_x=(6, 16),
            margin_y=(2, 8),
            y_skew=(1, 5),
            x_skew=(1, 8),
        )
        # 额外加一点左右不对称，但别太过
        if rng.random() < 0.5:
            quad[0,0] += rng.uniform(0, 6)
            quad[3,0] += rng.uniform(0, 8)
        else:
            quad[1,0] -= rng.uniform(0, 6)
            quad[2,0] -= rng.uniform(0, 8)
        quad[:,0] = np.clip(quad[:,0], 0, CANVAS_W-1)
        ratio = warped_ratio(quad)
        if 3.2 <= ratio <= 4.3:
            return warp_with_quad(img, quad)
    return warp_with_quad(img, make_rectish_quad(rng))


def warp_extreme(img, rng):
    # 大倾斜：板端后宽高比显著变小，letterbox 后要出现明显黑边
    for _ in range(40):
        target_ratio = rng.uniform(2.0, 2.9)
        target_h = rng.uniform(56, 68)
        target_w = target_h * target_ratio

        if rng.random() < 0.5:
            left_w = target_w * rng.uniform(0.95, 1.10)
            right_w = target_w * rng.uniform(0.45, 0.70)
        else:
            left_w = target_w * rng.uniform(0.45, 0.70)
            right_w = target_w * rng.uniform(0.95, 1.10)

        center_x = CANVAS_W / 2 + rng.uniform(-20, 20)
        center_y = CANVAS_H / 2 + rng.uniform(-6, 6)
        top_h = target_h * rng.uniform(0.88, 0.98)
        bot_h = target_h * rng.uniform(0.95, 1.05)

        quad = np.float32([
            [center_x - left_w/2, center_y - top_h/2],
            [center_x + right_w/2, center_y - top_h/2 + rng.uniform(-2, 2)],
            [center_x + right_w/2 + rng.uniform(-2, 2), center_y + bot_h/2],
            [center_x - left_w/2 + rng.uniform(-2, 2), center_y + bot_h/2 + rng.uniform(-1, 2)],
        ])
        quad[:,0] = np.clip(quad[:,0], 0, CANVAS_W-1)
        quad[:,1] = np.clip(quad[:,1], 0, CANVAS_H-1)
        area = cv2.contourArea(quad.reshape(-1,1,2))
        ratio = warped_ratio(quad)
        if area >= 1800 and 1.9 <= ratio <= 3.0:
            return warp_with_quad(img, quad)
    return warp_with_quad(img, make_rectish_quad(rng, margin_x=(20,35), margin_y=(4,10), y_skew=(2,8), x_skew=(6,16)))


def realism_simple(img, rng):
    # simple 基本不做额外处理
    return img.copy()


def realism_hard(img, rng):
    out = img.copy()
    if rng.random() < 0.85:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.4, 1.0))
    if rng.random() < 0.50:
        q = rng.randint(45, 70)
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    if rng.random() < 0.55:
        alpha = rng.uniform(0.92, 1.00)
        beta = rng.uniform(-8, 2)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return out


def realism_extreme(img, rng):
    out = img.copy()
    if rng.random() < 0.95:
        if rng.random() < 0.40:
            k = rng.choice([5, 7])
            kernel = np.zeros((k, k), dtype=np.float32)
            if rng.random() < 0.5:
                kernel[k // 2, :] = 1.0 / k
            else:
                kernel[:, k // 2] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)
        else:
            k = rng.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.9, 2.0))
    if rng.random() < 0.85:
        q = rng.randint(22, 45)
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    if rng.random() < 0.85:
        alpha = rng.uniform(0.78, 0.95)
        beta = rng.uniform(-18, -4)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.55:
        noise = np.random.normal(0.0, rng.uniform(1.5, 4.5), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def save_ccpd_style(img, quad, province, split, difficulty, uid, out_root):
    pdir = PROV_DIR.get(province, f'p{ord(province):02d}_u{ord(province):04x}')
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


def generate_dataset(args):
    rng = random.Random(args.seed)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'images').mkdir(exist_ok=True)
    (out_root / 'manifests').mkdir(exist_ok=True)
    (out_root / 'details').mkdir(exist_ok=True)

    used_texts = load_used_texts(args.avoid_text_files)
    chars_gen, augmenter = ensure_repo_imports(args.repo_root)

    quotas = {
        'train': {'simple': args.train_simple, 'hard': args.train_hard, 'extreme': args.train_extreme},
        'val': {'simple': args.val_simple, 'hard': args.val_hard, 'extreme': args.val_extreme},
        'test': {'simple': args.test_simple, 'hard': args.test_hard, 'extreme': args.test_extreme},
    }
    train_overrides = {
        '皖': {'simple': args.anhui_train_simple, 'hard': args.anhui_train_hard, 'extreme': args.anhui_train_extreme},
        '沪': {'simple': args.hu_train_simple, 'hard': args.hu_train_hard, 'extreme': args.hu_train_extreme},
        '粤': {'simple': args.yue_train_simple, 'hard': args.yue_train_hard, 'extreme': args.yue_train_extreme},
        '浙': {'simple': args.zhe_train_simple, 'hard': args.zhe_train_hard, 'extreme': args.zhe_train_extreme},
    }

    rows = []
    split_texts = defaultdict(set)

    for split in ['train', 'val', 'test']:
        for province in ALL_PROVINCES:
            for difficulty in ['simple', 'hard', 'extreme']:
                if split == 'train' and province in train_overrides:
                    target = train_overrides[province][difficulty]
                else:
                    target = quotas[split][difficulty]
                for idx in range(target):
                    text = make_random_green_plate(province, used_texts, rng)
                    split_texts[split].add(text)
                    base = build_base_plate(text, chars_gen, augmenter)

                    if difficulty == 'simple':
                        aug, quad = warp_simple(base, rng)
                        aug = realism_simple(aug, rng)
                    elif difficulty == 'hard':
                        aug, quad = warp_hard(base, rng)
                        aug = realism_hard(aug, rng)
                    else:
                        aug, quad = warp_extreme(base, rng)
                        aug = realism_extreme(aug, rng)

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
        'pair_counts_example': {f'{s}/{d}/{p}': c for (s,d,p), c in list(sorted(by_pair.items()))[:20]},
        'split_text_overlap': {
            'train_val': len(split_texts['train'] & split_texts['val']),
            'train_test': len(split_texts['train'] & split_texts['test']),
            'val_test': len(split_texts['val'] & split_texts['test']),
        },
        'accepted_tsv': str(tsv_path),
    }
    (out_root / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=20260405)
    ap.add_argument('--train_simple', type=int, default=168)
    ap.add_argument('--train_hard', type=int, default=48)
    ap.add_argument('--train_extreme', type=int, default=24)
    ap.add_argument('--anhui_train_simple', type=int, default=56)
    ap.add_argument('--anhui_train_hard', type=int, default=16)
    ap.add_argument('--anhui_train_extreme', type=int, default=8)
    ap.add_argument('--hu_train_simple', type=int, default=210)
    ap.add_argument('--hu_train_hard', type=int, default=60)
    ap.add_argument('--hu_train_extreme', type=int, default=30)
    ap.add_argument('--yue_train_simple', type=int, default=210)
    ap.add_argument('--yue_train_hard', type=int, default=60)
    ap.add_argument('--yue_train_extreme', type=int, default=30)
    ap.add_argument('--zhe_train_simple', type=int, default=180)
    ap.add_argument('--zhe_train_hard', type=int, default=42)
    ap.add_argument('--zhe_train_extreme', type=int, default=18)
    ap.add_argument('--val_simple', type=int, default=34)
    ap.add_argument('--val_hard', type=int, default=10)
    ap.add_argument('--val_extreme', type=int, default=4)
    ap.add_argument('--test_simple', type=int, default=34)
    ap.add_argument('--test_hard', type=int, default=10)
    ap.add_argument('--test_extreme', type=int, default=4)
    ap.add_argument('--avoid_text_files', nargs='*', default=[])
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rows, split_texts = generate_dataset(args)
    report = write_outputs(rows, split_texts, args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
