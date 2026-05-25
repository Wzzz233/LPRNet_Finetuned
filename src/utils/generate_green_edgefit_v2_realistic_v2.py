#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的 Edgefit 生成器 - 实用版本
核心改进：通过直接控制宽高比，让 harder 样本透视后更接近真实倾斜车牌
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
    """简单变形 - 接近标准比例（3.5-4.0）"""
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


def warp_edgefit_harder_realistic_v2(img, rng):
    """
    真实透视变形 v2 - 直接控制宽高比
    
    核心思想：
    1. 目标宽高比 2.5-3.5（与CCPD真实样本一致）
    2. 标准车牌 245x71 的宽高比是 3.45
    3. 要得到 2.5-3.5 的宽高比，需要在透视变换中压缩宽度
    
    方法：
    - 先生成目标宽高比 target_ratio ∈ [2.5, 3.5]
    - 计算透视变换后的目标高度 target_h
    - 根据梯形几何，让上宽和下宽有明显差异，模拟倾斜拍摄
    """
    # 标准车牌尺寸
    plate_w, plate_h = 245.0, 71.0
    
    # 目标宽高比（比标准车牌更扁）
    target_ratio = rng.uniform(2.3, 3.3)
    
    # 透视变换后的目标尺寸
    # 保持高度接近原始高度（或略小），宽度压缩
    target_h = rng.uniform(55, 68)  # 略小于原始 71
    target_w = target_h * target_ratio  # 计算对应宽度
    
    # 生成梯形 quad
    # 模拟从一侧拍摄：近端宽，远端窄
    
    # 选择倾斜方向：左低右高 或 左高右低
    if rng.random() < 0.5:
        # 左低右高：左边近（宽），右边远（窄）
        left_w = target_w * rng.uniform(0.9, 1.1)
        right_w = target_w * rng.uniform(0.5, 0.75)
    else:
        # 左高右低：左边远（窄），右边近（宽）
        left_w = target_w * rng.uniform(0.5, 0.75)
        right_w = target_w * rng.uniform(0.9, 1.1)
    
    # 中心位置（在画布内）
    center_x = CANVAS_W / 2 + rng.uniform(-30, 30)
    center_y = CANVAS_H / 2 + rng.uniform(-10, 10)
    
    # 垂直倾斜（模拟上下视角）
    top_h = target_h * rng.uniform(0.9, 1.0)
    bot_h = target_h * rng.uniform(0.95, 1.05)
    
    # 构建 quad（梯形）
    # 顺序：左上、右上、右下、左下
    dst = np.float32([
        [center_x - left_w/2, center_y - top_h/2],      # 左上
        [center_x + right_w/2, center_y - top_h/2],     # 右上
        [center_x + right_w/2, center_y + bot_h/2],     # 右下
        [center_x - left_w/2, center_y + bot_h/2],      # 左下
    ])
    
    # 添加随机扰动使更自然
    dst[:,0] += rng.uniform(-3, 3)
    dst[:,1] += rng.uniform(-3, 3)
    
    # 裁剪到画布
    dst[:,0] = np.clip(dst[:,0], 0, CANVAS_W-1)
    dst[:,1] = np.clip(dst[:,1], 0, CANVAS_H-1)
    
    # 检查 quad 有效性
    quad_area = cv2.contourArea(dst.reshape(-1, 1, 2))
    if quad_area < 2000:
        # 面积太小，使用保守参数重试
        return warp_edgefit_harder_legacy(img, rng)
    
    # 验证宽高比
    top_edge = np.linalg.norm(dst[1] - dst[0])
    bottom_edge = np.linalg.norm(dst[2] - dst[3])
    left_edge = np.linalg.norm(dst[3] - dst[0])
    right_edge = np.linalg.norm(dst[2] - dst[1])
    
    warped_w = max(top_edge, bottom_edge)
    warped_h = max(left_edge, right_edge)
    actual_ratio = warped_w / warped_h if warped_h > 0 else 3.5
    
    # 如果比例不在目标范围内，调整
    if actual_ratio < 2.0 or actual_ratio > 4.0:
        # 回退到 legacy 方法
        return warp_edgefit_harder_legacy(img, rng)
    
    # 应用透视变换
    src_rect = np.float32([[0,0],[plate_w,0],[plate_w,plate_h],[0,plate_h]])
    M = cv2.getPerspectiveTransform(src_rect, dst)
    warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return warped, dst.astype(np.float32)


def warp_edgefit_harder_legacy(img, rng):
    """原有的 harder 变形（作为回退）"""
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
    if rng.random() < 0.6:
        noise = np.random.normal(0.0, rng.uniform(2.0, 5.5), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        shadow = np.zeros_like(out)
        cv2.rectangle(shadow, (0, 0), (out.shape[1], out.shape[0]), (0,0,0), -1)
        mask = np.zeros((out.shape[0], out.shape[1]), dtype=np.uint8)
        pts = np.array([
            [rng.randint(0, out.shape[1]//3), 0],
            [rng.randint(out.shape[1]*2//3, out.shape[1]), 0],
            [rng.randint(out.shape[1]*2//3, out.shape[1]), out.shape[0]],
            [rng.randint(0, out.shape[1]//3), out.shape[0]]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        alpha_mask = mask.astype(np.float32) / 255.0 * rng.uniform(0.15, 0.35)
        alpha_mask = alpha_mask[:,:,np.newaxis]
        out = np.clip(out.astype(np.float32) * (1 - alpha_mask) + shadow.astype(np.float32) * alpha_mask, 0, 255).astype(np.uint8)
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

    # 兼容 load_data.py 的 CCPD 解析：parts[2]=bbox, parts[3]=quad
    filename = f"edgefit-real-{bbox_str}-{qstr}-{uid}.jpg"
    path = folder / filename
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    rel = f"images/{split}/{difficulty}/{pdir}/{filename}"
    return rel


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
        'train': {'simple': args.train_simple, 'harder': args.train_harder},
        'val': {'simple': args.val_simple, 'harder': args.val_harder},
        'test': {'simple': args.test_simple, 'harder': args.test_harder},
    }
    train_overrides = {
        '皖': {'simple': args.anhui_train_simple, 'harder': args.anhui_train_harder},
        '浙': {'simple': args.zhe_train_simple, 'harder': args.zhe_train_harder},
        '粤': {'simple': args.yue_train_simple, 'harder': args.yue_train_harder},
        '沪': {'simple': args.hu_train_simple, 'harder': args.hu_train_harder},
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
                        # 使用新的真实透视变形
                        aug, quad = warp_edgefit_harder_realistic_v2(base, rng)
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
    ap.add_argument('--seed', type=int, default=20260405)
    ap.add_argument('--train_simple', type=int, default=120)
    ap.add_argument('--train_harder', type=int, default=120)
    ap.add_argument('--anhui_train_simple', type=int, default=40)
    ap.add_argument('--anhui_train_harder', type=int, default=40)
    ap.add_argument('--zhe_train_simple', type=int, default=180)
    ap.add_argument('--zhe_train_harder', type=int, default=60)
    ap.add_argument('--yue_train_simple', type=int, default=120)
    ap.add_argument('--yue_train_harder', type=int, default=180)
    ap.add_argument('--hu_train_simple', type=int, default=120)
    ap.add_argument('--hu_train_harder', type=int, default=180)
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
