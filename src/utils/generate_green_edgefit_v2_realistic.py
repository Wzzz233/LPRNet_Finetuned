#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的 Edgefit 生成器
核心改进：warp_edgefit_harder_realistic 使用相机投影模型，
生成透视变换后宽高比更接近真实倾斜车牌的样本（2.5-3.5）
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
    """简单变形 - 保持接近标准比例"""
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


def warp_edgefit_harder_realistic(img, rng):
    """
    真实透视变形 - 模拟从倾斜角度拍摄平面车牌的投影效果
    
    关键改进：
    1. 使用相机投影模型，考虑相机与车牌平面的相对位置和角度
    2. 生成的 quad 在透视变换后会恢复为矩形，但宽高比被压缩（2.5-3.5）
    3. 这样 letterbox 处理时会产生黑边，与 CCPD 真实样本一致
    """
    # 标准车牌矩形（俯视视角）
    plate_w, plate_h = 245.0, 71.0
    src_rect = np.float32([
        [0, 0],
        [plate_w, 0],
        [plate_w, plate_h],
        [0, plate_h]
    ])
    
    # 相机参数：从倾斜角度拍摄
    # 相机位置（在车牌平面上方和侧面）
    camera_height = rng.uniform(150, 400)  # 相机高度（像素）
    camera_distance = rng.uniform(300, 800)  # 相机水平距离
    
    # 旋转角度（欧拉角）
    pitch = rng.uniform(-30, 30)  # 俯仰角（绕X轴）
    yaw = rng.uniform(-45, 45)    # 偏航角（绕Y轴）- 这是产生透视效果的关键
    roll = rng.uniform(-10, 10)   # 滚转角（绕Z轴）
    
    # 构建旋转矩阵
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)
    roll_rad = np.deg2rad(roll)
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Rx @ Ry
    
    # 相机内参（简化模型）
    focal_length = rng.uniform(400, 800)
    cx, cy = CANVAS_W / 2, CANVAS_H / 2
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    # 将车牌点投影到图像平面
    dst_points = []
    for pt in src_rect:
        # 车牌平面上的3D点（z=0）
        X = np.array([pt[0] - plate_w/2, pt[1] - plate_h/2, 0, 1])
        
        # 应用旋转和平移（相机在特定位置）
        T = np.array([0, -camera_height, camera_distance])
        X_rot = R @ X[:3] + T
        
        # 透视投影
        if X_rot[2] > 0:  # 确保点在相机前方
            x_proj = (X_rot[0] * focal_length / X_rot[2]) + cx
            y_proj = (X_rot[1] * focal_length / X_rot[2]) + cy
        else:
            # 回退到简单变形
            x_proj = pt[0] + rng.uniform(-20, 20)
            y_proj = pt[1] + rng.uniform(-10, 10)
        
        dst_points.append([x_proj, y_proj])
    
    dst = np.float32(dst_points)
    
    # 裁剪到画布范围内
    dst[:,0] = np.clip(dst[:,0], 0, CANVAS_W-1)
    dst[:,1] = np.clip(dst[:,1], 0, CANVAS_H-1)
    
    # 确保 quad 有效（四点不共线，面积足够）
    quad_area = cv2.contourArea(dst.reshape(-1, 1, 2))
    if quad_area < 1000:  # 面积太小，回退到原方法
        return warp_edgefit_harder_legacy(img, rng)
    
    # 应用透视变换
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
    
    qstr = '_'.join(f'{int(x)}&{int(y)}' for x, y in quad)
    filename = f"edgefit-v2-{uid}-{qstr}.jpg"
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
                        aug, quad = warp_edgefit_harder_realistic(base, rng)
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
    ap.add_argument('--seed', type=int, default=20260404)
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
