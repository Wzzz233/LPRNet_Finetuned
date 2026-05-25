#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4-min 绿牌 board-like 生成器
- 保留 exact-quad 母体
- 生成 board-like residual quad
- 用板端一致 prepare_board_ocr_input_from_quad_bgr888 做 acceptance
- 按最终 94x24 occ_ratio 分桶，而不是只按生成参数分桶
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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from lpr_pipeline_policy import apply_board_params

ALL_PROVINCES = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
]
NON_ANHUI_PROVINCES = [p for p in ALL_PROVINCES if p != '皖']
PROV_DIR = {p: f'p{i:02d}_u{ord(p):04x}' for i, p in enumerate(ALL_PROVINCES)}
LETTERS_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')
CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0, 0], [245, 0], [245, 71], [0, 71]])
IN_W, IN_H = 94, 24
SOURCE_TAG = 'synthetic_edgefit_v4min'

BUCKET_RANGES = {
    'geometry_clean': (0.84, 1.01),
    'board_mid_occ': (0.74, 0.84),
    'board_low_occ': (0.62, 0.74),
    'board_extreme_tail': (0.55, 0.68),
}

DEFAULT_QUOTAS = {
    'train': {'geometry_clean': 24, 'board_mid_occ': 36, 'board_low_occ': 28, 'board_extreme_tail': 12},
    'val': {'geometry_clean': 4, 'board_mid_occ': 6, 'board_low_occ': 5, 'board_extreme_tail': 3},
    'test': {'geometry_clean': 4, 'board_mid_occ': 6, 'board_low_occ': 5, 'board_extreme_tail': 3},
}

MANIFEST_FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type',
    'source', 'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad',
    'can_parse_ccpd_geom', 'can_perspective', 'bbox_source', 'quad_source',
    'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
]

ASYM_MODES = {
    'board_mid_occ': [('left_compressed', 0.45), ('right_compressed', 0.35), ('symmetric', 0.20)],
    'board_low_occ': [('left_compressed', 0.55), ('right_compressed', 0.30), ('symmetric', 0.15)],
    'board_extreme_tail': [('left_compressed', 0.58), ('right_compressed', 0.30), ('corner_skew', 0.12)],
}

VERTICAL_MARGIN_RANGES = {
    'geometry_clean': ((1.0, 3.0), (1.0, 3.0)),
    'board_mid_occ': ((0.6, 2.0), (0.6, 2.0)),
    'board_low_occ': ((0.4, 1.8), (0.4, 1.8)),
    'board_extreme_tail': ((0.2, 1.2), (0.2, 1.2)),
}


def ensure_repo_imports(repo_root: str):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        sys.path.insert(0, os.path.join(repo_root, 'src'))
    old = os.getcwd()
    os.chdir(repo_root)
    try:
        from generate_chars_image import CharsImageGenerator
        from generate_plate_template import LicensePlateImageGenerator
        from augment_image import ImageAugmentation
        from load_data import prepare_board_ocr_input_from_quad_bgr888
        chars_gen = CharsImageGenerator('small_new_energy')
        template_gen = LicensePlateImageGenerator('small_new_energy')
        template = template_gen.generate_template_image(chars_gen.plate_width, chars_gen.plate_height)
        augmenter = ImageAugmentation('small_new_energy', template)
        augmenter.env_data_paths = [os.path.abspath(os.path.join(repo_root, p)) for p in augmenter.env_data_paths]
        augmenter.smu = cv2.imread(os.path.abspath(os.path.join(repo_root, 'images', 'smu.jpg')))
    finally:
        os.chdir(old)
    return chars_gen, augmenter, prepare_board_ocr_input_from_quad_bgr888


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


def weighted_choice(weighted_items, rng):
    xs, ws = zip(*weighted_items)
    total = float(sum(ws))
    r = rng.random() * total
    acc = 0.0
    for x, w in zip(xs, ws):
        acc += float(w)
        if r <= acc:
            return x
    return xs[-1]


def clip_quad(quad):
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    q[:, 0] = np.clip(q[:, 0], 0, CANVAS_W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, CANVAS_H - 1)
    return q


def enforce_vertical_margins(quad, bucket, rng):
    q = clip_quad(quad)
    (top_lo, top_hi), (bot_lo, bot_hi) = VERTICAL_MARGIN_RANGES[bucket]
    top_margin = float(min(q[0, 1], q[1, 1]))
    bottom_margin = float((CANVAS_H - 1) - max(q[2, 1], q[3, 1]))

    target_top = rng.uniform(top_lo, top_hi)
    if top_margin < target_top:
        q[[0, 1], 1] += target_top - top_margin
    elif top_margin > top_hi:
        q[[0, 1], 1] -= min(top_margin - top_hi, top_margin - top_lo)

    target_bottom = rng.uniform(bot_lo, bot_hi)
    bottom_margin = float((CANVAS_H - 1) - max(q[2, 1], q[3, 1]))
    if bottom_margin < target_bottom:
        q[[2, 3], 1] -= target_bottom - bottom_margin
    elif bottom_margin > bot_hi:
        q[[2, 3], 1] += min(bottom_margin - bot_hi, bottom_margin - bot_lo)

    return clip_quad(q)


def quad_bbox(quad):
    q = np.asarray(quad, dtype=np.float32)
    x1 = int(np.floor(q[:, 0].min()))
    y1 = int(np.floor(q[:, 1].min()))
    x2 = int(np.ceil(q[:, 0].max()))
    y2 = int(np.ceil(q[:, 1].max()))
    x1 = max(0, min(CANVAS_W - 1, x1))
    x2 = max(0, min(CANVAS_W - 1, x2))
    y1 = max(0, min(CANVAS_H - 1, y1))
    y2 = max(0, min(CANVAS_H - 1, y2))
    return x1, y1, x2, y2


def warp_from_quad(img, dst_quad):
    quad = clip_quad(dst_quad)
    M = cv2.getPerspectiveTransform(SRC_RECT, quad.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, quad


def nonblack_bbox(img, threshold=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def render_plate_with_exact_and_board(base, exact_quad, bucket, rng):
    exact_img, exact_quad = warp_from_quad(base, exact_quad)
    board_quad, asym_mode = perturb_board_like(exact_quad, bucket, rng)
    board_img, board_quad = warp_from_quad(base, board_quad)
    return exact_img, exact_quad, board_img, board_quad, asym_mode


def make_exact_quad(bucket, rng):
    if bucket == 'geometry_clean':
        q = np.float32([
            [rng.uniform(3, 12), rng.uniform(3, 8)],
            [CANVAS_W - 1 - rng.uniform(3, 10), rng.uniform(4, 9)],
            [CANVAS_W - 1 - rng.uniform(2, 8), CANVAS_H - 1 - rng.uniform(3, 8)],
            [rng.uniform(4, 12), CANVAS_H - 1 - rng.uniform(2, 7)],
        ])
    elif bucket == 'board_mid_occ':
        q = np.float32([
            [rng.uniform(8, 30), rng.uniform(4, 14)],
            [CANVAS_W - 1 - rng.uniform(18, 40), rng.uniform(4, 16)],
            [CANVAS_W - 1 - rng.uniform(20, 42), CANVAS_H - 1 - rng.uniform(8, 16)],
            [rng.uniform(8, 28), CANVAS_H - 1 - rng.uniform(5, 14)],
        ])
    elif bucket == 'board_low_occ':
        q = np.float32([
            [rng.uniform(10, 44), rng.uniform(5, 18)],
            [CANVAS_W - 1 - rng.uniform(28, 62), rng.uniform(5, 20)],
            [CANVAS_W - 1 - rng.uniform(34, 72), CANVAS_H - 1 - rng.uniform(10, 20)],
            [rng.uniform(10, 36), CANVAS_H - 1 - rng.uniform(6, 18)],
        ])
    else:
        q = np.float32([
            [rng.uniform(12, 56), rng.uniform(8, 20)],
            [CANVAS_W - 1 - rng.uniform(44, 95), rng.uniform(8, 24)],
            [CANVAS_W - 1 - rng.uniform(50, 100), CANVAS_H - 1 - rng.uniform(10, 22)],
            [rng.uniform(12, 42), CANVAS_H - 1 - rng.uniform(8, 20)],
        ])
    return clip_quad(q)


def perturb_board_like(exact_quad, bucket, rng):
    q = np.asarray(exact_quad, dtype=np.float32).copy()
    if bucket == 'geometry_clean':
        asym_mode = 'symmetric'
        noise = np.array([[rng.uniform(-2, 2), rng.uniform(-1.5, 1.5)] for _ in range(4)], dtype=np.float32)
        return clip_quad(q + noise), asym_mode

    asym_mode = weighted_choice(ASYM_MODES[bucket], rng)
    if asym_mode == 'left_compressed':
        q[[0, 3], 0] += rng.uniform(4, 12)
        q[[1, 2], 0] -= rng.uniform(8, 22) if bucket != 'board_extreme_tail' else rng.uniform(16, 34)
        if bucket == 'board_extreme_tail':
            q[0, 0] += rng.uniform(3, 7)
            q[3, 0] += rng.uniform(2, 6)
            q[1, 0] -= rng.uniform(3, 7)
            q[2, 0] -= rng.uniform(4, 10)
    elif asym_mode == 'right_compressed':
        q[[1, 2], 0] -= rng.uniform(4, 12)
        q[[0, 3], 0] += rng.uniform(8, 22) if bucket != 'board_extreme_tail' else rng.uniform(16, 34)
        if bucket == 'board_extreme_tail':
            q[1, 0] -= rng.uniform(3, 7)
            q[2, 0] -= rng.uniform(2, 6)
            q[0, 0] += rng.uniform(3, 7)
            q[3, 0] += rng.uniform(4, 10)
    elif asym_mode == 'corner_skew':
        q[0] += np.array([rng.uniform(6, 18), rng.uniform(-1.2, 2.2)], dtype=np.float32)
        q[2] += np.array([rng.uniform(-14, 0), rng.uniform(-2.2, 2.2)], dtype=np.float32)
        if bucket == 'board_extreme_tail':
            if rng.random() < 0.5:
                q[1, 0] -= rng.uniform(4, 10)
                q[3, 0] += rng.uniform(4, 10)
                q[1, 1] += rng.uniform(4.0, 9.0)
                q[2, 1] -= rng.uniform(2.5, 6.0)
                q[0, 1] += rng.uniform(0.5, 2.0)
                q[3, 1] -= rng.uniform(0.5, 2.0)
            else:
                q[0, 0] += rng.uniform(4, 10)
                q[2, 0] -= rng.uniform(4, 10)
                q[0, 1] += rng.uniform(4.0, 9.0)
                q[3, 1] -= rng.uniform(2.5, 6.0)
                q[1, 1] += rng.uniform(0.5, 2.0)
                q[2, 1] -= rng.uniform(0.5, 2.0)
    else:
        q[:, 0] += rng.uniform(-4, 4)

    if bucket == 'board_mid_occ':
        global_jitter = 3.5
    elif bucket == 'board_low_occ':
        global_jitter = 6.0
    else:
        global_jitter = 9.0

    q += np.array([[rng.uniform(-global_jitter, global_jitter), rng.uniform(-global_jitter * 0.7, global_jitter * 0.7)] for _ in range(4)], dtype=np.float32)
    q = enforce_vertical_margins(q, bucket, rng)
    return clip_quad(q), asym_mode


def apply_appearance_by_bucket(img, bucket, rng):
    out = img.copy()
    meta = {'blur_strength': 0.0, 'jpeg_quality': 95, 'appearance_mode': bucket}
    if bucket == 'geometry_clean':
        if rng.random() < 0.6:
            sigma = rng.uniform(0.4, 1.1)
            k = rng.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), sigma)
            meta['blur_strength'] = sigma
        q = rng.randint(45, 70)
    elif bucket == 'board_mid_occ':
        if rng.random() < 0.8:
            sigma = rng.uniform(0.6, 1.6)
            k = rng.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), sigma)
            meta['blur_strength'] = sigma
        q = rng.randint(32, 58)
    elif bucket == 'board_low_occ':
        if rng.random() < 0.9:
            sigma = rng.uniform(0.8, 2.0)
            k = rng.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), sigma)
            meta['blur_strength'] = sigma
        q = rng.randint(24, 48)
    else:
        if rng.random() < 0.95:
            sigma = rng.uniform(1.0, 2.4)
            k = rng.choice([5, 7])
            out = cv2.GaussianBlur(out, (k, k), sigma)
            meta['blur_strength'] = sigma
        q = rng.randint(20, 42)
    meta['jpeg_quality'] = q
    ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None:
            out = dec
    alpha = rng.uniform(0.82, 0.99) if bucket != 'board_extreme_tail' else rng.uniform(0.76, 0.94)
    beta = rng.uniform(-18, -2) if bucket != 'board_extreme_tail' else rng.uniform(-24, -4)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < (0.45 if bucket == 'geometry_clean' else 0.65):
        sigma = 2.0 if bucket == 'geometry_clean' else 3.2
        noise = np.random.normal(0.0, rng.uniform(0.8, sigma), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, meta


def compute_side_width_ratio(quad):
    q = np.asarray(quad, dtype=np.float32)
    left_h = max(1e-6, float(np.linalg.norm(q[3] - q[0])))
    right_h = max(1e-6, float(np.linalg.norm(q[2] - q[1])))
    ratio = left_h / right_h
    return float(max(ratio, 1.0 / ratio))


def accept_bucket(bucket, occ_ratio):
    lo, hi = BUCKET_RANGES[bucket]
    if bucket == 'geometry_clean':
        return occ_ratio >= lo
    return lo <= occ_ratio <= hi


def _fmt_ccpd_point(pt):
    x = int(round(float(pt[0])))
    y = int(round(float(pt[1])))
    x = max(0, min(CANVAS_W - 1, x))
    y = max(0, min(CANVAS_H - 1, y))
    return f'{x}&{y}'


def build_ccpd_like_filename(bucket, uid, bbox, quad):
    x1, y1, x2, y2 = bbox
    bbox_part = f'{x1}&{y1}_{x2}&{y2}'
    quad_part = '_'.join(_fmt_ccpd_point(p) for p in np.asarray(quad, dtype=np.float32))
    return f'edgefit4-{bucket}-{bbox_part}-{quad_part}-{uid}.jpg'


def save_sample_image(img, province, split, bucket, uid, out_root, bbox=None, quad=None):
    pdir = PROV_DIR.get(province, f'p{ord(province):02d}_u{ord(province):04x}')
    folder = out_root / 'images' / split / bucket / pdir
    folder.mkdir(parents=True, exist_ok=True)
    if bbox is not None and quad is not None:
        filename = build_ccpd_like_filename(bucket, uid, bbox, quad)
    else:
        filename = f'edgefit4-{bucket}-{uid}.jpg'
    path = folder / filename
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    rel = f'images/{split}/{bucket}/{pdir}/{filename}'
    return rel, path


def build_sample_record(img, split, bucket, province, text, exact_quad, board_quad, asym_mode, appearance_meta, out_root, uid, prepare_fn=None, dataset_name='green_edgefit_v4_boardlike_a3000', source='v4_boardlike_edgefit'):
    if prepare_fn is None:
        repo_root = '/home/wzzz/LPRNet'
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
            sys.path.insert(0, os.path.join(repo_root, 'src'))
        from load_data import prepare_board_ocr_input_from_quad_bgr888 as prepare_fn
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = quad_bbox(board_quad)
    rel_path, abs_path = save_sample_image(
        img,
        province,
        split,
        bucket,
        uid,
        out_root,
        bbox=(bbox_x1, bbox_y1, bbox_x2, bbox_y2),
        quad=board_quad,
    )
    prepared, occ, warped, ordered_quad, matrix = prepare_fn(
        img,
        board_quad,
        IN_W,
        IN_H,
        'letterbox',
        'nn',
        'none',
        'bgr',
        quad_pad_ratio=0.0,
    )
    warped_h, warped_w = warped.shape[:2]
    row = {
        'split': split,
        'bucket': bucket,
        'province': province,
        'text': text,
        'rel_path': rel_path,
        'abs_path': str(abs_path),
        'exact_quad': np.asarray(exact_quad, dtype=np.float32).tolist(),
        'board_quad': np.asarray(board_quad, dtype=np.float32).tolist(),
        'quad_mode': 'board_like' if bucket != 'geometry_clean' else 'exact',
        'occ_ratio': float(occ),
        'warped_w': int(warped_w),
        'warped_h': int(warped_h),
        'warped_aspect': float(warped_w / max(1, warped_h)),
        'left_right_width_ratio': compute_side_width_ratio(board_quad),
        'bbox_x1': bbox_x1,
        'bbox_y1': bbox_y1,
        'bbox_x2': bbox_x2,
        'bbox_y2': bbox_y2,
        'asym_mode': asym_mode,
        'blur_strength': float(appearance_meta.get('blur_strength', 0.0)),
        'jpeg_quality': int(appearance_meta.get('jpeg_quality', 95)),
        'appearance_mode': appearance_meta.get('appearance_mode', bucket),
        'source_tag': SOURCE_TAG,
    }
    manifest_row = {
        'img_path': str(abs_path),
        'img_rel_path': rel_path,
        'dataset_name': dataset_name,
        'split': split,
        'text': text,
        'plate_len': len(text),
        'family': 'green8',
        'sub_type': 'green_small',
        'source': source,
        'is_real': 0,
        'need_tilt_aug': 1,
        'preprocess_group': 'ccpd_board',
        'has_bbox': 1,
        'has_quad': 1,
        'can_parse_ccpd_geom': 1,
        'can_perspective': 1,
        'bbox_source': source,
        'quad_source': source,
    }
    row['manifest_row'] = apply_board_params(manifest_row)
    return row


def apply_quota_overrides(quotas, args):
    for split in ['train', 'val', 'test']:
        for bucket in ['geometry_clean', 'board_mid_occ', 'board_low_occ', 'board_extreme_tail']:
            arg_name = f'{split}_{bucket}'
            value = getattr(args, arg_name, None)
            if value is not None:
                quotas[split][bucket] = value
    if args.train_geometry_clean is not None:
        quotas['train']['geometry_clean'] = args.train_geometry_clean
        quotas['train']['board_mid_occ'] = args.train_board_mid_occ
        quotas['train']['board_low_occ'] = args.train_board_low_occ
        quotas['train']['board_extreme_tail'] = args.train_board_extreme_tail
    return quotas


def province_sampling_order(args, split):
    if split != 'train':
        return list(ALL_PROVINCES)
    if not args.non_anhui_priority:
        return list(ALL_PROVINCES)
    return list(NON_ANHUI_PROVINCES) + ['皖']


def train_target_by_province(total_train, anhui_ratio_max):
    non_anhui_total = len(NON_ANHUI_PROVINCES)
    anhui_limit = min(int(total_train * anhui_ratio_max), total_train)
    remainder = max(0, total_train - anhui_limit)
    base = remainder // non_anhui_total
    extra = remainder % non_anhui_total
    targets = {p: base for p in NON_ANHUI_PROVINCES}
    for p in NON_ANHUI_PROVINCES[:extra]:
        targets[p] += 1
    targets['皖'] = anhui_limit
    return targets


def allocate_bucket_anhui_counts(train_quotas, anhui_total):
    buckets = list(train_quotas.keys())
    raw = {bucket: train_quotas[bucket] * anhui_total / max(1, sum(train_quotas.values())) for bucket in buckets}
    counts = {bucket: int(raw[bucket]) for bucket in buckets}
    remainder = anhui_total - sum(counts.values())
    if remainder > 0:
        order = sorted(buckets, key=lambda b: (raw[b] - counts[b], train_quotas[b]), reverse=True)
        for bucket in order[:remainder]:
            counts[bucket] += 1
    return counts


def attempt_budget_for_bucket(bucket, target):
    scale = {
        'geometry_clean': 40,
        'board_mid_occ': 60,
        'board_low_occ': 100,
        'board_extreme_tail': 320,
    }.get(bucket, 40)
    floor = {
        'geometry_clean': 50,
        'board_mid_occ': 80,
        'board_low_occ': 120,
        'board_extreme_tail': 800,
    }.get(bucket, 50)
    return max(floor, target * scale)


def generate_dataset(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'images').mkdir(exist_ok=True)
    (out_root / 'manifests').mkdir(exist_ok=True)
    (out_root / 'details').mkdir(exist_ok=True)

    used_texts = load_used_texts(args.avoid_text_files)
    chars_gen, augmenter, prepare_fn = ensure_repo_imports(args.repo_root)
    rows = []
    split_texts = defaultdict(set)
    rejects = Counter()

    quotas = apply_quota_overrides({k: dict(v) for k, v in DEFAULT_QUOTAS.items()}, args)
    train_total = sum(quotas['train'].values())
    train_targets = train_target_by_province(train_total, args.anhui_ratio_max)
    anhui_bucket_counts = allocate_bucket_anhui_counts(quotas['train'], train_targets['皖'])
    train_bucket_plan = defaultdict(dict)
    for bucket, total in quotas['train'].items():
        non_anhui_total = total - anhui_bucket_counts[bucket]
        base = non_anhui_total // len(NON_ANHUI_PROVINCES)
        extra = non_anhui_total % len(NON_ANHUI_PROVINCES)
        for idx, province in enumerate(NON_ANHUI_PROVINCES):
            train_bucket_plan[province][bucket] = base + (1 if idx < extra else 0)
        train_bucket_plan['皖'][bucket] = anhui_bucket_counts[bucket]

    for split in ['train', 'val', 'test']:
        province_list = province_sampling_order(args, split)
        for province in province_list:
            if split == 'train' and train_targets[province] <= 0:
                continue
            for bucket, target in quotas[split].items():
                if split == 'train':
                    target = train_bucket_plan[province].get(bucket, 0)
                if target <= 0:
                    continue
                made = 0
                attempts = 0
                max_attempts = attempt_budget_for_bucket(bucket, target)
                while made < target and attempts < max_attempts:
                    attempts += 1
                    text = make_random_green_plate(province, used_texts, rng)
                    base = build_base_plate(text, chars_gen, augmenter)
                    exact_quad_seed = make_exact_quad(bucket, rng)
                    exact_img, exact_quad, board_img_raw, board_quad, asym_mode = render_plate_with_exact_and_board(base, exact_quad_seed, bucket, rng)
                    board_img, appearance_meta = apply_appearance_by_bucket(board_img_raw, bucket, rng)
                    # board_img is now rendered directly from board_quad, so the rendered geometry and stored quad are consistent.
                    # Do not replace it with an axis-aligned coverage rectangle: that fixed truncation but made boxes visibly too large.
                    # Keep the perspective quad itself as supervision; only use nonblack bbox for future diagnostics if needed.
                    prepared, occ, warped, ordered_quad, matrix = prepare_fn(
                        board_img,
                        board_quad,
                        IN_W,
                        IN_H,
                        'letterbox',
                        'nn',
                        'none',
                        'bgr',
                        quad_pad_ratio=0.0,
                    )
                    if not accept_bucket(bucket, float(occ)):
                        rejects[(split, bucket, province)] += 1
                        used_texts.discard(text)
                        continue
                    uid = f'{split}-{bucket}-{province}-{made:04d}-{text}'
                    record = build_sample_record(
                        img=board_img,
                        split=split,
                        bucket=bucket,
                        province=province,
                        text=text,
                        exact_quad=exact_quad,
                        board_quad=board_quad,
                        asym_mode=asym_mode,
                        appearance_meta=appearance_meta,
                        out_root=out_root,
                        uid=uid,
                        prepare_fn=prepare_fn,
                        dataset_name=args.dataset_name,
                        source=args.source_name,
                    )
                    rows.append(record)
                    split_texts[split].add(text)
                    made += 1
                if split == 'train' and made < target:
                    raise RuntimeError(
                        f'Failed to reach target for train/{bucket}/{province}: made={made} target={target} attempts={attempts} max_attempts={max_attempts}'
                    )
    return rows, split_texts, rejects


def write_preview_triplet(record, out_root, prepare_fn):
    img = cv2.imread(record['abs_path'])
    if img is None:
        return None
    quad = np.asarray(record['board_quad'], dtype=np.float32)
    vis = img.copy()
    pts = np.rint(quad).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    prepared, occ, warped, ordered_quad, matrix = prepare_fn(
        img, quad, IN_W, IN_H, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0
    )
    warped_vis = cv2.resize(warped, (246, 72), interpolation=cv2.INTER_NEAREST)
    prepared_vis = cv2.resize(prepared, (246, 72), interpolation=cv2.INTER_NEAREST)
    canvas = np.concatenate([vis, warped_vis, prepared_vis], axis=1)
    preview_dir = out_root / 'qa_preview' / record['split'] / record['bucket']
    preview_dir.mkdir(parents=True, exist_ok=True)
    out_path = preview_dir / (Path(record['rel_path']).stem + '_triplet.jpg')
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return str(out_path)


def write_outputs(rows, split_texts, out_dir, prepare_fn=None, preview_per_bucket=2):
    out_root = Path(out_dir)
    (out_root / 'manifests').mkdir(parents=True, exist_ok=True)
    (out_root / 'details').mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    bucket_counts = Counter((r['split'], r['bucket']) for r in rows)
    preview_counts = Counter()
    preview_paths = []
    for r in rows:
        by_split[r['split']].append(r)

    for split, items in by_split.items():
        txt_path = out_root / 'manifests' / f'{split}_labels.txt'
        with txt_path.open('w', encoding='utf-8') as f:
            for r in items:
                f.write(f"{r['rel_path']} {r['text']}\n")

    fieldnames = [
        'split', 'bucket', 'province', 'text', 'rel_path', 'exact_quad', 'board_quad', 'quad_mode',
        'occ_ratio', 'warped_w', 'warped_h', 'warped_aspect', 'left_right_width_ratio',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'asym_mode',
        'blur_strength', 'jpeg_quality', 'appearance_mode', 'source_tag'
    ]
    tsv_path = out_root / 'details' / 'accepted.tsv'
    with tsv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr['exact_quad'] = json.dumps(rr['exact_quad'], ensure_ascii=False)
            rr['board_quad'] = json.dumps(rr['board_quad'], ensure_ascii=False)
            rr.pop('abs_path', None)
            rr.pop('manifest_row', None)
            w.writerow(rr)

    manifest_rows = [dict(r['manifest_row']) for r in rows]
    manifest_path = out_root / 'manifests' / 'train_manifest_v4.csv'
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest_rows)

    if prepare_fn is not None and preview_per_bucket > 0:
        for record in rows:
            key = (record['split'], record['bucket'])
            if preview_counts[key] >= preview_per_bucket:
                continue
            preview = write_preview_triplet(record, out_root, prepare_fn)
            if preview:
                preview_counts[key] += 1
                preview_paths.append(preview)

    report = {
        'total': len(rows),
        'split_counts': {k: len(v) for k, v in by_split.items()},
        'bucket_counts': {f'{s}/{b}': c for (s, b), c in sorted(bucket_counts.items())},
        'split_text_overlap': {
            'train_val': len(split_texts['train'] & split_texts['val']),
            'train_test': len(split_texts['train'] & split_texts['test']),
            'val_test': len(split_texts['val'] & split_texts['test']),
        },
        'accepted_tsv': str(tsv_path),
        'train_manifest': str(manifest_path),
        'preview_paths': preview_paths,
    }
    (out_root / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=20260410)
    ap.add_argument('--dataset_name', default='green_edgefit_v4_boardlike_a3000')
    ap.add_argument('--source_name', default='v4_boardlike_edgefit')
    ap.add_argument('--anhui_ratio_max', type=float, default=0.15)
    ap.add_argument('--non_anhui_priority', type=int, default=1)
    ap.add_argument('--preview_per_bucket', type=int, default=2)
    for split in ['train', 'val', 'test']:
        for bucket in ['geometry_clean', 'board_mid_occ', 'board_low_occ', 'board_extreme_tail']:
            ap.add_argument(f'--{split}_{bucket}', type=int)
    ap.add_argument('--avoid_text_files', nargs='*', default=[])
    return ap.parse_args()


def main():
    args = parse_args()
    rows, split_texts, rejects = generate_dataset(args)
    _, _, prepare_fn = ensure_repo_imports(args.repo_root)
    report = write_outputs(rows, split_texts, args.out_dir, prepare_fn=prepare_fn, preview_per_bucket=args.preview_per_bucket)
    report['rejects'] = {f'{s}/{b}/{p}': c for (s, b, p), c in sorted(rejects.items())[:80]}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
