#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
from pathlib import Path

import cv2
import numpy as np

from load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name


def sample_rows_from_csv(path, dataset_name, n, seed=20260328):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['dataset_name'] == dataset_name:
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def sample_rows_from_manifest(path, dataset_name, split, n, seed=20260328):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['dataset_name'] == dataset_name and r['split'] == split:
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def board_process(img_path: Path):
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    quad = parse_ccpd_quad_from_name(img_path.name)
    if quad is None:
        return None
    prepared, occ, warped, ordered_quad, matrix = prepare_board_ocr_input_from_quad_bgr888(
        image,
        quad,
        94,
        24,
        'letterbox',
        'nn',
        'none',
        'bgr',
        quad_pad_ratio=0.0,
    )
    return prepared


def save_pair(src_path: Path, out_root: Path, group: str, idx: int, text: str):
    processed = board_process(src_path)
    if processed is None:
        return None
    src_img = cv2.imread(str(src_path))
    src_small = cv2.resize(src_img, (src_img.shape[1]*2, src_img.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    proc_big = cv2.resize(processed, (94*4, 24*4), interpolation=cv2.INTER_NEAREST)
    pad_h = max(src_small.shape[0], proc_big.shape[0])
    left = np.full((pad_h, src_small.shape[1], 3), 255, dtype=np.uint8)
    right = np.full((pad_h, proc_big.shape[1], 3), 255, dtype=np.uint8)
    left[:src_small.shape[0], :src_small.shape[1]] = src_small
    right[:proc_big.shape[0], :proc_big.shape[1]] = proc_big
    canvas = np.concatenate([left, right], axis=1)
    out_path = out_root / f'{group}_{idx:02d}_{text}.jpg'
    cv2.imwrite(str(out_path), canvas)
    return out_path


def main():
    out_root = Path('/home/wzzz/LPRNet/qa_board_processed_samples')
    out_root.mkdir(parents=True, exist_ok=True)
    lines = []

    # git_plate pseudo geom
    git_rows = sample_rows_from_csv('/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/success_records.csv', 'git_plate', 4)
    for i, r in enumerate(git_rows, 1):
        src = Path(r['output_image'])
        out = save_pair(src, out_root, 'git_plate', i, r['text'])
        lines.append(f'git_plate\t{i}\t{r["text"]}\t{src}\t{out}')

    # CRPD
    crpd_rows = sample_rows_from_manifest('/home/wzzz/LPRNet/manifests/crpd_all_raw_board_v1_supported.csv', 'real', 'train', 4)
    for i, r in enumerate(crpd_rows, 1):
        src = Path(r['img_path'])
        out = save_pair(src, out_root, 'crpd', i, r['text'])
        lines.append(f'crpd\t{i}\t{r["text"]}\t{src}\t{out}')

    # CCPD baseline reference
    ccpd_rows = sample_rows_from_manifest('/home/wzzz/LPRNet/manifests/unified_manifest_v3.csv', 'ccpd2019', 'train', 4)
    for i, r in enumerate(ccpd_rows, 1):
        src = Path(r['img_path'])
        out = save_pair(src, out_root, 'ccpd', i, r['text'])
        lines.append(f'ccpd\t{i}\t{r["text"]}\t{src}\t{out}')

    (out_root / 'index.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out_root / 'README.txt').write_text(
        '左边是原始车牌 crop，右边是按当前板端一致链路（obb_warp + letterbox + nn + none + bgr）处理后的 94x24 放大图。\n'
        '本轮没有再画彩色 quad，也没有额外覆盖奇怪文字，只保留文件名本身。\n',
        encoding='utf-8'
    )
    print(str(out_root))


if __name__ == '__main__':
    main()
