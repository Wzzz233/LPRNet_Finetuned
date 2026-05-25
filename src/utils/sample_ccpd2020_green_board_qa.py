#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
from pathlib import Path

import cv2
import numpy as np

from load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name

MANIFEST = Path('/home/wzzz/LPRNet/manifests/Archive/unified_manifest_v4_board_aligned_real_only_crpd_raw.csv')
OUT_ROOT = Path('/home/wzzz/LPRNet/qa_board_processed_samples_ccpd2020_green')


def load_rows(per_split=3, seed=20260329):
    bucket = {'train': [], 'val': [], 'test': []}
    with MANIFEST.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['dataset_name'] == 'ccpd2020_green' and r['split'] in bucket:
                bucket[r['split']].append(r)
    rng = random.Random(seed)
    picked = []
    for split in ('train', 'val', 'test'):
        rows = bucket[split]
        rng.shuffle(rows)
        picked.extend(rows[:per_split])
    return picked


def render(row, idx):
    img_path = Path(row['img_path'])
    image = cv2.imread(str(img_path))
    if image is None:
        return f'BADIMG\t{row["split"]}\t{row["text"]}\t{img_path}'
    quad = parse_ccpd_quad_from_name(row['img_rel_path'])
    if quad is None:
        return f'BADQUAD\t{row["split"]}\t{row["text"]}\t{img_path}'
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

    vis = image.copy()
    pts = ordered_quad.astype(int)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    for k, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(vis, str(k), (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

    scale = min(900 / vis.shape[1], 1.0)
    left = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    right = cv2.resize(prepared, (94 * 8, 24 * 8), interpolation=cv2.INTER_NEAREST)
    pad_h = max(left.shape[0], right.shape[0])
    left_pad = np.full((pad_h, left.shape[1], 3), 255, dtype=np.uint8)
    right_pad = np.full((pad_h, right.shape[1], 3), 255, dtype=np.uint8)
    left_pad[:left.shape[0], :left.shape[1]] = left
    right_pad[:right.shape[0], :right.shape[1]] = right
    canvas = np.concatenate([left_pad, right_pad], axis=1)

    safe_text = ''.join(ch if ch.isalnum() or ch >= '\u4e00' else '_' for ch in row['text'])
    out_path = OUT_ROOT / f'{idx:02d}_{row["split"]}_{safe_text}.jpg'
    cv2.imwrite(str(out_path), canvas)
    return '\t'.join(['OK', row['split'], row['text'], row['img_path'], str(out_path)])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = []
    for idx, row in enumerate(load_rows(), 1):
        logs.append(render(row, idx))
    (OUT_ROOT / 'index.tsv').write_text('\n'.join(logs) + '\n', encoding='utf-8')
    (OUT_ROOT / 'README.txt').write_text(
        '左图：CCPD2020 green 原图+quad；右图：按板端一致链路 obb_warp + letterbox + nn + none + bgr 后，真正喂给模型的 94x24 输入放大图。\n',
        encoding='utf-8'
    )
    print(str(OUT_ROOT))


if __name__ == '__main__':
    main()
