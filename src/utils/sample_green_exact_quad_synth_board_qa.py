#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path

import cv2
import numpy as np

from load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name

ROOT = Path('/home/wzzz/LPRNet/green_exact_quad_synthetic_v1')
LABEL_FILES = [
    ROOT / 'manifests' / 'train_synthetic_labels.txt',
    ROOT / 'manifests' / 'val_synthetic_labels.txt',
    ROOT / 'manifests' / 'test_synthetic_labels.txt',
]
OUT_ROOT = Path('/home/wzzz/LPRNet/qa_board_processed_samples_green_exact_quad_synth')


def load_samples(per_split=3, seed=20260329):
    rng = random.Random(seed)
    picked = []
    for label_file in LABEL_FILES:
        split = label_file.stem.split('_')[0]
        rows = []
        for line in label_file.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            rel_path, text = line.strip().split(maxsplit=1)
            rows.append((split, rel_path, text.strip().upper()))
        rng.shuffle(rows)
        picked.extend(rows[:per_split])
    return picked


def render(idx, split, rel_path, text):
    img_path = ROOT / rel_path
    image = cv2.imread(str(img_path))
    if image is None:
        return f'BADIMG\t{split}\t{text}\t{img_path}'
    quad = parse_ccpd_quad_from_name(rel_path)
    if quad is None:
        return f'BADQUAD\t{split}\t{text}\t{img_path}'
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

    safe_text = ''.join(ch if ch.isalnum() or ch >= '\u4e00' else '_' for ch in text)
    out_path = OUT_ROOT / f'{idx:02d}_{split}_{safe_text}.jpg'
    cv2.imwrite(str(out_path), canvas)
    return '\t'.join(['OK', split, text, rel_path, str(out_path)])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = []
    for idx, (split, rel_path, text) in enumerate(load_samples(), 1):
        logs.append(render(idx, split, rel_path, text))
    (OUT_ROOT / 'index.tsv').write_text('\n'.join(logs) + '\n', encoding='utf-8')
    (OUT_ROOT / 'README.txt').write_text(
        '左图：green_exact_quad_synthetic_v1 原图+quad；右图：按板端一致链路 obb_warp + letterbox + nn + none + bgr 后，真正喂给模型的 94x24 输入放大图。\n',
        encoding='utf-8'
    )
    print(str(OUT_ROOT))


if __name__ == '__main__':
    main()
