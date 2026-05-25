#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path

import cv2
import numpy as np

from data.load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name

ROOT = Path('/home/wzzz/LPRNet/suhu_fresh_synth_holdout_v1')
LABEL_FILE = ROOT / 'manifests' / 'holdout_labels.txt'
OUT_ROOT = Path('/home/wzzz/LPRNet/qa_board_processed_samples_suhu_fresh_holdout_v1')


def load_samples(per_province=4, seed=20260403):
    rng = random.Random(seed)
    rows = {'沪': [], '苏': []}
    for line in LABEL_FILE.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        rel_path, text = line.strip().split(maxsplit=1)
        text = text.strip().upper()
        prov = text[0]
        if prov in rows:
            rows[prov].append((rel_path, text))
    picked = []
    for prov in ['沪', '苏']:
        rng.shuffle(rows[prov])
        for rel_path, text in rows[prov][:per_province]:
            picked.append((prov, rel_path, text))
    return picked


def render(idx, prov, rel_path, text):
    img_path = ROOT / rel_path
    image = cv2.imread(str(img_path))
    if image is None:
        return f'BADIMG\t{prov}\t{text}\t{img_path}'
    quad = parse_ccpd_quad_from_name(rel_path)
    if quad is None:
        return f'BADQUAD\t{prov}\t{text}\t{img_path}'
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
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(vis, str(k), (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    left = cv2.resize(vis, (vis.shape[1] * 5, vis.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
    right = cv2.resize(prepared, (94 * 8, 24 * 8), interpolation=cv2.INTER_NEAREST)
    pad_h = max(left.shape[0], right.shape[0])
    left_pad = np.full((pad_h, left.shape[1], 3), 255, dtype=np.uint8)
    right_pad = np.full((pad_h, right.shape[1], 3), 255, dtype=np.uint8)
    left_pad[:left.shape[0], :left.shape[1]] = left
    right_pad[:right.shape[0], :right.shape[1]] = right
    canvas = np.concatenate([left_pad, right_pad], axis=1)

    safe_text = ''.join(ch if ch.isalnum() or ch >= '\u4e00' else '_' for ch in text)
    out_path = OUT_ROOT / f'{idx:02d}_{prov}_{safe_text}.jpg'
    cv2.imwrite(str(out_path), canvas)
    return '\t'.join(['OK', prov, text, rel_path, f'{occ:.4f}', str(out_path)])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = []
    picked = load_samples()
    for idx, (prov, rel_path, text) in enumerate(picked, 1):
        logs.append(render(idx, prov, rel_path, text))
    (OUT_ROOT / 'index.tsv').write_text('\n'.join(logs) + '\n', encoding='utf-8')
    (OUT_ROOT / 'README.txt').write_text(
        '左图：新生成苏/沪 synthetic 原图+quad；右图：按板端一致链路 obb_warp + letterbox + nn + none + bgr 后，真正喂给模型的 94x24 输入放大图。\n',
        encoding='utf-8'
    )
    print(str(OUT_ROOT))


if __name__ == '__main__':
    main()
