#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

from data.load_data import prepare_board_ocr_input_from_quad_bgr888

ROOT = Path('/home/wzzz/LPRNet/suhu_exactquad_holdout_v2')
TSV = ROOT / 'details' / 'accepted.tsv'
OUT_ROOT = Path('/home/wzzz/LPRNet/qa_board_processed_samples_suhu_exactquad_holdout_v2')


def load_rows():
    rows = []
    with TSV.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            row['quad'] = np.asarray(json.loads(row['quad']), dtype=np.float32)
            rows.append(row)
    return rows


def pick_rows(per_province=4, seed=20260403):
    rng = random.Random(seed)
    rows = load_rows()
    grouped = {'沪': [], '苏': []}
    for row in rows:
        if row['province'] in grouped:
            grouped[row['province']].append(row)
    picked = []
    for prov in ['沪', '苏']:
        rng.shuffle(grouped[prov])
        picked.extend(grouped[prov][:per_province])
    return picked


def render(idx, row):
    img_path = ROOT / row['rel_path']
    image = cv2.imread(str(img_path))
    if image is None:
        return f'BADIMG\t{row["province"]}\t{row["text"]}\t{img_path}'
    prepared, occ, warped, ordered_quad, matrix = prepare_board_ocr_input_from_quad_bgr888(
        image,
        row['quad'],
        94,
        24,
        'letterbox',
        'nn',
        'none',
        'bgr',
        quad_pad_ratio=0.0,
    )
    vis = image.copy()
    pts = row['quad'].astype(int)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 1)
    for k, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(vis, str(k), (int(x) + 2, int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    left = cv2.resize(vis, (vis.shape[1] * 5, vis.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
    right = cv2.resize(prepared, (94 * 8, 24 * 8), interpolation=cv2.INTER_NEAREST)
    pad_h = max(left.shape[0], right.shape[0])
    left_pad = np.full((pad_h, left.shape[1], 3), 255, dtype=np.uint8)
    right_pad = np.full((pad_h, right.shape[1], 3), 255, dtype=np.uint8)
    left_pad[:left.shape[0], :left.shape[1]] = left
    right_pad[:right.shape[0], :right.shape[1]] = right
    canvas = np.concatenate([left_pad, right_pad], axis=1)

    safe_text = ''.join(ch if ch.isalnum() or ch >= '\u4e00' else '_' for ch in row['text'])
    out_path = OUT_ROOT / f'{idx:02d}_{row["province"]}_{safe_text}.jpg'
    cv2.imwrite(str(out_path), canvas)
    return '\t'.join(['OK', row['province'], row['text'], row['rel_path'], f'{occ:.4f}', str(out_path)])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = []
    for idx, row in enumerate(pick_rows(), 1):
        logs.append(render(idx, row))
    (OUT_ROOT / 'index.tsv').write_text('\n'.join(logs) + '\n', encoding='utf-8')
    (OUT_ROOT / 'README.txt').write_text(
        '左图：exact quad 新生成苏/沪样本原图+quad；右图：按板端一致链路 obb_warp + letterbox + nn + none + bgr 后的 94x24 输入放大图。\n',
        encoding='utf-8'
    )
    print(str(OUT_ROOT))

if __name__ == '__main__':
    main()
