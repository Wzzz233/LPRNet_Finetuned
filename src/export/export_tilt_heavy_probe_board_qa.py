#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np

from data.load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name

ROOT_ORIG = Path('/home/wzzz/LPRNet/suhu_exactquad_holdout_v3')
PROBE = Path('/home/wzzz/LPRNet/suhu_tilt_heavy_probe_20260403')
META = PROBE / 'details' / 'accepted.json'
OUT = Path('/home/wzzz/LPRNet/qa_tilt_heavy_probe_board_processed_20260403')


def make_text(lines, width):
    canvas = np.full((36 * len(lines) + 20, width, 3), 255, dtype=np.uint8)
    y = 34
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (15, 15, 15), 2, cv2.LINE_AA)
        y += 36
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(META.read_text(encoding='utf-8'))
    logs = []
    for row in rows:
        src_path = Path(row['src_abs_path'])
        aug_path = PROBE / row['aug_rel_path']
        src = cv2.imread(str(src_path))
        aug = cv2.imread(str(aug_path))
        if src is None or aug is None:
            logs.append(f'BAD\t{src_path}\t{aug_path}')
            continue
        quad_src = parse_ccpd_quad_from_name(src_path.name)
        quad_aug = np.asarray(row['quad'], dtype=np.float32)
        proc_src, occ_src, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(src, quad_src, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)
        proc_aug, occ_aug, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(aug, quad_aug, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)

        vis_src = src.copy()
        pts1 = np.asarray(quad_src, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(vis_src, [pts1], True, (0,255,0), 1)
        vis_aug = aug.copy()
        pts2 = np.asarray(quad_aug, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(vis_aug, [pts2], True, (0,255,0), 1)

        top_l = cv2.resize(vis_src, (src.shape[1]*4, src.shape[0]*4), interpolation=cv2.INTER_NEAREST)
        top_r = cv2.resize(vis_aug, (aug.shape[1]*4, aug.shape[0]*4), interpolation=cv2.INTER_NEAREST)
        h1 = max(top_l.shape[0], top_r.shape[0])
        a = np.full((h1, top_l.shape[1], 3), 255, dtype=np.uint8)
        b = np.full((h1, top_r.shape[1], 3), 255, dtype=np.uint8)
        a[:top_l.shape[0], :top_l.shape[1]] = top_l
        b[:top_r.shape[0], :top_r.shape[1]] = top_r
        top = np.concatenate([a, b], axis=1)

        bot_l = cv2.resize(proc_src, (94*10, 24*10), interpolation=cv2.INTER_NEAREST)
        bot_r = cv2.resize(proc_aug, (94*10, 24*10), interpolation=cv2.INTER_NEAREST)
        h2 = max(bot_l.shape[0], bot_r.shape[0])
        c = np.full((h2, bot_l.shape[1], 3), 255, dtype=np.uint8)
        d = np.full((h2, bot_r.shape[1], 3), 255, dtype=np.uint8)
        c[:bot_l.shape[0], :bot_l.shape[1]] = bot_l
        d[:bot_r.shape[0], :bot_r.shape[1]] = bot_r
        bot = np.concatenate([c, d], axis=1)

        width = max(top.shape[1], bot.shape[1])
        if top.shape[1] < width:
            top = np.concatenate([top, np.full((top.shape[0], width-top.shape[1], 3), 255, dtype=np.uint8)], axis=1)
        if bot.shape[1] < width:
            bot = np.concatenate([bot, np.full((bot.shape[0], width-bot.shape[1], 3), 255, dtype=np.uint8)], axis=1)

        header = make_text([
            f'idx={row["idx"]:02d} province={row["province"]} text={row["text"]}',
            'top-left: original exact-quad synthetic | top-right: stronger-tilt + mild realism',
            f'bottom-left: board input original (occ={occ_src:.4f}) | bottom-right: board input augmented (occ={occ_aug:.4f})',
        ], width)
        canvas = np.concatenate([header, top, bot], axis=0)
        out_path = OUT / f'{int(row["idx"]):02d}_{row["province"]}_{row["text"]}.jpg'
        cv2.imwrite(str(out_path), canvas)
        logs.append('\t'.join(['OK', row['province'], row['text'], str(out_path)]))
    (OUT / 'index.tsv').write_text('\n'.join(logs) + '\n', encoding='utf-8')
    (OUT / 'README.txt').write_text(
        '四宫格：左上原始exact-quad synthetic，右上更强倾斜+温和真实化增强；左下原始板端94x24输入，右下增强后板端94x24输入。重点看：倾斜是否更接近真实、字符是否仍保持可读、是否比FastCUT更克制。\n',
        encoding='utf-8'
    )
    print(str(OUT))

if __name__ == '__main__':
    main()
