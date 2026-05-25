#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from data.load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name

META_JSON = Path('/home/wzzz/style_transfer_green/domains/green_plate_infer_fastcut_probe_20260403/selected_meta.json')
FAKE_DIR = Path('/home/wzzz/style_transfer_green/outputs/green_fastcut_v1/probe_20260403_latest/images/fake_B')
OUT_DIR = Path('/home/wzzz/LPRNet/qa_fastcut_probe_board_processed_20260403')


def make_text_header(lines, width):
    canvas = np.full((36 * len(lines) + 24, width, 3), 255, dtype=np.uint8)
    y = 34
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        y += 36
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META_JSON.read_text(encoding='utf-8'))
    logs = []
    for row in meta:
        src_path = Path(row['src_abs_path'])
        fake_path = FAKE_DIR / (Path(row['probe_input_path']).stem + '.png')
        src = cv2.imread(str(src_path))
        fake = cv2.imread(str(fake_path))
        if src is None or fake is None:
            logs.append(f'BAD\t{src_path}\t{fake_path}')
            continue
        quad = parse_ccpd_quad_from_name(src_path.name)
        if quad is None:
            logs.append(f'BADQUAD\t{src_path}')
            continue

        # 注意：fake图是风格迁移结果，沿用源图quad做板端warp，仅用于人工审查是否更像真实域
        proc_src, occ_src, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(src, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)
        proc_fake, occ_fake, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(fake, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)

        vis_src = src.copy()
        pts = np.asarray(quad, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(vis_src, [pts], True, (0,255,0), 1)
        vis_fake = fake.copy()
        cv2.polylines(vis_fake, [pts], True, (0,255,0), 1)

        left_top = cv2.resize(vis_src, (src.shape[1]*4, src.shape[0]*4), interpolation=cv2.INTER_NEAREST)
        right_top = cv2.resize(vis_fake, (fake.shape[1]*4, fake.shape[0]*4), interpolation=cv2.INTER_NEAREST)
        top_h = max(left_top.shape[0], right_top.shape[0])
        a = np.full((top_h, left_top.shape[1], 3), 255, dtype=np.uint8)
        b = np.full((top_h, right_top.shape[1], 3), 255, dtype=np.uint8)
        a[:left_top.shape[0], :left_top.shape[1]] = left_top
        b[:right_top.shape[0], :right_top.shape[1]] = right_top
        top = np.concatenate([a, b], axis=1)

        left_bottom = cv2.resize(proc_src, (94*10, 24*10), interpolation=cv2.INTER_NEAREST)
        right_bottom = cv2.resize(proc_fake, (94*10, 24*10), interpolation=cv2.INTER_NEAREST)
        bot_h = max(left_bottom.shape[0], right_bottom.shape[0])
        c = np.full((bot_h, left_bottom.shape[1], 3), 255, dtype=np.uint8)
        d = np.full((bot_h, right_bottom.shape[1], 3), 255, dtype=np.uint8)
        c[:left_bottom.shape[0], :left_bottom.shape[1]] = left_bottom
        d[:right_bottom.shape[0], :right_bottom.shape[1]] = right_bottom
        bottom = np.concatenate([c, d], axis=1)

        width = max(top.shape[1], bottom.shape[1])
        if top.shape[1] < width:
            pad = np.full((top.shape[0], width-top.shape[1], 3), 255, dtype=np.uint8)
            top = np.concatenate([top, pad], axis=1)
        if bottom.shape[1] < width:
            pad = np.full((bottom.shape[0], width-bottom.shape[1], 3), 255, dtype=np.uint8)
            bottom = np.concatenate([bottom, pad], axis=1)

        header = make_text_header([
            f'idx={row["idx"]:02d} province={row["province"]} text={row["text"]}',
            f'top-left: original exact-quad synthetic | top-right: FastCUT translated',
            f'bottom-left: board input from original (occ={occ_src:.4f}) | bottom-right: board input from FastCUT (occ={occ_fake:.4f})',
        ], width)
        canvas = np.concatenate([header, top, bottom], axis=0)
        out_path = OUT_DIR / f'{int(row["idx"]):02d}_{row["province"]}_{row["text"]}.jpg'
        cv2.imwrite(str(out_path), canvas)
        logs.append('\t'.join(['OK', row['province'], row['text'], str(out_path)]))

    (OUT_DIR / 'index.tsv').write_text('\n'.join(logs) + '\n', encoding='utf-8')
    (OUT_DIR / 'README.txt').write_text(
        '四宫格对比：左上原始exact-quad synthetic，右上FastCUT翻译结果；左下原始图经板端一致处理的94x24输入，右下FastCUT图经同一quad和同一板端处理得到的94x24输入。用途是人工判断FastCUT是否让视觉风格更像真实，同时不明显破坏字符结构。\n',
        encoding='utf-8'
    )
    print(str(OUT_DIR))

if __name__ == '__main__':
    main()
