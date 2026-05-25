#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np

from data.load_data import parse_ccpd_quad_from_name, prepare_board_ocr_input_from_quad_bgr888

REPORTS = [
    '/home/wzzz/LPRNet/reports/suhu_crossdomain_error_analysis_20260403.json',
    '/home/wzzz/LPRNet/reports/tj_xiang_yue_zhe_crossdomain_error_analysis_20260403.json',
]
OUT_DIR = Path('/home/wzzz/LPRNet/qa_real_crossdomain_review_pack_20260403')


def collect_cases():
    picked = []
    seen = set()
    for rp in REPORTS:
        obj = json.loads(Path(rp).read_text(encoding='utf-8'))
        real = obj['real_summary']
        for prov, summary in real.items():
            errs = summary.get('top_errors', [])
            # 优先挑长度坍塌，再挑纯省份错，再补 province_plus_other
            ordered = []
            for t in ['length', 'province_only', 'province_plus_other', 'tail_only', 'pos2_only']:
                ordered.extend([x for x in errs if x['error_type'] == t])
            budget = 3 if prov in {'苏', '沪'} else 2
            cnt = 0
            for row in ordered:
                key = row['image_path']
                if key in seen:
                    continue
                seen.add(key)
                picked.append(row)
                cnt += 1
                if cnt >= budget:
                    break
    return picked


def draw_text_block(lines, width=980, line_h=34, margin=18):
    canvas = np.full((margin * 2 + line_h * len(lines), width, 3), 255, dtype=np.uint8)
    y = margin + 24
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2, cv2.LINE_AA)
        y += line_h
    return canvas


def render_case(idx, row):
    img_path = Path(row['image_path'])
    img = cv2.imread(str(img_path))
    if img is None:
        return f'BADIMG\t{img_path}'
    quad = parse_ccpd_quad_from_name(img_path.name)
    if quad is None:
        return f'BADQUAD\t{img_path}'
    prepared, occ, warped, ordered_quad, matrix = prepare_board_ocr_input_from_quad_bgr888(
        img, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0
    )
    vis = img.copy()
    pts = np.asarray(quad, dtype=np.int32)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    for k, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(vis, str(k), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    left = cv2.resize(vis, (vis.shape[1] * 2, vis.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    right = cv2.resize(prepared, (94 * 10, 24 * 10), interpolation=cv2.INTER_NEAREST)
    h = max(left.shape[0], right.shape[0])
    left_pad = np.full((h, left.shape[1], 3), 255, dtype=np.uint8)
    right_pad = np.full((h, right.shape[1], 3), 255, dtype=np.uint8)
    left_pad[:left.shape[0], :left.shape[1]] = left
    right_pad[:right.shape[0], :right.shape[1]] = right
    body = np.concatenate([left_pad, right_pad], axis=1)

    lines = [
        f'Case {idx:02d} | province={row["province"]} | error_type={row["error_type"]} | occ={occ:.4f}',
        f'GT:   {row["gt"]}',
        f'PRED: {row["pred"]}',
        f'FILE: {img_path.name}',
    ]
    header = draw_text_block(lines, width=body.shape[1])
    canvas = np.concatenate([header, body], axis=0)
    safe = f'{idx:02d}_{row["province"]}_{row["error_type"]}_{row["gt"]}_to_{row["pred"]}'.replace('/', '_')
    out_path = OUT_DIR / f'{safe}.jpg'
    cv2.imwrite(str(out_path), canvas)
    return '\t'.join(['OK', row['province'], row['error_type'], row['gt'], row['pred'], str(out_path)])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = collect_cases()
    logs = []
    for idx, row in enumerate(cases, 1):
        logs.append(render_case(idx, row))
    (OUT_DIR / 'index.tsv').write_text('\n'.join(logs) + '\n', encoding='utf-8')
    (OUT_DIR / 'README.txt').write_text(
        '每张图上方写明省份/错误类型/GT/PRED；下方左侧是真实原图+CCPD文件名解析的quad，右侧是按板端一致链路处理后的94x24输入放大图。重点看 length 样本是否在板端输入里已经模糊/裁切/形变，以及 province_only 样本首字区域是否明显不稳。\n',
        encoding='utf-8'
    )
    print(str(OUT_DIR))

if __name__ == '__main__':
    main()
