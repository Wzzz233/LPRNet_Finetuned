#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import json
from pathlib import Path
import cv2
import numpy as np

TARGET_W, TARGET_H = 94, 24
BASE = Path('/home/wzzz/LPRNet/green_edgefit_tier3_probe')
TSV = BASE / 'details' / 'accepted.tsv'
OUT = Path('/home/wzzz/LPRNet/qa_samples_tier3_probe')
OUT.mkdir(exist_ok=True)


def warp_quad_to_rect(image, quad):
    q = np.asarray(quad, dtype=np.float32)
    top = np.linalg.norm(q[1] - q[0])
    bottom = np.linalg.norm(q[2] - q[3])
    left = np.linalg.norm(q[3] - q[0])
    right = np.linalg.norm(q[2] - q[1])
    dst_w = max(1, int(max(top, bottom) + 0.5))
    dst_h = max(1, int(max(left, right) + 0.5))
    dst = np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128,128,128))
    return warped, dst_w, dst_h


def letterbox(src):
    src_h, src_w = src.shape[:2]
    out = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    scale = min(TARGET_W / float(src_w), TARGET_H / float(src_h))
    sw = max(1, min(TARGET_W, int(src_w * scale + 0.5)))
    sh = max(1, min(TARGET_H, int(src_h * scale + 0.5)))
    ox = (TARGET_W - sw) // 2
    oy = (TARGET_H - sh) // 2
    resized = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_NEAREST)
    out[oy:oy+sh, ox:ox+sw] = resized
    occ = sw / TARGET_W
    return out, scale, sw, sh, ox, oy, occ

# 每档抽 沪/湘/粤 各1张，便于人工看差异
wanted = [('沪','simple'),('沪','hard'),('沪','extreme'),('湘','simple'),('湘','hard'),('湘','extreme'),('粤','simple'),('粤','hard'),('粤','extreme')]

rows = []
with open(TSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    all_rows = list(reader)

selected = []
for prov, diff in wanted:
    for r in all_rows:
        if r['province'] == prov and r['difficulty'] == diff and r['split'] == 'train':
            selected.append(r)
            break

for r in selected:
    img_path = BASE / r['rel_path']
    text = r['text']
    diff = r['difficulty']
    prov = r['province']
    quad = json.loads(r['quad'])
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    src = img.copy()
    q = np.array(quad, dtype=np.int32).reshape(-1,1,2)
    cv2.polylines(src, [q], True, (0,255,0), 2)

    warped, ww, wh = warp_quad_to_rect(img, quad)
    final_img, scale, sw, sh, ox, oy, occ = letterbox(warped)
    ratio = ww / max(wh, 1)

    src_big = cv2.resize(src, (src.shape[1]*2, src.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    warped_big = cv2.resize(warped, (warped.shape[1]*4, warped.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    final_big = cv2.resize(final_img, (TARGET_W*8, TARGET_H*8), interpolation=cv2.INTER_NEAREST)

    margin = 15
    text_h = 55
    max_h = max(src_big.shape[0], warped_big.shape[0], final_big.shape[0])
    total_w = src_big.shape[1] + warped_big.shape[1] + final_big.shape[1] + margin*4
    total_h = max_h + text_h + margin*2
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    y = text_h + margin
    x1 = margin
    x2 = x1 + src_big.shape[1] + margin
    x3 = x2 + warped_big.shape[1] + margin
    canvas[y:y+src_big.shape[0], x1:x1+src_big.shape[1]] = src_big
    canvas[y:y+warped_big.shape[0], x2:x2+warped_big.shape[1]] = warped_big
    canvas[y:y+final_big.shape[0], x3:x3+final_big.shape[1]] = final_big

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f'{prov} {diff} {text}', (x1, 22), font, 0.6, (0,0,0), 2)
    cv2.putText(canvas, '1. source+quad', (x1, 46), font, 0.5, (0,0,0), 1)
    cv2.putText(canvas, f'2. warp {ww}x{wh} ratio={ratio:.2f}', (x2, 22), font, 0.55, (0,0,0), 2)
    cv2.putText(canvas, f'3. board 94x24 occ={occ:.2f}', (x3, 22), font, 0.55, (0,0,0), 2)
    cv2.putText(canvas, f'scaled={sw}x{sh} offset=({ox},{oy})', (x3, 46), font, 0.5, (0,0,0), 1)

    save = OUT / f'{prov}_{diff}_{text}.jpg'
    cv2.imwrite(str(save), canvas)
    print(f'{save.name} ratio={ratio:.2f} occ={occ:.2f}')

print(f'OUT={OUT}')
