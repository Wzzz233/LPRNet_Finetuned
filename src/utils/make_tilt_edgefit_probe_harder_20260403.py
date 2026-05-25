#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/wzzz/LPRNet/suhu_exactquad_holdout_v3')
TSV = ROOT / 'details' / 'accepted.tsv'
OUT = Path('/home/wzzz/LPRNet/suhu_tilt_edgefit_probe_harder_20260403')
CANVAS_W, CANVAS_H = 246, 72


def load_pick(seed=20260403, per_prov=6):
    rng = random.Random(seed)
    rows = []
    with TSV.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            row['quad'] = np.asarray(json.loads(row['quad']), dtype=np.float32)
            rows.append(row)
    picked = []
    for prov in ['沪', '苏']:
        sub = [r for r in rows if r['province'] == prov]
        rng.shuffle(sub)
        picked.extend(sub[:per_prov])
    return picked


def warp_plate_patch_with_true_quad(src, quad_src, rng):
    patch = cv2.warpPerspective(
        src,
        cv2.getPerspectiveTransform(np.asarray(quad_src, dtype=np.float32), np.float32([[0,0],[245,0],[245,71],[0,71]])),
        (CANVAS_W, CANVAS_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # 更高难度：加大倾斜/透视，但quad仍然定义为牌面边缘本身
    left_pull = rng.uniform(4, 28)
    right_pull = rng.uniform(0, 10)
    top_drop_l = rng.uniform(0, 14)
    top_drop_r = rng.uniform(4, 24)
    bot_raise_l = rng.uniform(0, 8)
    bot_raise_r = rng.uniform(0, 12)
    dst = np.float32([
        [left_pull, top_drop_l],
        [CANVAS_W - 1 - right_pull, top_drop_r],
        [CANVAS_W - 1 - rng.uniform(0, 4), CANVAS_H - 1 - bot_raise_r],
        [rng.uniform(0, 32), CANVAS_H - 1 - bot_raise_l],
    ])
    if rng.random() < 0.5:
        dst[[0,3],1] += rng.uniform(5, 12)
    else:
        dst[[1,2],1] += rng.uniform(5, 12)
    dst[:,0] = np.clip(dst[:,0], 0, CANVAS_W-1)
    dst[:,1] = np.clip(dst[:,1], 0, CANVAS_H-1)
    M = cv2.getPerspectiveTransform(np.float32([[0,0],[245,0],[245,71],[0,71]]), dst.astype(np.float32))
    warped = cv2.warpPerspective(patch, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped, dst.astype(np.float32)


def harder_realism(img, rng):
    out = img.copy()
    # 更强模糊：允许轻度运动模糊 + 高斯模糊
    if rng.random() < 0.95:
        if rng.random() < 0.45:
            k = rng.choice([5, 7])
            kernel = np.zeros((k, k), dtype=np.float32)
            if rng.random() < 0.5:
                kernel[k // 2, :] = 1.0 / k
            else:
                kernel[:, k // 2] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)
        else:
            k = rng.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.9, 2.2))
    # 更重压缩
    if rng.random() < 0.9:
        q = rng.randint(20, 45)
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    # 更暗一点，但不允许像FastCUT那样直接发黑
    if rng.random() < 0.95:
        alpha = rng.uniform(0.74, 0.93)
        beta = rng.uniform(-24, -6)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # 加噪声
    if rng.random() < 0.65:
        noise = np.random.normal(0.0, rng.uniform(2.0, 6.5), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # 轻度边缘脏污，但避免盖到整字
    if rng.random() < 0.5:
        h, w = out.shape[:2]
        band = rng.randint(2, 6)
        shade = rng.randint(10, 35)
        out[:band, :, :] = np.clip(out[:band, :, :].astype(np.int16) - shade, 0, 255).astype(np.uint8)
        out[h-band:, :, :] = np.clip(out[h-band:, :, :].astype(np.int16) - shade, 0, 255).astype(np.uint8)
    return out


def quad_bbox(quad):
    q = np.asarray(quad, dtype=np.float32)
    x1 = int(np.floor(q[:,0].min())); y1 = int(np.floor(q[:,1].min()))
    x2 = int(np.ceil(q[:,0].max())); y2 = int(np.ceil(q[:,1].max()))
    x1 = max(0, min(CANVAS_W-1, x1)); x2 = max(0, min(CANVAS_W-1, x2))
    y1 = max(0, min(CANVAS_H-1, y1)); y2 = max(0, min(CANVAS_H-1, y2))
    return x1,y1,x2,y2


def main():
    rng = random.Random(20260403)
    np.random.seed(20260403)
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'details').mkdir(parents=True, exist_ok=True)
    rows = load_pick()
    accepted = []
    for idx, row in enumerate(rows, 1):
        src_path = ROOT / row['rel_path']
        src = cv2.imread(str(src_path))
        if src is None:
            continue
        aug, quad = warp_plate_patch_with_true_quad(src, row['quad'], rng)
        aug = harder_realism(aug, rng)
        x1,y1,x2,y2 = quad_bbox(quad)
        q = np.rint(quad).astype(np.int32)
        quad_part = '_'.join(f'{int(x)}&{int(y)}' for x, y in q)
        name = f'edgefitH-0-{x1}&{y1}_{x2}&{y2}-{quad_part}-{idx:02d}-{row["province"]}-{row["text"]}.jpg'
        out_path = OUT / 'images' / name
        cv2.imwrite(str(out_path), aug)
        accepted.append({
            'idx': idx,
            'province': row['province'],
            'text': row['text'],
            'src_abs_path': str(src_path),
            'src_rel_path': row['rel_path'],
            'aug_rel_path': str(out_path.relative_to(OUT)).replace('\\','/'),
            'quad': quad.tolist(),
        })
    (OUT / 'details' / 'accepted.json').write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'DIFFICULTY_PROFILE_HARDER.txt').write_text(
        '版本：suhu_tilt_edgefit_probe_harder_20260403\n定位：在简单难度基础上加大真实化与姿态难度，但仍保持quad严格贴边。\n关键加强：\n- 倾斜/透视更强：右上/右下下压更大，左边留更大水平偏移\n- 模糊更强：高斯模糊sigma 0.9~2.2 或 5/7核运动模糊\n- 压缩更重：JPEG quality 20~45\n- 更暗：alpha 0.74~0.93, beta -24~-6\n- 噪声更重：sigma 2.0~6.5\n- 轻度上下边缘脏污，但避免大面积压黑字符\n',
        encoding='utf-8'
    )
    print(json.dumps({'out': str(OUT), 'count': len(accepted)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
