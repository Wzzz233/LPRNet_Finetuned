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
OUT = Path('/home/wzzz/LPRNet/suhu_tilt_edgefit_probe_20260403')
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
    # 核心修正：先用真quad把原图牌面精确裁成rect patch，再把这个patch投影到新quad。
    patch = cv2.warpPerspective(
        src,
        cv2.getPerspectiveTransform(np.asarray(quad_src, dtype=np.float32), np.float32([[0,0],[245,0],[245,71],[0,71]])),
        (CANVAS_W, CANVAS_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 在空白画布上定义新的目标quad；quad本身就是车牌边缘，不再让框漂在外面
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    dst = np.float32([
        [rng.uniform(0, 20), rng.uniform(0, 10)],
        [CANVAS_W-1-rng.uniform(0, 8), rng.uniform(2, 18)],
        [CANVAS_W-1-rng.uniform(0, 2), CANVAS_H-1-rng.uniform(0, 8)],
        [rng.uniform(0, 25), CANVAS_H-1-rng.uniform(2, 14)],
    ])
    # 让左右高度差更明显，增强倾斜
    if rng.random() < 0.5:
        dst[[0,3],1] += rng.uniform(4, 10)
    else:
        dst[[1,2],1] += rng.uniform(4, 10)
    dst[:,0] = np.clip(dst[:,0], 0, CANVAS_W-1)
    dst[:,1] = np.clip(dst[:,1], 0, CANVAS_H-1)

    M = cv2.getPerspectiveTransform(np.float32([[0,0],[245,0],[245,71],[0,71]]), dst.astype(np.float32))
    warped = cv2.warpPerspective(patch, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    quad = dst.astype(np.float32)
    return warped, quad


def mild_realism(img, rng):
    out = img.copy()
    if rng.random() < 0.8:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.5, 1.4))
    if rng.random() < 0.7:
        q = rng.randint(32, 60)
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    if rng.random() < 0.85:
        alpha = rng.uniform(0.86, 0.99)
        beta = rng.uniform(-15, -2)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        noise = np.random.normal(0.0, rng.uniform(1.2, 4.0), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
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
        aug = mild_realism(aug, rng)
        x1,y1,x2,y2 = quad_bbox(quad)
        q = np.rint(quad).astype(np.int32)
        quad_part = '_'.join(f'{int(x)}&{int(y)}' for x, y in q)
        name = f'edgefit-0-{x1}&{y1}_{x2}&{y2}-{quad_part}-{idx:02d}-{row["province"]}-{row["text"]}.jpg'
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
    print(json.dumps({'out': str(OUT), 'count': len(accepted)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
