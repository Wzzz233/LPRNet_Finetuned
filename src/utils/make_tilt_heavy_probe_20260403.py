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
OUT = Path('/home/wzzz/LPRNet/suhu_tilt_heavy_probe_20260403')


def clip_quad(quad, w, h):
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    quad[:, 0] = np.clip(quad[:, 0], 0, max(0, w - 1))
    quad[:, 1] = np.clip(quad[:, 1], 0, max(0, h - 1))
    return quad


def quad_bbox(quad, w, h):
    quad = clip_quad(quad, w, h)
    x1 = int(np.floor(np.min(quad[:, 0]))); y1 = int(np.floor(np.min(quad[:, 1])))
    x2 = int(np.ceil(np.max(quad[:, 0]))); y2 = int(np.ceil(np.max(quad[:, 1])))
    x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2)); y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def strong_perspective(img, rng):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src.copy()
    max_dx = 22
    max_dy = 12
    # 增强左右不对称，故意拉大倾斜，但不做极端遮挡/黑化
    dst[0] += [rng.uniform(-max_dx, max_dx*0.5), rng.uniform(-max_dy, max_dy)]
    dst[1] += [rng.uniform(-max_dx*0.5, max_dx), rng.uniform(-max_dy, max_dy)]
    dst[2] += [rng.uniform(-max_dx*0.5, max_dx), rng.uniform(-max_dy, max_dy)]
    dst[3] += [rng.uniform(-max_dx, max_dx*0.5), rng.uniform(-max_dy, max_dy)]
    dst[:, 0] = np.clip(dst[:, 0], 0, w - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, h - 1)
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    quad = cv2.perspectiveTransform(src.reshape(1, 4, 2), M).reshape(4, 2)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return warped, clip_quad(quad, w, h)


def mild_realism(img, rng):
    out = img.copy()
    if rng.random() < 0.85:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.6, 1.8))
    if rng.random() < 0.7:
        q = rng.randint(28, 55)
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    if rng.random() < 0.9:
        alpha = rng.uniform(0.82, 0.98)
        beta = rng.uniform(-18, -3)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.55:
        noise = np.random.normal(0.0, rng.uniform(1.5, 5.5), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


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


def main():
    rng = random.Random(20260403)
    np.random.seed(20260403)
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'details').mkdir(parents=True, exist_ok=True)
    picked = load_pick()
    accepted = []
    for idx, row in enumerate(picked, 1):
        src = cv2.imread(str(ROOT / row['rel_path']))
        if src is None:
            continue
        aug, quad = strong_perspective(src, rng)
        aug = mild_realism(aug, rng)
        h, w = aug.shape[:2]
        x1, y1, x2, y2 = quad_bbox(quad, w, h)
        q = np.rint(quad).astype(np.int32)
        quad_part = '_'.join(f'{int(x)}&{int(y)}' for x, y in q)
        name = f'tiltx-0-{x1}&{y1}_{x2}&{y2}-{quad_part}-{idx:02d}-{row["province"]}-{row["text"]}.jpg'
        out_path = OUT / 'images' / name
        cv2.imwrite(str(out_path), aug)
        accepted.append({
            'idx': idx,
            'province': row['province'],
            'text': row['text'],
            'src_rel_path': row['rel_path'],
            'src_abs_path': str(ROOT / row['rel_path']),
            'aug_rel_path': str(out_path.relative_to(OUT)).replace('\\', '/'),
            'quad': quad.tolist(),
        })
    (OUT / 'details' / 'accepted.json').write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(OUT), 'count': len(accepted)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
