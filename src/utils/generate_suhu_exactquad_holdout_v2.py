#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROVINCES = ['沪', '苏']
LETTERS_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')
PROV_DIR = {'沪': 'p02_u6caa', '苏': 'p10_u82cf'}


def ensure_repo_imports(repo_root):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    old = os.getcwd()
    os.chdir(repo_root)
    try:
        from generate_chars_image import CharsImageGenerator
        from generate_plate_template import LicensePlateImageGenerator
        from augment_image import ImageAugmentation
        chars_gen = CharsImageGenerator('small_new_energy')
        template_gen = LicensePlateImageGenerator('small_new_energy')
        template = template_gen.generate_template_image(chars_gen.plate_width, chars_gen.plate_height)
        augmenter = ImageAugmentation('small_new_energy', template)
        augmenter.env_data_paths = [os.path.abspath(os.path.join(repo_root, path)) for path in augmenter.env_data_paths]
        augmenter.smu = cv2.imread(os.path.abspath(os.path.join(repo_root, 'images', 'smu.jpg')))
    finally:
        os.chdir(old)
    return chars_gen, augmenter


def load_used_texts(audit_dir):
    audit = json.loads((Path(audit_dir) / 'audit_report.json').read_text(encoding='utf-8'))
    used = set()
    for key in ['train', 'val', 'test']:
        txt = audit['cleaned_manifests'][key]
        for line in Path(txt).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            _rel, text = line.strip().split(maxsplit=1)
            used.add(text.strip().upper())
    return used


def make_random_green_plate(province, used_texts, rng):
    while True:
        text = province + rng.choice(LETTERS_NO_IO) + rng.choice(['D', 'F']) + rng.choice(ALNUM_NO_IO) + ''.join(rng.choice(DIGITS) for _ in range(4))
        if text not in used_texts:
            used_texts.add(text)
            return text


def build_base_plate(text, chars_gen, augmenter):
    # exact quad真值从整张图边界开始跟踪，不再用检测器猜
    char_img = chars_gen.generate_images([text])[0]
    img = augmenter.augment(char_img, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(img, (246, 72), interpolation=cv2.INTER_AREA)


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


def maybe_perspective(img, quad, rng):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src.copy()
    max_dx = 8
    max_dy = 5
    for i in range(4):
        dst[i] += [rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)]
    dst[:, 0] = np.clip(dst[:, 0], 0, w - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, h - 1)
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    quad2 = cv2.perspectiveTransform(np.asarray(quad, dtype=np.float32).reshape(1, 4, 2), M).reshape(4, 2)
    return warped, clip_quad(quad2, w, h)


def maybe_blur(img, rng):
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.4, 1.0))
    return img


def maybe_compress(img, rng):
    if rng.random() < 0.6:
        q = rng.randint(45, 85)
        ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                img = dec
    return img


def maybe_brightness(img, rng):
    if rng.random() < 0.5:
        alpha = rng.uniform(0.9, 1.1)
        beta = rng.uniform(-8, 8)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img


def maybe_noise(img, rng):
    if rng.random() < 0.35:
        noise = np.random.normal(0.0, rng.uniform(1.0, 5.0), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def augment_plate(img, rng):
    h, w = img.shape[:2]
    quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    img, quad = maybe_perspective(img, quad, rng)
    img = maybe_blur(img, rng)
    img = maybe_compress(img, rng)
    img = maybe_brightness(img, rng)
    img = maybe_noise(img, rng)
    return img, clip_quad(quad, w, h)


def save_image(img, quad, province, uid, output_root):
    h, w = img.shape[:2]
    quad = clip_quad(quad, w, h)
    x1, y1, x2, y2 = quad_bbox(quad, w, h)
    quad_int = np.rint(quad).astype(np.int32)
    quad_part = '_'.join(f'{int(x)}&{int(y)}' for x, y in quad_int)
    name = f'genx-0-{x1}&{y1}_{x2}&{y2}-{quad_part}-{uid}.jpg'
    out_dir = Path(output_root) / 'images' / PROV_DIR[province]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f'write failed: {out_path}')
    return str(out_path.relative_to(output_root)).replace('\\', '/'), quad.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--audit_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_province', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20260403)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    used_texts = load_used_texts(args.audit_dir)
    chars_gen, augmenter = ensure_repo_imports(args.repo_root)

    out_dir = Path(args.out_dir)
    (out_dir / 'details').mkdir(parents=True, exist_ok=True)
    (out_dir / 'manifests').mkdir(parents=True, exist_ok=True)
    (out_dir / 'preview').mkdir(parents=True, exist_ok=True)

    accepted = []
    for province in PROVINCES:
        for i in range(int(args.per_province)):
            text = make_random_green_plate(province, used_texts, rng)
            base = build_base_plate(text, chars_gen, augmenter)
            aug, quad = augment_plate(base, rng)
            uid = f'{province}-{i:04d}'
            rel_path, quad_list = save_image(aug, quad, province, uid, out_dir)
            accepted.append({'split': 'holdout', 'province': province, 'text': text, 'rel_path': rel_path, 'quad': quad_list})
            if i < 6:
                preview = aug.copy()
                pts = np.asarray(quad, dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(preview, [pts], True, (0,255,0), 1)
                cv2.imwrite(str((out_dir / 'preview' / Path(rel_path).name)), preview)

    holdout_txt = out_dir / 'manifests' / 'holdout_labels.txt'
    with holdout_txt.open('w', encoding='utf-8') as f:
        for row in accepted:
            f.write(f"{row['rel_path']} {row['text']}\n")

    accepted_tsv = out_dir / 'details' / 'accepted.tsv'
    with accepted_tsv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'province', 'text', 'rel_path', 'quad'], delimiter='\t')
        writer.writeheader()
        for row in accepted:
            out = dict(row)
            out['quad'] = json.dumps(out['quad'], ensure_ascii=False)
            writer.writerow(out)

    report = {
        'accepted_count': len(accepted),
        'province_distribution': {p: sum(1 for r in accepted if r['province'] == p) for p in PROVINCES},
        'holdout_txt': str(holdout_txt),
        'accepted_tsv': str(accepted_tsv),
        'preview_dir': str(out_dir / 'preview'),
    }
    (out_dir / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
