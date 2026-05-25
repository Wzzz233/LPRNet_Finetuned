#!/usr/bin/env python3
import argparse
import csv
import json
import math
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


def load_manifest_rows(txt_path):
    rows = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rel, text = line.split(maxsplit=1)
            rows.append({'rel_path': rel.replace('\\','/'), 'text': text.strip()})
    return rows


def ensure_repo_imports(repo_root):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        from generate_chars_image import CharsImageGenerator
        from generate_plate_template import LicensePlateImageGenerator
        from augment_image import ImageAugmentation
        chars_gen = CharsImageGenerator('small_new_energy')
        template_gen = LicensePlateImageGenerator('small_new_energy')
        template = template_gen.generate_template_image(chars_gen.plate_width, chars_gen.plate_height)
        repo_augmenter = ImageAugmentation('small_new_energy', template)
        repo_augmenter.env_data_paths = [os.path.abspath(os.path.join(repo_root, path)) for path in repo_augmenter.env_data_paths]
        repo_augmenter.smu = cv2.imread(os.path.abspath(os.path.join(repo_root, 'images', 'smu.jpg')))
    finally:
        os.chdir(old_cwd)
    return chars_gen, repo_augmenter


def make_random_green_plate(province, used_texts, rng):
    while True:
        text = province + rng.choice(LETTERS_NO_IO) + rng.choice(['D', 'F']) + rng.choice(ALNUM_NO_IO) + ''.join(rng.choice(DIGITS) for _ in range(4))
        if text not in used_texts:
            used_texts.add(text)
            return text


def compose(chars_gen, repo_augmenter, text, rng, out_w=246, out_h=72):
    render_item = chars_gen.generate_images_with_metadata([text])[0]
    chars_img = render_item['image']
    clean = repo_augmenter.augment(chars_img, horizontal_sight_direction=rng.choice(('left','mid','right')), vertical_sight_direction=rng.choice(('up','mid','down')))
    clean = cv2.resize(clean, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return clean


def apply_tier(img, rng):
    out = img.copy()
    if rng.random() < 0.8:
        q = rng.randint(45, 88)
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    if rng.random() < 0.7:
        sigma = rng.uniform(0.2, 1.4)
        out = cv2.GaussianBlur(out, (3,3), sigma)
    if rng.random() < 0.5:
        alpha = rng.uniform(0.88, 1.12)
        beta = rng.uniform(-12, 12)
        out = np.clip(out.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        noise = np.random.normal(0.0, rng.uniform(1.0, 10.0), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def order_quad_points(pts):
    quad = np.asarray(pts, dtype=np.float32).reshape(4,2)
    sums = quad.sum(axis=1)
    diffs = quad[:,1] - quad[:,0]
    out = np.zeros((4,2), dtype=np.float32)
    out[0] = quad[int(np.argmin(sums))]
    out[2] = quad[int(np.argmax(sums))]
    out[1] = quad[int(np.argmin(diffs))]
    out[3] = quad[int(np.argmax(diffs))]
    return out


def quad_bbox(quad, img_w, img_h):
    q = np.asarray(quad, dtype=np.float32).reshape(4,2)
    q[:,0] = np.clip(q[:,0], 0.0, float(max(0,img_w-1)))
    q[:,1] = np.clip(q[:,1], 0.0, float(max(0,img_h-1)))
    x1 = int(np.floor(np.min(q[:,0]))); y1 = int(np.floor(np.min(q[:,1])))
    x2 = int(np.ceil(np.max(q[:,0]))); y2 = int(np.ceil(np.max(q[:,1])))
    x1=max(0,min(x1,img_w-1)); y1=max(0,min(y1,img_h-1)); x2=max(x1,min(x2,img_w-1)); y2=max(y1,min(y2,img_h-1))
    return x1,y1,x2,y2


def alias_name(uid, quad, img_w, img_h):
    q = order_quad_points(quad)
    x1,y1,x2,y2 = quad_bbox(q, img_w, img_h)
    q_int = np.rint(q).astype(np.int32)
    quad_part = '_'.join(f'{int(x)}&{int(y)}' for x,y in q_int)
    return f'fresh-0-{x1}&{y1}_{x2}&{y2}-{quad_part}-{uid}.jpg'


def detect_quad(model, img_bgr, imgsz=640, conf=0.25, predict_conf=0.001):
    pred = model.predict(source=img_bgr, imgsz=imgsz, conf=predict_conf, verbose=False)[0]
    if pred.obb is None or len(pred.obb) == 0:
        return None
    confs = pred.obb.conf.cpu().numpy()
    if confs is None or len(confs) == 0:
        return None
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])
    if best_conf < conf:
        return None
    pts = pred.obb.xyxyxyxy.cpu().numpy()[best_idx]
    return {'quad': order_quad_points(pts), 'conf': best_conf, 'det_count': int(len(confs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--audit_dir', required=True)
    ap.add_argument('--yolo_weights', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_province', type=int, default=40)
    ap.add_argument('--oversupply', type=int, default=70)
    ap.add_argument('--seed', type=int, default=20260403)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2**32 - 1))

    audit = json.loads((Path(args.audit_dir) / 'audit_report.json').read_text(encoding='utf-8'))
    train_rows = load_manifest_rows(audit['cleaned_manifests']['train'])
    val_rows = load_manifest_rows(audit['cleaned_manifests']['val'])
    test_rows = load_manifest_rows(audit['cleaned_manifests']['test'])
    used_texts = {r['text'] for r in train_rows + val_rows + test_rows}

    chars_gen, repo_augmenter = ensure_repo_imports(args.repo_root)
    from ultralytics import YOLO
    model = YOLO(args.yolo_weights)

    out_dir = Path(args.out_dir)
    img_root = out_dir / 'images'
    prev_root = out_dir / 'preview'
    det_root = out_dir / 'details'
    man_root = out_dir / 'manifests'
    for d in [img_root, prev_root, det_root, man_root]:
        d.mkdir(parents=True, exist_ok=True)

    accepted = []
    rejected = []
    for prov in PROVINCES:
        target = int(args.per_province)
        attempt = 0
        kept = 0
        while kept < target and attempt < int(args.oversupply):
            text = make_random_green_plate(prov, used_texts, rng)
            img = compose(chars_gen, repo_augmenter, text, rng)
            img = apply_tier(img, rng)
            det = detect_quad(model, img)
            uid = f'{prov}-{attempt:04d}'
            attempt += 1
            if det is None:
                rejected.append({'province': prov, 'text': text, 'reason': 'detect_failed'})
                continue
            prov_dir = img_root / ('p02_u6caa' if prov == '沪' else 'p10_u82cf')
            prov_dir.mkdir(parents=True, exist_ok=True)
            name = alias_name(uid, det['quad'], img.shape[1], img.shape[0])
            out_img = prov_dir / name
            cv2.imwrite(str(out_img), img)
            rel = str(out_img.relative_to(out_dir)).replace('\\','/')
            accepted.append({'rel_path': rel, 'text': text, 'province': prov, 'conf': det['conf'], 'det_count': det['det_count']})
            if kept < 10:
                preview = img.copy()
                poly = np.asarray(det['quad'], dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(preview, [poly], True, (0,255,0), 2)
                cv2.imwrite(str(prev_root / name), preview)
            kept += 1

    holdout_txt = man_root / 'holdout_labels.txt'
    with holdout_txt.open('w', encoding='utf-8') as f:
        for r in accepted:
            f.write(f"{r['rel_path']} {r['text']}\n")

    with (det_root / 'accepted.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rel_path','text','province','conf','det_count'], delimiter='\t')
        writer.writeheader(); writer.writerows(accepted)
    with (det_root / 'rejected.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['province','text','reason'], delimiter='\t')
        writer.writeheader(); writer.writerows(rejected)

    report = {
        'accepted_count': len(accepted),
        'rejected_count': len(rejected),
        'province_distribution': {
            '沪': sum(1 for r in accepted if r['province']=='沪'),
            '苏': sum(1 for r in accepted if r['province']=='苏'),
        },
        'manifest': str(holdout_txt),
        'preview_dir': str(prev_root),
        'examples': accepted[:10],
    }
    (out_dir / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
