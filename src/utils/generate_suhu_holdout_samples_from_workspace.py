#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROVINCE_DIR = {'沪': 'p02_u6caa', '苏': 'p10_u82cf'}


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    quad = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    sums = quad.sum(axis=1)
    diffs = quad[:, 1] - quad[:, 0]
    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = quad[int(np.argmin(sums))]
    out[2] = quad[int(np.argmax(sums))]
    out[1] = quad[int(np.argmin(diffs))]
    out[3] = quad[int(np.argmax(diffs))]
    return out


def quad_bbox(quad: np.ndarray, img_w: int, img_h: int):
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    q[:, 0] = np.clip(q[:, 0], 0.0, float(max(0, img_w - 1)))
    q[:, 1] = np.clip(q[:, 1], 0.0, float(max(0, img_h - 1)))
    x1 = int(np.floor(np.min(q[:, 0])))
    y1 = int(np.floor(np.min(q[:, 1])))
    x2 = int(np.ceil(np.max(q[:, 0])))
    y2 = int(np.ceil(np.max(q[:, 1])))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1, min(x2, img_w - 1))
    y2 = max(y1, min(y2, img_h - 1))
    return x1, y1, x2, y2


def alias_name(uid: str, quad: np.ndarray, img_w: int, img_h: int):
    q = order_quad_points(quad)
    x1, y1, x2, y2 = quad_bbox(q, img_w, img_h)
    q_int = np.rint(q).astype(np.int32)
    quad_part = '_'.join(f'{int(x)}&{int(y)}' for x, y in q_int)
    return f'genx-0-{x1}&{y1}_{x2}&{y2}-{quad_part}-{uid}.jpg'


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
    ap.add_argument('--holdout_txt', required=True)
    ap.add_argument('--src_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--yolo_weights', required=True)
    args = ap.parse_args()

    rows=[]
    for line in Path(args.holdout_txt).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        rel, text = line.strip().split(maxsplit=1)
        rows.append((rel, text.strip()))

    out_dir = Path(args.out_dir)
    img_dir = out_dir / 'images'
    details_dir = out_dir / 'details'
    manifests_dir = out_dir / 'manifests'
    preview_dir = out_dir / 'preview'
    for d in [img_dir, details_dir, manifests_dir, preview_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.yolo_weights)
    accepted = []
    rejected = []
    for idx, (rel, text) in enumerate(rows):
        src = Path(args.src_root) / rel
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            rejected.append({'rel_path': rel, 'text': text, 'reason': 'read_failed'})
            continue
        det = detect_quad(model, img)
        if det is None:
            rejected.append({'rel_path': rel, 'text': text, 'reason': 'detect_failed'})
            continue
        p = text[0]
        pdir = img_dir / PROVINCE_DIR.get(p, 'pXX')
        pdir.mkdir(parents=True, exist_ok=True)
        name = alias_name(f'holdout-{idx:05d}', det['quad'], img.shape[1], img.shape[0])
        out_img = pdir / name
        cv2.imwrite(str(out_img), img)
        rel_out = str(out_img.relative_to(out_dir)).replace('\\', '/')
        accepted.append({'rel_path': rel_out, 'text': text, 'province': p, 'conf': det['conf'], 'det_count': det['det_count'], 'source_rel_path': rel})
        if len(accepted) <= 20:
            preview = img.copy()
            poly = np.asarray(det['quad'], dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(preview, [poly], True, (0,255,0), 2)
            cv2.imwrite(str(preview_dir / name), preview)

    manifest = manifests_dir / 'holdout_labels.txt'
    with manifest.open('w', encoding='utf-8') as f:
        for r in accepted:
            f.write(f"{r['rel_path']} {r['text']}\n")

    with (details_dir / 'accepted.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rel_path','text','province','conf','det_count','source_rel_path'], delimiter='\t')
        writer.writeheader(); writer.writerows(accepted)
    with (details_dir / 'rejected.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rel_path','text','reason'], delimiter='\t')
        writer.writeheader(); writer.writerows(rejected)

    report = {
        'accepted_count': len(accepted),
        'rejected_count': len(rejected),
        'province_distribution': {
            '苏': sum(1 for r in accepted if r['province']=='苏'),
            '沪': sum(1 for r in accepted if r['province']=='沪'),
        },
        'manifest': str(manifest),
        'preview_dir': str(preview_dir),
        'examples': accepted[:10],
    }
    (out_dir / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
