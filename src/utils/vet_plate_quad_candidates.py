#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path('/home/wzzz/LPRNet')
OUT = ROOT / 'artifacts' / 'plate_quad_vetting'
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    'obb_best_local': ROOT / 'external_detectors' / 'obb_best.pt',
    'chinese_anpr_detect_local': ROOT / 'external_detectors' / 'chinese_anpr_yolov8_last.pt',
    'hf_koushim_detect': ROOT / 'external_detectors' / 'hf_koushim_best.pt',
}


def ensure_hf_weight():
    target = CANDIDATES['hf_koushim_detect']
    if target.exists() and target.stat().st_size > 1_000_000:
        return
    import requests
    url = 'https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt'
    r = requests.get(url, timeout=120, headers={'User-Agent': 'Hermes'}, allow_redirects=True)
    r.raise_for_status()
    target.write_bytes(r.content)


def parse_ccpd_quad_from_name(image_name: str):
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) < 4:
        return None
    pts = []
    try:
        for item in parts[3].split('_'):
            x, y = item.split('&', 1)
            pts.append((float(x), float(y)))
    except Exception:
        return None
    if len(pts) != 4:
        return None
    return np.asarray(pts, dtype=np.float32)


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    out = np.zeros((4,2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def quad_iou_like(pred: np.ndarray, gt: np.ndarray, shape_hw: Tuple[int,int]) -> float:
    h, w = shape_hw
    pred = order_quad(pred).astype(np.int32)
    gt = order_quad(gt).astype(np.int32)
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask1, pred, 1)
    cv2.fillConvexPoly(mask2, gt, 1)
    inter = int(np.logical_and(mask1, mask2).sum())
    union = int(np.logical_or(mask1, mask2).sum())
    return 0.0 if union == 0 else inter / union


def warp_from_quad(img: np.ndarray, quad: np.ndarray, out_w=94, out_h=24) -> np.ndarray:
    src = order_quad(quad).astype(np.float32)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def sharpness_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_ccpd_samples(n=12):
    # a few stable GT samples from ccpd2019 for objective geometry check
    items = sorted((ROOT/'CCPD2019'/'ccpd_base').glob('*.jpg'))
    out = []
    step = max(1, len(items)//(n*20))
    for p in items[::step]:
        q = parse_ccpd_quad_from_name(p.name)
        if q is not None:
            out.append(p)
        if len(out) >= n:
            break
    return out


def pick_plain_samples():
    out = []
    out += sorted((ROOT/'git_plate'/'val'/'val_verify').glob('*.jpg'))[:8]
    out += sorted((ROOT/'targeted_green_missing_18'/'val'/'gui').glob('*.jpg'))[:8]
    return out


def detect_best(model, img_path: Path):
    res = model.predict(source=str(img_path), conf=0.2, verbose=False)[0]
    if getattr(model, 'task', '') == 'obb' and getattr(res, 'obb', None) is not None and res.obb is not None and res.obb.conf is not None and len(res.obb.conf) > 0:
        conf = res.obb.conf.cpu().numpy()
        quads = res.obb.xyxyxyxy.cpu().numpy()
        i = int(np.argmax(conf))
        return {'type':'quad','conf':float(conf[i]),'quad':quads[i].tolist()}
    if getattr(res, 'boxes', None) is not None and res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
        conf = res.boxes.conf.cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()
        i = int(np.argmax(conf))
        x1,y1,x2,y2 = xyxy[i].tolist()
        quad = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        return {'type':'box_as_quad','conf':float(conf[i]),'quad':quad}
    return None


def main():
    ensure_hf_weight()
    ccpd = pick_ccpd_samples(12)
    plain = pick_plain_samples()
    overall = {'objective_ccpd':{}, 'plain_samples':{}, 'candidates':{}}

    for name, path in CANDIDATES.items():
        if not path.exists():
            continue
        model = YOLO(str(path))
        cand_dir = OUT / name
        cand_dir.mkdir(parents=True, exist_ok=True)
        obj_records = []
        ious = []
        sharps = []
        detected = 0
        for p in ccpd:
            img = cv2.imread(str(p))
            gt = parse_ccpd_quad_from_name(p.name)
            det = detect_best(model, p)
            rec = {'image': str(p), 'detected': det is not None}
            if det is not None:
                detected += 1
                pred = np.asarray(det['quad'], dtype=np.float32)
                iou = quad_iou_like(pred, gt, img.shape[:2])
                warp = warp_from_quad(img, pred)
                sharp = sharpness_score(warp)
                ious.append(iou)
                sharps.append(sharp)
                rec.update({'conf': det['conf'], 'iou_like_vs_gt': iou, 'warp_sharpness': sharp, 'det_type': det['type']})
                vis = img.copy()
                cv2.polylines(vis, [order_quad(gt).astype(np.int32)], True, (0,255,0), 2)
                cv2.polylines(vis, [order_quad(pred).astype(np.int32)], True, (0,0,255), 2)
                pair = np.concatenate([cv2.resize(vis, (img.shape[1], img.shape[0])), cv2.resize(warp, (img.shape[1], img.shape[0]))], axis=1)
                cv2.imwrite(str(cand_dir / f'ccpd_{p.name}'), pair)
            obj_records.append(rec)
        overall['objective_ccpd'][name] = {
            'task': model.task,
            'detected': detected,
            'total': len(ccpd),
            'detection_rate': detected / len(ccpd) if ccpd else 0,
            'mean_iou_like_vs_gt': (sum(ious)/len(ious) if ious else 0),
            'mean_warp_sharpness': (sum(sharps)/len(sharps) if sharps else 0),
        }

        plain_records = []
        for p in plain:
            img = cv2.imread(str(p))
            det = detect_best(model, p)
            rec = {'image': str(p), 'detected': det is not None}
            if det is not None:
                pred = np.asarray(det['quad'], dtype=np.float32)
                warp = warp_from_quad(img, pred)
                sharp = sharpness_score(warp)
                rec.update({'conf': det['conf'], 'warp_sharpness': sharp, 'det_type': det['type']})
                vis = img.copy()
                cv2.polylines(vis, [order_quad(pred).astype(np.int32)], True, (0,0,255), 2)
                pair = np.concatenate([cv2.resize(vis, (img.shape[1], img.shape[0])), cv2.resize(warp, (img.shape[1], img.shape[0]))], axis=1)
                cv2.imwrite(str(cand_dir / f'plain_{p.name}'), pair)
            plain_records.append(rec)
        overall['plain_samples'][name] = plain_records
        overall['candidates'][name] = {'path': str(path), 'task': model.task}

    (OUT/'summary.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
