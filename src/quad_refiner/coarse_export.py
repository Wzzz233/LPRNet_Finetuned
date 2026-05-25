from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from load_data import order_quad_points, quad_to_box
from quad_refiner.dataset import build_crpd_records


def load_split_paths(root, split_file):
    root = Path(root)
    split_file = Path(split_file)
    paths = []
    with split_file.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rel = parts[0]
            path = root / rel
            if path.exists():
                paths.append(path)
    return paths


def extract_obb_detections(result):
    obb = getattr(result, 'obb', None)
    if obb is None or getattr(obb, 'conf', None) is None:
        return []
    conf = obb.conf.cpu().numpy().tolist()
    if not conf:
        return []
    quads = obb.xyxyxyxy.cpu().numpy().tolist()
    cls = obb.cls.cpu().numpy().tolist() if getattr(obb, 'cls', None) is not None else [0] * len(conf)
    out = []
    for idx in range(len(conf)):
        out.append({
            'conf': float(conf[idx]),
            'cls': int(cls[idx]) if idx < len(cls) else 0,
            'det_count': len(conf),
            'quad': order_quad_points(np.asarray(quads[idx], dtype=np.float32)).tolist(),
        })
    return out


def choose_best_obb(result):
    dets = extract_obb_detections(result)
    if not dets:
        return None
    return max(dets, key=lambda x: x['conf'])


def build_ccpd_coarse_record(image_path, source_name: str, det: dict, split: str | None = None):
    image_path = Path(image_path)
    rec = {
        'sample_id': f'{source_name}:{image_path.stem}',
        'image_path': str(image_path),
        'source_name': str(source_name),
        'coarse_quad': det['quad'],
        'det_conf': float(det['conf']),
        'det_cls': int(det.get('cls', 0)),
        'det_count': int(det.get('det_count', 1)),
    }
    if split is not None:
        rec['split'] = str(split)
    return rec


def _bbox_iou_from_quads(quad_a, quad_b):
    a = quad_to_box(np.asarray(quad_a, dtype=np.float32))
    b = quad_to_box(np.asarray(quad_b, dtype=np.float32))
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = float((ix2 - ix1 + 1) * (iy2 - iy1 + 1))
    area_a = float(a.w * a.h)
    area_b = float(b.w * b.h)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def match_detections_to_gt(gt_records, detections, min_iou: float = 0.05):
    candidates = []
    for gi, gt in enumerate(gt_records):
        for di, det in enumerate(detections):
            iou = _bbox_iou_from_quads(gt['gt_quad'], det['quad'])
            if iou >= float(min_iou):
                candidates.append((iou, det['conf'], gi, di))
    candidates.sort(reverse=True)
    used_gt = set()
    used_det = set()
    rows = []
    for iou, _conf, gi, di in candidates:
        if gi in used_gt or di in used_det:
            continue
        gt = gt_records[gi]
        det = detections[di]
        used_gt.add(gi)
        used_det.add(di)
        rows.append({
            'sample_id': gt['sample_id'],
            'image_path': gt.get('image_path', ''),
            'source_name': gt.get('source_name', ''),
            'split': gt.get('split'),
            'text': gt.get('text', ''),
            'coarse_quad': det['quad'],
            'det_conf': float(det['conf']),
            'det_cls': int(det.get('cls', 0)),
            'det_count': int(det.get('det_count', len(detections))),
            'match_iou': float(iou),
        })
    rows.sort(key=lambda x: x['sample_id'])
    return rows


def build_crpd_coarse_records(image_path, label_path, split: str, source_name: str, detections, min_iou: float = 0.05):
    gt_records = build_crpd_records(image_path, label_path, split=split, source_name=source_name)
    return match_detections_to_gt(gt_records, detections, min_iou=min_iou)


def iter_ccpd2019_paths(root, split: str, split_file: str = ''):
    root = Path(root)
    file_path = Path(split_file) if split_file else root / f'{split}.txt'
    if not file_path.exists():
        return []
    return load_split_paths(root, file_path)


def iter_ccpd2020_green_paths(root, split: str):
    root = Path(root) / split
    if not root.exists():
        return []
    return sorted(root.rglob('*.jpg'))


def iter_crpd_pairs(root, split: str):
    root = Path(root)
    pairs = []
    for subset in ['CRPD_single', 'CRPD_double', 'CRPD_multi']:
        img_dir = root / subset / split / 'images'
        label_dir = root / subset / split / 'labels'
        if not img_dir.exists() or not label_dir.exists():
            continue
        for img_path in sorted(img_dir.glob('*.jpg')):
            label_path = label_dir / f'{img_path.stem}.txt'
            if label_path.exists():
                pairs.append((subset, img_path, label_path))
    return pairs


def write_jsonl(records: Iterable[dict], output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def export_paths_to_coarse_jsonl(paths, model, source_name: str, output_path, conf_thres: float = 0.25, split: str | None = None, imgsz: int = 640):
    records = []
    misses = []
    for image_path in paths:
        result = model.predict(source=str(image_path), conf=conf_thres, imgsz=imgsz, verbose=False)[0]
        det = choose_best_obb(result)
        if det is None or float(det['conf']) < float(conf_thres):
            misses.append(str(image_path))
            continue
        records.append(build_ccpd_coarse_record(image_path, source_name=source_name, det=det, split=split))
    write_jsonl(records, output_path)
    summary = {
        'output_jsonl': str(output_path),
        'source_name': source_name,
        'count': len(records),
        'miss_count': len(misses),
        'conf_thres': float(conf_thres),
        'split': split,
    }
    Path(output_path).with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if misses:
        Path(output_path).with_suffix('.misses.txt').write_text('\n'.join(misses) + '\n', encoding='utf-8')
    return summary


def export_crpd_pairs_to_coarse_jsonl(pairs, model, output_path, conf_thres: float = 0.25, split: str | None = None, imgsz: int = 640, min_iou: float = 0.05):
    records = []
    misses = []
    for subset, image_path, label_path in pairs:
        result = model.predict(source=str(image_path), conf=conf_thres, imgsz=imgsz, verbose=False)[0]
        detections = extract_obb_detections(result)
        if not detections:
            misses.append(str(image_path))
            continue
        rows = build_crpd_coarse_records(image_path, label_path, split=split or '', source_name=f'crpd_raw/{subset}', detections=detections, min_iou=min_iou)
        if not rows:
            misses.append(str(image_path))
            continue
        records.extend(rows)
    write_jsonl(records, output_path)
    summary = {
        'output_jsonl': str(output_path),
        'source_name': 'crpd_raw',
        'count': len(records),
        'miss_count': len(misses),
        'conf_thres': float(conf_thres),
        'split': split,
        'min_iou': float(min_iou),
    }
    Path(output_path).with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if misses:
        Path(output_path).with_suffix('.misses.txt').write_text('\n'.join(misses) + '\n', encoding='utf-8')
    return summary
