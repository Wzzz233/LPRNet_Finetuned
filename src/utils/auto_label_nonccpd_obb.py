#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from ultralytics import YOLO


@dataclass
class Sample:
    image_path: Path
    text: str
    split: str
    dataset_name: str
    source_name: str
    family: str
    sub_type: str
    rel_key: str


def normalize_text(text: str) -> str:
    return (text or '').strip().upper()


def family_from_text(text: str) -> Tuple[str, str]:
    t = normalize_text(text)
    if len(t) == 8:
        return 'green8', 'green'
    if len(t) == 7:
        return 'normal7', 'blue'
    return 'special', 'special'


def parse_targeted_txt(txt_path: Path, root: Path, split: str) -> List[Sample]:
    samples = []
    with txt_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, text = line.split(maxsplit=1)
            rel = rel.replace('\\', '/')
            marker = 'targeted_green_missing_18/'
            idx = rel.find(marker)
            if idx >= 0:
                rel = rel[idx:]
            img_path = root / rel
            family, sub_type = family_from_text(text)
            samples.append(Sample(
                image_path=img_path,
                text=normalize_text(text),
                split=split,
                dataset_name='targeted_green_missing_18',
                source_name='targeted_green_missing_18',
                family=family,
                sub_type=sub_type,
                rel_key=str(Path(rel).with_suffix('')).replace('\\', '/'),
            ))
    return samples


def parse_git_plate_dir(dir_path: Path) -> List[Sample]:
    samples = []
    for img_path in sorted(dir_path.glob('*.jpg')):
        stem = img_path.stem
        text = stem.split('_')[0]
        family, sub_type = family_from_text(text)
        samples.append(Sample(
            image_path=img_path,
            text=normalize_text(text),
            split='val',
            dataset_name='git_plate',
            source_name='git_plate_val_verify',
            family=family,
            sub_type=sub_type,
            rel_key=img_path.stem,
        ))
    return samples


def safe_token(s: str) -> str:
    s = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff._-]+', '_', s)
    return s.strip('_') or 'x'


def box_from_quad(quad):
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    x1 = int(max(0, min(xs)))
    y1 = int(max(0, min(ys)))
    x2 = int(max(0, max(xs)))
    y2 = int(max(0, max(ys)))
    return x1, y1, x2, y2


def quad_to_ccpd_text(quad) -> str:
    return '_'.join(f'{int(round(x))}&{int(round(y))}' for x, y in quad)


def make_pseudo_ccpd_name(sample: Sample, quad, suffix='jpg') -> str:
    x1, y1, x2, y2 = box_from_quad(quad)
    bbox = f'{x1}&{y1}_{x2}&{y2}'
    quad_text = quad_to_ccpd_text(quad)
    text_tag = safe_token(sample.text)
    origin_tag = safe_token(sample.rel_key)
    # 只要求保持 CCPD 关键槽位：第3段 bbox，第4段 quad
    return f'autoobb-0-{bbox}-{quad_text}-{text_tag}-0-0-{origin_tag}.{suffix}'


def pick_best_obb(result):
    if getattr(result, 'obb', None) is None or result.obb is None or result.obb.conf is None:
        return None
    conf = result.obb.conf.cpu().numpy().tolist()
    if not conf:
        return None
    quads = result.obb.xyxyxyxy.cpu().numpy().tolist()
    cls = result.obb.cls.cpu().numpy().tolist() if result.obb.cls is not None else [0] * len(conf)
    idx = max(range(len(conf)), key=lambda i: conf[i])
    return {
        'conf': float(conf[idx]),
        'cls': int(cls[idx]) if idx < len(cls) else 0,
        'quad': quads[idx],
        'det_count': len(conf),
    }


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_label_txt(records: Iterable[Tuple[str, str]], out_path: Path):
    ensure_dir(out_path.parent)
    with out_path.open('w', encoding='utf-8') as f:
        for rel, text in records:
            f.write(f'{rel} {text}\n')


def process_samples(samples: List[Sample], model: YOLO, out_root: Path, conf_thres: float, limit: int = 0):
    if limit > 0:
        samples = samples[:limit]

    success_rows = []
    fail_rows = []
    label_records: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    summary: Dict[str, Dict[str, int]] = {}

    for sample in samples:
        key = f'{sample.dataset_name}:{sample.split}:{sample.family}'
        summary.setdefault(key, {'total': 0, 'success': 0, 'fail': 0})
        summary[key]['total'] += 1

        if not sample.image_path.exists():
            fail_reason = 'missing_image'
            det = None
        else:
            result = model.predict(source=str(sample.image_path), conf=conf_thres, verbose=False)[0]
            det = pick_best_obb(result)
            fail_reason = 'no_detection'
            if det is not None and det['conf'] < conf_thres:
                det = None
                fail_reason = 'low_conf'

        if det is None:
            fail_dir = out_root / 'failed' / sample.family / sample.dataset_name / sample.split
            ensure_dir(fail_dir)
            target = fail_dir / sample.image_path.name
            if sample.image_path.exists() and not target.exists():
                shutil.copy2(sample.image_path, target)
            fail_rows.append({
                'image_path': str(sample.image_path),
                'dataset_name': sample.dataset_name,
                'split': sample.split,
                'family': sample.family,
                'sub_type': sample.sub_type,
                'text': sample.text,
                'status': 'failed',
                'reason': fail_reason,
                'output_image': str(target),
            })
            summary[key]['fail'] += 1
            continue

        quad = det['quad']
        suffix = sample.image_path.suffix.lstrip('.') or 'jpg'
        pseudo_name = make_pseudo_ccpd_name(sample, quad, suffix=suffix)
        success_dir = out_root / 'success' / sample.family / sample.dataset_name / sample.split
        ensure_dir(success_dir)
        out_img = success_dir / pseudo_name
        shutil.copy2(sample.image_path, out_img)
        rel = str(out_img.relative_to(out_root)).replace('\\', '/')
        label_records.setdefault((sample.family, sample.split), []).append((rel, sample.text))
        success_rows.append({
            'image_path': str(sample.image_path),
            'dataset_name': sample.dataset_name,
            'split': sample.split,
            'family': sample.family,
            'sub_type': sample.sub_type,
            'text': sample.text,
            'status': 'success',
            'reason': 'detected',
            'det_conf': det['conf'],
            'det_count': det['det_count'],
            'output_image': str(out_img),
            'bbox': box_from_quad(quad),
            'quad': quad,
        })
        summary[key]['success'] += 1

    ensure_dir(out_root)
    with (out_root / 'success_records.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['image_path', 'dataset_name', 'split', 'family', 'sub_type', 'text', 'status', 'reason', 'det_conf', 'det_count', 'output_image', 'bbox', 'quad']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(success_rows)
    with (out_root / 'failed_records.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['image_path', 'dataset_name', 'split', 'family', 'sub_type', 'text', 'status', 'reason', 'output_image']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(fail_rows)

    for (family, split), rows in sorted(label_records.items()):
        txt_path = out_root / 'labels' / family / f'{split}.txt'
        write_label_txt(rows, txt_path)

    report = {
        'out_root': str(out_root),
        'conf_thres': conf_thres,
        'total_samples': len(samples),
        'success_count': len(success_rows),
        'fail_count': len(fail_rows),
        'groups': summary,
        'success_records_csv': str(out_root / 'success_records.csv'),
        'failed_records_csv': str(out_root / 'failed_records.csv'),
    }
    (out_root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def collect_samples(root: Path) -> List[Sample]:
    samples = []
    samples.extend(parse_targeted_txt(root / 'targeted_green_missing_18' / 'train.txt', root, 'train'))
    samples.extend(parse_targeted_txt(root / 'targeted_green_missing_18' / 'val.txt', root, 'val'))
    git_dir = root / 'git_plate' / 'val' / 'val_verify'
    if git_dir.exists():
        samples.extend(parse_git_plate_dir(git_dir))
    return samples


def main():
    ap = argparse.ArgumentParser(description='Use obb_best.pt to auto-label non-CCPD plates and emit CCPD-like filenames.')
    ap.add_argument('--root', default='/home/wzzz/LPRNet')
    ap.add_argument('--weights', default='/home/wzzz/LPRNet/external_detectors/obb_best.pt')
    ap.add_argument('--out-root', default='/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    model = YOLO(args.weights)
    samples = collect_samples(root)
    process_samples(samples, model, Path(args.out_root), args.conf, limit=args.limit)


if __name__ == '__main__':
    main()
