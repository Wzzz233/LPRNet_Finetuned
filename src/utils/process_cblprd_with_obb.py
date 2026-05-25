#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO


@dataclass
class Sample:
    image_path: Path
    rel_path: str
    text: str
    plate_type: str
    split: str
    family: str
    sub_type: str


def normalize_text(text: str) -> str:
    return (text or '').strip().upper()


def classify_sample(text: str, plate_type: str) -> Tuple[str, str, int]:
    text = normalize_text(text)
    plate_type = (plate_type or '').strip()
    if '新能源' in plate_type:
        return 'green8', 'green', 1
    if plate_type in {'普通蓝牌', '单层黄牌'}:
        return 'normal7', 'blue' if plate_type == '普通蓝牌' else 'yellow', 1 if plate_type == '普通蓝牌' else 0
    return 'special', 'special', 0


def safe_token(s: str) -> str:
    s = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff._-]+', '_', str(s))
    return s.strip('_') or 'x'


def make_name(sample: Sample, quad, suffix: str) -> str:
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    bbox = f'{x1}&{y1}_{x2}&{y2}'
    quad_text = '_'.join(f'{int(round(x))}&{int(round(y))}' for x, y in quad)
    return f'autoobb-0-{bbox}-{quad_text}-{safe_token(sample.text)}-0-0-{safe_token(sample.rel_path[:-4])}.{suffix}'


def read_txt(txt_path: Path, root: Path, split: str) -> List[Sample]:
    out = []
    with txt_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rel = parts[0].replace('\\', '/')
            text = parts[1]
            plate_type = ''.join(parts[2:])
            family, sub_type, _ = classify_sample(text, plate_type)
            out.append(Sample(
                image_path=root / rel,
                rel_path=rel,
                text=normalize_text(text),
                plate_type=plate_type,
                split=split,
                family=family,
                sub_type=sub_type,
            ))
    return out


def pick_best_obb(res, conf_thres: float):
    if getattr(res, 'obb', None) is None or res.obb is None or res.obb.conf is None:
        return None
    conf = res.obb.conf.cpu().numpy().tolist()
    if not conf:
        return None
    quads = res.obb.xyxyxyxy.cpu().numpy().tolist()
    idx = max(range(len(conf)), key=lambda i: conf[i])
    if conf[idx] < conf_thres:
        return None
    return {'conf': float(conf[idx]), 'quad': quads[idx], 'det_count': len(conf)}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description='Run obb_best on CBLPRD and split success/failed by family.')
    ap.add_argument('--root', default='/home/wzzz/LPRNet/CBLPRD-330k_v1')
    ap.add_argument('--weights', default='/home/wzzz/LPRNet/external_detectors/obb_best.pt')
    ap.add_argument('--out-root', default='/home/wzzz/LPRNet/cblprd_obb_autolabel_v1')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start-index', type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    samples = read_txt(root / 'train.txt', root, 'train') + read_txt(root / 'val.txt', root, 'val')
    missing_rows = []
    filtered = []
    for s in samples:
        if s.image_path.exists():
            filtered.append(s)
        else:
            missing_rows.append({
                'image_path': str(s.image_path),
                'rel_path': s.rel_path,
                'text': s.text,
                'plate_type': s.plate_type,
                'split': s.split,
                'family': s.family,
                'sub_type': s.sub_type,
                'status': 'failed',
                'output_image': '',
                'reason': 'missing_image',
            })
    samples = filtered
    if args.start_index > 0:
        samples = samples[args.start_index:]
    if args.limit > 0:
        samples = samples[:args.limit]

    model = YOLO(args.weights)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    success_rows = []
    failed_rows = list(missing_rows)
    labels = defaultdict(list)
    group_counter = defaultdict(lambda: Counter(total=0, success=0, fail=0))

    for start in range(0, len(samples), args.batch):
        batch_samples = samples[start:start + args.batch]
        batch_paths = [str(s.image_path) for s in batch_samples]
        results = model.predict(source=batch_paths, conf=args.conf, verbose=False, batch=args.batch)
        for sample, res in zip(batch_samples, results):
            key = f'{sample.split}:{sample.family}:{sample.plate_type}'
            group_counter[key]['total'] += 1
            det = pick_best_obb(res, args.conf)
            if det is None:
                fail_dir = out_root / 'failed' / sample.family / sample.split / safe_token(sample.plate_type)
                ensure_dir(fail_dir)
                dst = fail_dir / sample.image_path.name
                if sample.image_path.exists() and not dst.exists():
                    shutil.copy2(sample.image_path, dst)
                failed_rows.append({
                    'image_path': str(sample.image_path),
                    'rel_path': sample.rel_path,
                    'text': sample.text,
                    'plate_type': sample.plate_type,
                    'split': sample.split,
                    'family': sample.family,
                    'sub_type': sample.sub_type,
                    'status': 'failed',
                    'output_image': str(dst),
                    'reason': 'no_detection',
                })
                group_counter[key]['fail'] += 1
                continue

            suffix = sample.image_path.suffix.lstrip('.') or 'jpg'
            out_name = make_name(sample, det['quad'], suffix)
            succ_dir = out_root / 'success' / sample.family / sample.split / safe_token(sample.plate_type)
            ensure_dir(succ_dir)
            dst = succ_dir / out_name
            if not dst.exists():
                shutil.copy2(sample.image_path, dst)
            rel = str(dst.relative_to(out_root)).replace('\\', '/')
            labels[(sample.family, sample.split)].append((rel, sample.text))
            success_rows.append({
                'image_path': str(sample.image_path),
                'rel_path': sample.rel_path,
                'text': sample.text,
                'plate_type': sample.plate_type,
                'split': sample.split,
                'family': sample.family,
                'sub_type': sample.sub_type,
                'status': 'success',
                'det_conf': det['conf'],
                'det_count': det['det_count'],
                'output_image': str(dst),
                'quad': det['quad'],
            })
            group_counter[key]['success'] += 1

        if (start // args.batch) % 50 == 0:
            print(json.dumps({'processed': start + len(batch_samples), 'total': len(samples)}, ensure_ascii=False), flush=True)

    with (out_root / 'success_records.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_path', 'rel_path', 'text', 'plate_type', 'split', 'family', 'sub_type', 'status', 'det_conf', 'det_count', 'output_image', 'quad'])
        w.writeheader(); w.writerows(success_rows)
    with (out_root / 'failed_records.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_path', 'rel_path', 'text', 'plate_type', 'split', 'family', 'sub_type', 'status', 'output_image', 'reason'])
        w.writeheader(); w.writerows(failed_rows)

    label_root = out_root / 'labels'
    for (family, split), rows in labels.items():
        ensure_dir(label_root / family)
        with (label_root / family / f'{split}.txt').open('w', encoding='utf-8') as f:
            for rel, text in rows:
                f.write(f'{rel} {text}\n')

    report = {
        'total_samples': len(samples),
        'success_count': len(success_rows),
        'fail_count': len(failed_rows),
        'groups': {k: dict(v) for k, v in sorted(group_counter.items())},
        'success_csv': str(out_root / 'success_records.csv'),
        'failed_csv': str(out_root / 'failed_records.csv'),
    }
    (out_root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
