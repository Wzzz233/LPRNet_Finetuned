#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ultralytics import YOLO

SUPPORTED_CHARS = set([
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'I', 'O', '-'
])

PLATE_TYPE_META = {
    '普通蓝牌': ('normal7', 'blue'),
    '新能源小型车': ('green8', 'green_small'),
    '新能源大型车': ('green8', 'green_large'),
    '单层黄牌': ('special', 'yellow_single'),
    '双层黄牌': ('special', 'yellow_double'),
    '拖拉机绿牌': ('special', 'tractor_green'),
    '黑色车牌': ('special', 'black'),
}


@dataclass
class Sample:
    image_path: Path
    rel_path: str
    text: str
    plate_type: str
    split: str
    family: str
    sub_type: str


def safe_token(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or '\u4e00' <= ch <= '\u9fff' or ch in '._-':
            out.append(ch)
        else:
            out.append('_')
    token = ''.join(out).strip('_')
    return token or 'x'


def text_supported(text: str) -> bool:
    return all(ch in SUPPORTED_CHARS for ch in text)


def parse_txt(txt_path: Path, root: Path, split: str) -> Tuple[List[Sample], List[dict]]:
    samples: List[Sample] = []
    skipped: List[dict] = []
    with txt_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                skipped.append({'split': split, 'line_no': line_no, 'reason': 'bad_line', 'raw': line})
                continue
            rel_path, text, plate_type = parts
            family, sub_type = PLATE_TYPE_META.get(plate_type, ('special', safe_token(plate_type)))
            image_path = root / rel_path
            if not image_path.exists():
                skipped.append({'split': split, 'line_no': line_no, 'reason': 'missing_image', 'rel_path': rel_path, 'text': text, 'plate_type': plate_type})
                continue
            if not text_supported(text):
                bad_chars = ''.join(sorted({ch for ch in text if ch not in SUPPORTED_CHARS}))
                skipped.append({'split': split, 'line_no': line_no, 'reason': 'unsupported_chars', 'rel_path': rel_path, 'text': text, 'plate_type': plate_type, 'unsupported_chars': bad_chars})
                continue
            samples.append(Sample(
                image_path=image_path,
                rel_path=rel_path.replace('\\', '/'),
                text=text,
                plate_type=plate_type,
                split=split,
                family=family,
                sub_type=sub_type,
            ))
    return samples, skipped


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
    origin_tag = safe_token(Path(sample.rel_path).with_suffix('').as_posix())
    ptype_tag = safe_token(sample.plate_type)
    return f'autoobb-0-{bbox}-{quad_text}-{text_tag}-{ptype_tag}-0-{origin_tag}.{suffix}'


def pick_best_obb(result):
    if getattr(result, 'obb', None) is None or result.obb is None or result.obb.conf is None:
        return None
    conf = result.obb.conf.cpu().numpy().tolist()
    if not conf:
        return None
    quads = result.obb.xyxyxyxy.cpu().numpy().tolist()
    idx = max(range(len(conf)), key=lambda i: conf[i])
    return {
        'conf': float(conf[idx]),
        'quad': quads[idx],
        'det_count': len(conf),
    }


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_symlink(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except FileExistsError:
        return


def batched(items: List[Sample], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def process_samples(samples: List[Sample], model: YOLO, out_root: Path, conf_thres: float, batch_size: int):
    success_rows = []
    fail_rows = []
    summary = Counter()
    family_split_counter = Counter()

    for chunk in batched(samples, batch_size):
        paths = [str(s.image_path) for s in chunk]
        results = model.predict(source=paths, conf=conf_thres, verbose=False, batch=batch_size)
        for sample, result in zip(chunk, results):
            det = pick_best_obb(result)
            key = (sample.split, sample.family, sample.sub_type)
            family_split_counter[key] += 1
            if det is None or det['conf'] < conf_thres:
                fail_rows.append({
                    'image_path': str(sample.image_path),
                    'rel_path': sample.rel_path,
                    'split': sample.split,
                    'family': sample.family,
                    'sub_type': sample.sub_type,
                    'plate_type': sample.plate_type,
                    'text': sample.text,
                    'status': 'failed',
                    'reason': 'no_detection',
                })
                summary['fail'] += 1
                continue
            quad = det['quad']
            suffix = sample.image_path.suffix.lstrip('.') or 'jpg'
            pseudo_name = make_pseudo_ccpd_name(sample, quad, suffix=suffix)
            out_img = out_root / 'success' / sample.family / sample.split / pseudo_name
            make_symlink(sample.image_path.resolve(), out_img)
            success_rows.append({
                'image_path': str(sample.image_path),
                'rel_path': sample.rel_path,
                'split': sample.split,
                'family': sample.family,
                'sub_type': sample.sub_type,
                'plate_type': sample.plate_type,
                'text': sample.text,
                'status': 'success',
                'reason': 'detected',
                'det_conf': det['conf'],
                'det_count': det['det_count'],
                'output_image': str(out_img),
                'bbox': box_from_quad(quad),
                'quad': quad,
            })
            summary['success'] += 1

    ensure_dir(out_root)
    with (out_root / 'success_records.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['image_path', 'rel_path', 'split', 'family', 'sub_type', 'plate_type', 'text', 'status', 'reason', 'det_conf', 'det_count', 'output_image', 'bbox', 'quad']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(success_rows)
    with (out_root / 'failed_records.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['image_path', 'rel_path', 'split', 'family', 'sub_type', 'plate_type', 'text', 'status', 'reason']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(fail_rows)

    report = {
        'out_root': str(out_root),
        'conf_thres': conf_thres,
        'total_supported_samples': len(samples),
        'success_count': len(success_rows),
        'fail_count': len(fail_rows),
        'by_split_family_subtype': {
            f'{split}:{family}:{sub_type}': count
            for (split, family, sub_type), count in sorted(family_split_counter.items())
        },
    }
    (out_root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description='Use OBB detector to pseudo-label CBLPRD and create symlinked CCPD-like success samples.')
    ap.add_argument('--root', default='/home/wzzz/LPRNet/CBLPRD-330k_v1')
    ap.add_argument('--weights', default='/home/wzzz/LPRNet/external_detectors/obb_best.pt')
    ap.add_argument('--out-root', default='/home/wzzz/LPRNet/cblprd_obb_autolabel_v1')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--batch-size', type=int, default=128)
    args = ap.parse_args()

    root = Path(args.root)
    model = YOLO(args.weights)
    train_samples, train_skipped = parse_txt(root / 'train.txt', root, 'train')
    val_samples, val_skipped = parse_txt(root / 'val.txt', root, 'val')
    all_samples = train_samples + val_samples

    out_root = Path(args.out_root)
    ensure_dir(out_root)
    with (out_root / 'skipped_unsupported_or_missing.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['split', 'line_no', 'reason', 'rel_path', 'text', 'plate_type', 'unsupported_chars', 'raw']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(train_skipped + val_skipped)

    support_report = {
        'train_supported': len(train_samples),
        'val_supported': len(val_samples),
        'supported_total': len(all_samples),
        'skipped_total': len(train_skipped) + len(val_skipped),
        'skipped_reasons': dict(Counter(r['reason'] for r in (train_skipped + val_skipped))),
    }
    (out_root / 'input_support_summary.json').write_text(json.dumps(support_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    process_samples(all_samples, model, out_root, args.conf, args.batch_size)


if __name__ == '__main__':
    main()
