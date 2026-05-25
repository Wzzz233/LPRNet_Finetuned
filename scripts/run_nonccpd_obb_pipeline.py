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
from typing import Iterable, List, Optional

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
    split: str
    dataset_name: str
    family: str
    sub_type: str
    plate_type: str
    source_name: str
    is_generated: bool


@dataclass
class DatasetSpec:
    name: str
    kind: str
    is_generated: bool


def normalize_text(text: str) -> str:
    return (text or '').strip().upper()


def safe_token(s: str) -> str:
    s = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff._-]+', '_', str(s))
    return s.strip('_') or 'x'


def text_supported(text: str) -> bool:
    return all(ch in SUPPORTED_CHARS for ch in text)


def family_from_text(text: str):
    t = normalize_text(text)
    if len(t) == 8:
        return 'green8', 'green'
    if len(t) == 7:
        return 'normal7', 'blue'
    return 'special', 'special'


def box_from_quad(quad):
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    return int(max(0, min(xs))), int(max(0, min(ys))), int(max(0, max(xs))), int(max(0, max(ys)))


def quad_to_ccpd_text(quad) -> str:
    return '_'.join(f'{int(round(x))}&{int(round(y))}' for x, y in quad)


def make_pseudo_ccpd_name(sample: Sample, quad, suffix='jpg') -> str:
    x1, y1, x2, y2 = box_from_quad(quad)
    bbox = f'{x1}&{y1}_{x2}&{y2}'
    quad_text = quad_to_ccpd_text(quad)
    text_tag = safe_token(sample.text)
    plate_type_tag = safe_token(sample.plate_type) if sample.plate_type else '0'
    origin_tag = safe_token(Path(sample.rel_path).with_suffix('').as_posix())
    return f'autoobb-0-{bbox}-{quad_text}-{text_tag}-{plate_type_tag}-0-{origin_tag}.{suffix}'


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


def parse_targeted_txt(root: Path, split: str) -> List[Sample]:
    txt_path = root / 'targeted_green_missing_18' / f'{split}.txt'
    samples = []
    if not txt_path.exists():
        return samples
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
            text = normalize_text(text)
            family, sub_type = family_from_text(text)
            samples.append(Sample(
                image_path=img_path,
                rel_path=rel,
                text=text,
                split=split,
                dataset_name='targeted_green_missing_18',
                family=family,
                sub_type=sub_type,
                plate_type='',
                source_name='targeted_green_missing_18',
                is_generated=True,
            ))
    return samples


def parse_git_plate_dir(root: Path) -> List[Sample]:
    dir_path = root / 'git_plate' / 'val' / 'val_verify'
    samples = []
    if not dir_path.exists():
        return samples
    for img_path in sorted(dir_path.glob('*.jpg')):
        text = normalize_text(img_path.stem.split('_')[0])
        family, sub_type = family_from_text(text)
        samples.append(Sample(
            image_path=img_path,
            rel_path=str(img_path.relative_to(root)).replace('\\', '/'),
            text=text,
            split='val',
            dataset_name='git_plate',
            family=family,
            sub_type=sub_type,
            plate_type='',
            source_name='git_plate_val_verify',
            is_generated=True,
        ))
    return samples


def parse_cblprd_txt(root: Path, split: str) -> List[Sample]:
    txt_path = root / 'CBLPRD-330k_v1' / f'{split}.txt'
    samples = []
    if not txt_path.exists():
        return samples
    with txt_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                continue
            rel_path, text, plate_type = parts
            text = normalize_text(text)
            family, sub_type = PLATE_TYPE_META.get(plate_type, ('special', safe_token(plate_type)))
            samples.append(Sample(
                image_path=root / 'CBLPRD-330k_v1' / rel_path,
                rel_path=rel_path.replace('\\', '/'),
                text=text,
                split=split,
                dataset_name='CBLPRD',
                family=family,
                sub_type=sub_type,
                plate_type=plate_type,
                source_name='CBLPRD-330k_v1',
                is_generated=False,
            ))
    return samples


def collect_samples(root: Path, include_names: Optional[set[str]] = None, skip_generated: bool = True) -> tuple[list[Sample], list[dict], list[dict]]:
    dataset_specs = [
        DatasetSpec('targeted_green_missing_18', 'targeted', True),
        DatasetSpec('git_plate', 'git_plate', True),
        DatasetSpec('CBLPRD', 'cblprd', False),
    ]
    samples: list[Sample] = []
    skipped: list[dict] = []
    catalog: list[dict] = []

    for spec in dataset_specs:
        if include_names and spec.name not in include_names:
            continue
        if skip_generated and spec.is_generated:
            catalog.append({'dataset_name': spec.name, 'status': 'skipped_generated'})
            continue
        before = len(samples)
        if spec.kind == 'targeted':
            samples.extend(parse_targeted_txt(root, 'train'))
            samples.extend(parse_targeted_txt(root, 'val'))
        elif spec.kind == 'git_plate':
            samples.extend(parse_git_plate_dir(root))
        elif spec.kind == 'cblprd':
            samples.extend(parse_cblprd_txt(root, 'train'))
            samples.extend(parse_cblprd_txt(root, 'val'))
        catalog.append({'dataset_name': spec.name, 'status': 'included', 'sample_count': len(samples) - before})

    kept = []
    for s in samples:
        if not s.image_path.exists():
            skipped.append({'dataset_name': s.dataset_name, 'split': s.split, 'image_path': str(s.image_path), 'text': s.text, 'reason': 'missing_image'})
            continue
        if not text_supported(s.text):
            skipped.append({'dataset_name': s.dataset_name, 'split': s.split, 'image_path': str(s.image_path), 'text': s.text, 'reason': 'unsupported_chars'})
            continue
        kept.append(s)
    return kept, skipped, catalog


def process_samples(samples: List[Sample], model: YOLO, out_root: Path, conf_thres: float, batch_size: int, limit: int = 0):
    if limit > 0:
        samples = samples[:limit]
    success_rows = []
    fail_rows = []
    groups = defaultdict(lambda: Counter(total=0, success=0, fail=0))

    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        paths = [str(s.image_path) for s in chunk]
        results = model.predict(source=paths, conf=conf_thres, verbose=False, batch=batch_size)
        for sample, result in zip(chunk, results):
            group_key = f'{sample.dataset_name}:{sample.split}:{sample.family}'
            groups[group_key]['total'] += 1
            det = pick_best_obb(result)
            if det is None or det['conf'] < conf_thres:
                fail_dir = out_root / 'failed' / sample.family / sample.dataset_name / sample.split
                if sample.plate_type:
                    fail_dir = fail_dir / safe_token(sample.plate_type)
                ensure_dir(fail_dir)
                dst = fail_dir / sample.image_path.name
                if sample.image_path.exists() and not dst.exists():
                    shutil.copy2(sample.image_path, dst)
                fail_rows.append({
                    'image_path': str(sample.image_path),
                    'rel_path': sample.rel_path,
                    'dataset_name': sample.dataset_name,
                    'source_name': sample.source_name,
                    'split': sample.split,
                    'family': sample.family,
                    'sub_type': sample.sub_type,
                    'plate_type': sample.plate_type,
                    'text': sample.text,
                    'status': 'failed',
                    'reason': 'no_detection',
                    'output_image': str(dst),
                    'is_generated': int(sample.is_generated),
                })
                groups[group_key]['fail'] += 1
                continue

            suffix = sample.image_path.suffix.lstrip('.') or 'jpg'
            name = make_pseudo_ccpd_name(sample, det['quad'], suffix=suffix)
            success_dir = out_root / 'success' / sample.family / sample.dataset_name / sample.split
            if sample.plate_type:
                success_dir = success_dir / safe_token(sample.plate_type)
            ensure_dir(success_dir)
            out_img = success_dir / name
            if not out_img.exists():
                shutil.copy2(sample.image_path, out_img)
            success_rows.append({
                'image_path': str(sample.image_path),
                'rel_path': sample.rel_path,
                'dataset_name': sample.dataset_name,
                'source_name': sample.source_name,
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
                'bbox': box_from_quad(det['quad']),
                'quad': det['quad'],
                'is_generated': int(sample.is_generated),
            })
            groups[group_key]['success'] += 1

    ensure_dir(out_root)
    with (out_root / 'success_records.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['image_path', 'rel_path', 'dataset_name', 'source_name', 'split', 'family', 'sub_type', 'plate_type', 'text', 'status', 'reason', 'det_conf', 'det_count', 'output_image', 'bbox', 'quad', 'is_generated']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(success_rows)
    with (out_root / 'failed_records.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['image_path', 'rel_path', 'dataset_name', 'source_name', 'split', 'family', 'sub_type', 'plate_type', 'text', 'status', 'reason', 'output_image', 'is_generated']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(fail_rows)
    report = {
        'out_root': str(out_root),
        'conf_thres': conf_thres,
        'total_samples': len(samples),
        'success_count': len(success_rows),
        'fail_count': len(fail_rows),
        'groups': {k: dict(v) for k, v in sorted(groups.items())},
        'success_records_csv': str(out_root / 'success_records.csv'),
        'failed_records_csv': str(out_root / 'failed_records.csv'),
    }
    (out_root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description='Formal non-CCPD OBB pipeline with dataset registry, generated-data skip, and family/dataset/split outputs.')
    ap.add_argument('--root', default='/home/wzzz/LPRNet')
    ap.add_argument('--weights', default='/home/wzzz/LPRNet/external_detectors/obb_best.pt')
    ap.add_argument('--out-root', default='/home/wzzz/LPRNet/nonccpd_obb_pipeline_v2')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--datasets', default='all', help='Comma list like CBLPRD,git_plate,targeted_green_missing_18 or all')
    ap.add_argument('--skip-generated', default='true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    skip_generated = str(args.skip_generated).lower() not in {'0', 'false', 'no', 'n'}
    include_names = None if args.datasets == 'all' else {x.strip() for x in args.datasets.split(',') if x.strip()}
    root = Path(args.root)
    out_root = Path(args.out_root)

    samples, skipped, catalog = collect_samples(root, include_names=include_names, skip_generated=skip_generated)
    ensure_dir(out_root)
    (out_root / 'dataset_catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (out_root / 'skipped_inputs.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['dataset_name', 'split', 'image_path', 'text', 'reason']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(skipped)

    pre_report = {
        'dataset_catalog': catalog,
        'skip_generated': skip_generated,
        'selected_sample_count': len(samples),
        'skipped_input_count': len(skipped),
        'selected_by_dataset': dict(sorted(Counter(s.dataset_name for s in samples).items())),
        'selected_by_family': dict(sorted(Counter(s.family for s in samples).items())),
        'skipped_reasons': dict(sorted(Counter(r['reason'] for r in skipped).items())),
    }
    (out_root / 'input_summary.json').write_text(json.dumps(pre_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    if args.dry_run:
        print(json.dumps(pre_report, ensure_ascii=False, indent=2))
        return

    model = YOLO(args.weights)
    report = process_samples(samples, model, out_root, args.conf, args.batch_size, limit=args.limit)
    merged = {'input_summary': pre_report, 'run_summary': report}
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
