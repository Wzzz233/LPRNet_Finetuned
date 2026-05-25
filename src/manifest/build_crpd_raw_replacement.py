#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from lpr_pipeline_policy import apply_board_params
from load_data import CHARS_DICT, parse_ccpd_bbox_from_name, parse_ccpd_quad_from_name

FIELDS = [
    'img_path','img_rel_path','dataset_name','split','text','plate_len','family','sub_type','source','is_real','need_tilt_aug',
    'preprocess_group','has_bbox','has_quad','can_parse_ccpd_geom','can_perspective','bbox_source','quad_source',
    'ocr_channel_order','ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc','ocr_min_occ_ratio','ocr_quad_pad_ratio'
]

SUPPORTED_CHARS = set(CHARS_DICT.keys())
SPECIAL_HINT_CHARS = set('警学挂使领港澳')
OLD_CRPD_DATASETS = {'crpd_ccpd_strict_yolo_v2', 'CRPD_CCPD_STRICT_YOLO_v2'}


def text_supported(text: str) -> bool:
    return all(ch in SUPPORTED_CHARS for ch in text)


def is_green_small(text: str) -> bool:
    return len(text) == 8 and text[2] in {'D', 'F'}


def is_green_large(text: str) -> bool:
    return len(text) == 8 and text[-1] in {'D', 'F'}


def classify_record(text: str, cls: str) -> Tuple[str, str, str]:
    text = text.strip().upper()
    if is_green_small(text):
        return 'green8', 'green_small', 'green'
    if is_green_large(text):
        return 'green8', 'green_large', 'green'
    if cls == '0' and len(text) == 7 and not any(ch in SPECIAL_HINT_CHARS for ch in text):
        return 'normal7', 'blue', 'blue'
    if cls == '1':
        return 'special', 'yellow_single', 'yellow'
    if cls == '2':
        return 'special', 'yellow_double', 'yellow'
    return 'special', 'special', 'special'


def safe_text_token(text: str) -> str:
    return ''.join(ch if ch.isalnum() or ch >= '\u4e00' else '_' for ch in text)


def make_ccpd_like_name(src_stem: str, obj_idx: int, text: str, quad_points: List[Tuple[int, int]]) -> str:
    xs = [p[0] for p in quad_points]
    ys = [p[1] for p in quad_points]
    bbox = f'{min(xs)}&{min(ys)}_{max(xs)}&{max(ys)}'
    quad = '_'.join(f'{x}&{y}' for x, y in quad_points)
    return f'crpd-raw-{bbox}-{quad}-0-0-{safe_text_token(text)}-{src_stem}_obj{obj_idx}.jpg'


def ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.path.realpath(dst) == str(src.resolve()):
            return
        dst.unlink()
    os.symlink(src.resolve(), dst)


def iter_crpd_raw_records(crpd_root: Path):
    for subset_dir in sorted(crpd_root.glob('CRPD_*')):
        if not subset_dir.is_dir():
            continue
        subset_name = subset_dir.name
        for split_dir in sorted(subset_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            labels_dir = split_dir / 'labels'
            images_dir = split_dir / 'images'
            if not labels_dir.exists() or not images_dir.exists():
                continue
            for label_path in sorted(labels_dir.glob('*.txt')):
                img_path = images_dir / f'{label_path.stem}.jpg'
                yield subset_name, split, img_path, label_path


def build_rows(crpd_root: Path, output_root: Path, repo_root: Path):
    manifest_rows: List[Dict[str, str]] = []
    all_records: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    for subset_name, split, img_path, label_path in iter_crpd_raw_records(crpd_root):
        if not img_path.exists():
            skipped.append({
                'subset_name': subset_name,
                'split': split,
                'label_path': str(label_path),
                'image_path': str(img_path),
                'reason': 'missing_image',
            })
            continue
        lines = [x for x in label_path.read_text(encoding='utf-8').splitlines() if x.strip()]
        for obj_idx, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) < 10:
                skipped.append({
                    'subset_name': subset_name,
                    'split': split,
                    'label_path': str(label_path),
                    'image_path': str(img_path),
                    'reason': 'bad_label_line',
                    'raw_line': line,
                })
                continue
            try:
                quad = [(int(round(float(parts[i]))), int(round(float(parts[i + 1])))) for i in range(0, 8, 2)]
            except ValueError:
                skipped.append({
                    'subset_name': subset_name,
                    'split': split,
                    'label_path': str(label_path),
                    'image_path': str(img_path),
                    'reason': 'bad_quad_values',
                    'raw_line': line,
                })
                continue
            cls = parts[8].strip()
            text = parts[9].strip().upper()
            family, sub_type, plate_group = classify_record(text, cls)
            new_name = make_ccpd_like_name(img_path.stem, obj_idx, text, quad)
            out_img = output_root / sub_type / split / new_name
            ensure_symlink(img_path, out_img)

            supported = text_supported(text)
            unsupported_chars = ''.join(sorted({ch for ch in text if ch not in SUPPORTED_CHARS}))
            out_img_abs = out_img.absolute()
            record = {
                'subset_name': subset_name,
                'split': split,
                'src_image': str(img_path.resolve()),
                'src_label': str(label_path.resolve()),
                'obj_index': obj_idx,
                'raw_cls': cls,
                'text': text,
                'plate_len': len(text),
                'family': family,
                'sub_type': sub_type,
                'plate_group': plate_group,
                'supported': 1 if supported else 0,
                'unsupported_chars': unsupported_chars,
                'output_image': str(out_img_abs),
            }
            all_records.append(record)

            if not supported:
                skipped.append({
                    'subset_name': subset_name,
                    'split': split,
                    'label_path': str(label_path),
                    'image_path': str(img_path),
                    'raw_cls': cls,
                    'text': text,
                    'family': family,
                    'sub_type': sub_type,
                    'reason': 'unsupported_chars',
                    'unsupported_chars': unsupported_chars,
                    'output_image': str(out_img_abs),
                })
                continue

            row = {
                'img_path': str(out_img_abs),
                'img_rel_path': str(out_img_abs.relative_to(repo_root)).replace('\\', '/'),
                'dataset_name': 'crpd_all_raw',
                'split': split,
                'text': text,
                'plate_len': len(text),
                'family': family,
                'sub_type': sub_type,
                'source': 'real',
                'is_real': 1,
                'need_tilt_aug': 1 if family in {'normal7', 'green8'} else 0,
                'preprocess_group': 'ccpd_board',
                'has_bbox': 1,
                'has_quad': 1,
                'can_parse_ccpd_geom': 1,
                'can_perspective': 1,
                'bbox_source': 'raw_label_txt',
                'quad_source': 'raw_label_txt',
            }
            manifest_rows.append(apply_board_params(row))
    return manifest_rows, all_records, skipped


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(all_records: List[Dict], manifest_rows: List[Dict], skipped: List[Dict]) -> Dict:
    return {
        'all_record_count': len(all_records),
        'supported_manifest_count': len(manifest_rows),
        'skipped_count': len(skipped),
        'all_by_group': dict(sorted(Counter(r['plate_group'] for r in all_records).items())),
        'all_by_family': dict(sorted(Counter(r['family'] for r in all_records).items())),
        'all_by_sub_type': dict(sorted(Counter(r['sub_type'] for r in all_records).items())),
        'all_by_raw_cls': dict(sorted(Counter(r['raw_cls'] for r in all_records).items())),
        'all_by_split': dict(sorted(Counter(r['split'] for r in all_records).items())),
        'supported_by_group': dict(sorted(Counter(r['sub_type'] if r['family'] == 'special' else r['family'] for r in manifest_rows).items())),
        'supported_manifest_families': dict(sorted(Counter(r['family'] for r in manifest_rows).items())),
        'supported_manifest_sub_types': dict(sorted(Counter(r['sub_type'] for r in manifest_rows).items())),
        'supported_manifest_splits': dict(sorted(Counter(r['split'] for r in manifest_rows).items())),
        'skipped_reasons': dict(sorted(Counter(r['reason'] for r in skipped).items())),
        'skipped_unsupported_chars': dict(sorted(Counter(r.get('unsupported_chars', '') for r in skipped if r.get('reason') == 'unsupported_chars').items())),
    }


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def replace_old_crpd_rows(base_manifest: Path, out_manifest: Path, crpd_rows: List[Dict]):
    base_rows = load_manifest_rows(base_manifest)
    kept = [r for r in base_rows if (r.get('dataset_name') or '') not in OLD_CRPD_DATASETS]
    merged = kept + crpd_rows
    write_csv(out_manifest, merged, FIELDS)
    return {
        'base_manifest': str(base_manifest),
        'base_count': len(base_rows),
        'removed_old_crpd_count': len(base_rows) - len(kept),
        'added_new_crpd_count': len(crpd_rows),
        'final_count': len(merged),
    }


def quick_verify(manifest_rows: List[Dict]):
    problems = []
    for row in manifest_rows[:20]:
        img_path = row['img_path']
        img_rel_path = row['img_rel_path']
        if not Path(img_path).exists():
            problems.append({'img_path': img_path, 'reason': 'missing_img'})
            continue
        if parse_ccpd_bbox_from_name(img_rel_path) is None:
            problems.append({'img_path': img_path, 'reason': 'bad_bbox_parse'})
        if parse_ccpd_quad_from_name(img_rel_path) is None:
            problems.append({'img_path': img_path, 'reason': 'bad_quad_parse'})
    return problems


def main():
    ap = argparse.ArgumentParser(description='Build CRPD raw replacement rows grouped by blue/green/yellow/special and replace old processed CRPD rows.')
    ap.add_argument('--repo-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--crpd-root', default='/home/wzzz/LPRNet/CRPD_all')
    ap.add_argument('--output-root', default='/home/wzzz/LPRNet/CRPD_raw_ccpd_board_v1')
    ap.add_argument('--out-manifest', default='/home/wzzz/LPRNet/manifests/crpd_all_raw_board_v1_supported.csv')
    ap.add_argument('--out-all-records', default='/home/wzzz/LPRNet/reports/crpd_all_raw_board_v1_all_records.csv')
    ap.add_argument('--out-skipped', default='/home/wzzz/LPRNet/reports/crpd_all_raw_board_v1_skipped.csv')
    ap.add_argument('--out-summary', default='/home/wzzz/LPRNet/reports/crpd_all_raw_board_v1_summary.json')
    ap.add_argument('--base-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_v4_board_aligned_real_only.csv')
    ap.add_argument('--out-replaced-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_v4_board_aligned_real_only_crpd_raw.csv')
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    crpd_root = Path(args.crpd_root).resolve()
    output_root = Path(args.output_root).resolve()

    manifest_rows, all_records, skipped = build_rows(crpd_root, output_root, repo_root)
    summary = summarize(all_records, manifest_rows, skipped)
    summary['quick_verify'] = quick_verify(manifest_rows)

    write_csv(Path(args.out_manifest), manifest_rows, FIELDS)
    if all_records:
        write_csv(Path(args.out_all_records), all_records, list(all_records[0].keys()))
    else:
        write_csv(Path(args.out_all_records), [], ['subset_name','split','src_image','src_label','obj_index','raw_cls','text','plate_len','family','sub_type','plate_group','supported','unsupported_chars','output_image'])
    if skipped:
        write_csv(Path(args.out_skipped), skipped, sorted({k for row in skipped for k in row.keys()}))
    else:
        write_csv(Path(args.out_skipped), [], ['subset_name','split','label_path','image_path','reason'])

    replace_stats = replace_old_crpd_rows(Path(args.base_manifest), Path(args.out_replaced_manifest), manifest_rows)
    summary['replace_stats'] = replace_stats

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'out_manifest': args.out_manifest,
        'out_replaced_manifest': args.out_replaced_manifest,
        'out_summary': args.out_summary,
        'supported_manifest_count': len(manifest_rows),
        'skipped_count': len(skipped),
        'quick_verify_problem_count': len(summary['quick_verify']),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
