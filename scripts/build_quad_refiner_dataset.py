#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quad_refiner.dataset import build_ccpd_record, build_crpd_records, load_jsonl_records, write_jsonl_records


def load_coarse_map(path):
    if not path:
        return {}
    records = load_jsonl_records(path)
    out = {}
    for row in records:
        sample_id = row.get('sample_id')
        if sample_id and row.get('coarse_quad'):
            out[sample_id] = row['coarse_quad']
    return out


def load_family_whitelist(path):
    if not path:
        return set()
    allowed = set()
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                allowed.add(token)
    return allowed


def read_split_file(root, split_file):
    rows = []
    with Path(split_file).open('r', encoding='utf-8') as f:
        for line in f:
            rel = line.strip().split()[0]
            if rel:
                rows.append(root / rel)
    return rows


def maybe_override(rec, coarse_map, strict_coarse=False):
    coarse = coarse_map.get(rec['sample_id'])
    if coarse is not None:
        rec['coarse_quad'] = coarse
        return rec
    if strict_coarse and coarse_map:
        return None
    return rec


def maybe_filter_family(rec, allowed_families):
    if not allowed_families:
        return rec
    if rec.get('family') not in allowed_families:
        return None
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description='Build JSONL dataset for OBB quad refiner.')
    ap.add_argument('--output-jsonl', required=True)
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--ccpd2019-root', default='/home/wzzz/LPRNet/datasets/CCPD2019')
    ap.add_argument('--ccpd2019-split-file', default='')
    ap.add_argument('--ccpd2020-root', default='/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green')
    ap.add_argument('--crpd-root', default='/home/wzzz/LPRNet/datasets/CRPD_all')
    ap.add_argument('--skip-crpd', action='store_true')
    ap.add_argument('--coarse-jsonl', default='')
    ap.add_argument('--strict-coarse', action='store_true')
    ap.add_argument('--family-whitelist', default='')
    ap.add_argument('--append-jsonl', default='')
    ap.add_argument('--limit-ccpd2019', type=int, default=0)
    ap.add_argument('--limit-ccpd2020', type=int, default=0)
    ap.add_argument('--limit-crpd', type=int, default=0)
    args = ap.parse_args(argv)

    records = []
    coarse_map = load_coarse_map(args.coarse_jsonl)
    allowed_families = load_family_whitelist(args.family_whitelist)

    ccpd2019_root = Path(args.ccpd2019_root)
    split_file = Path(args.ccpd2019_split_file) if args.ccpd2019_split_file else ccpd2019_root / f'{args.split}.txt'
    if split_file.exists():
        paths = read_split_file(ccpd2019_root, split_file)
        if args.limit_ccpd2019 > 0:
            paths = paths[:args.limit_ccpd2019]
        for path in paths:
            if path.exists():
                rec = build_ccpd_record(path, split=args.split, source_name='ccpd2019', family='normal7', sub_type='blue')
                rec = maybe_filter_family(rec, allowed_families)
                if rec is None:
                    continue
                rec = maybe_override(rec, coarse_map, strict_coarse=args.strict_coarse)
                if rec is not None:
                    records.append(rec)

    ccpd2020_dir = Path(args.ccpd2020_root) / args.split
    if ccpd2020_dir.exists():
        paths = sorted(ccpd2020_dir.rglob('*.jpg'))
        if args.limit_ccpd2020 > 0:
            paths = paths[:args.limit_ccpd2020]
        for path in paths:
            rec = build_ccpd_record(path, split=args.split, source_name='ccpd2020_green', family='green8', sub_type='green')
            rec = maybe_filter_family(rec, allowed_families)
            if rec is None:
                continue
            rec = maybe_override(rec, coarse_map, strict_coarse=args.strict_coarse)
            if rec is not None:
                records.append(rec)

    if not args.skip_crpd:
        crpd_root = Path(args.crpd_root)
        crpd_total = 0
        for subset in ['CRPD_single', 'CRPD_double', 'CRPD_multi']:
            img_dir = crpd_root / subset / args.split / 'images'
            label_dir = crpd_root / subset / args.split / 'labels'
            if not img_dir.exists() or not label_dir.exists():
                continue
            for img_path in sorted(img_dir.glob('*.jpg')):
                label_path = label_dir / f'{img_path.stem}.txt'
                if not label_path.exists():
                    continue
                chunk = build_crpd_records(img_path, label_path, split=args.split, source_name=f'crpd_raw/{subset}')
                for rec in chunk:
                    rec = maybe_filter_family(rec, allowed_families)
                    if rec is None:
                        continue
                    rec = maybe_override(rec, coarse_map, strict_coarse=args.strict_coarse)
                    if rec is not None:
                        records.append(rec)
                crpd_total += len(chunk)
                if args.limit_crpd > 0 and crpd_total >= args.limit_crpd:
                    break
            if args.limit_crpd > 0 and crpd_total >= args.limit_crpd:
                break

    if args.append_jsonl:
        records.extend(load_jsonl_records(args.append_jsonl))

    write_jsonl_records(records, args.output_jsonl)
    summary = {'count': len(records), 'output_jsonl': args.output_jsonl, 'split': args.split}
    Path(args.output_jsonl).with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
