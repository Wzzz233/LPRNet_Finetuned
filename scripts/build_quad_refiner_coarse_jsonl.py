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

from quad_refiner.coarse_export import (
    export_crpd_pairs_to_coarse_jsonl,
    export_paths_to_coarse_jsonl,
    iter_ccpd2019_paths,
    iter_ccpd2020_green_paths,
    iter_crpd_pairs,
)


def append_jsonl(dst_path: Path, src_path: Path):
    if not src_path.exists():
        return 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    text = src_path.read_text(encoding='utf-8')
    mode = 'a' if dst_path.exists() else 'w'
    with dst_path.open(mode, encoding='utf-8') as f:
        f.write(text)
    return len([line for line in text.splitlines() if line.strip()])


def main(argv=None):
    ap = argparse.ArgumentParser(description='Batch export YOLO OBB coarse quads to quad-refiner JSONL.')
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--output-jsonl', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--ccpd2019-root', default='/home/wzzz/LPRNet/datasets/CCPD2019')
    ap.add_argument('--ccpd2019-split-file', default='')
    ap.add_argument('--ccpd2019-weights', default='/home/wzzz/LPRNet/models/detectors/yolov8n_obb_trained_70epoch/best.pt')
    ap.add_argument('--ccpd2020-root', default='/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green')
    ap.add_argument('--ccpd2020-weights', default='/home/wzzz/LPRNet/datasets/downloaded_green_clone_success/best.pt')
    ap.add_argument('--crpd-root', default='/home/wzzz/LPRNet/datasets/CRPD_all')
    ap.add_argument('--crpd-weights', default='/home/wzzz/LPRNet/models/detectors/yolov8n_obb_trained_70epoch/best.pt')
    ap.add_argument('--crpd-min-iou', type=float, default=0.05)
    ap.add_argument('--shared-weights', default='')
    ap.add_argument('--skip-ccpd2019', action='store_true')
    ap.add_argument('--skip-ccpd2020', action='store_true')
    ap.add_argument('--skip-crpd', action='store_true')
    ap.add_argument('--limit-ccpd2019', type=int, default=0)
    ap.add_argument('--limit-ccpd2020', type=int, default=0)
    ap.add_argument('--limit-crpd', type=int, default=0)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    out_path = Path(args.output_jsonl)
    if out_path.exists():
        out_path.unlink()
    summaries = []
    ccpd2019_weights = args.shared_weights or args.ccpd2019_weights
    ccpd2020_weights = args.shared_weights or args.ccpd2020_weights
    crpd_weights = args.shared_weights or args.crpd_weights

    if not args.skip_ccpd2019:
        paths = iter_ccpd2019_paths(args.ccpd2019_root, args.split, args.ccpd2019_split_file)
        if args.limit_ccpd2019 > 0:
            paths = paths[:args.limit_ccpd2019]
        tmp = out_path.with_name(out_path.stem + '.ccpd2019.tmp.jsonl')
        model = YOLO(ccpd2019_weights)
        summaries.append(export_paths_to_coarse_jsonl(paths, model, 'ccpd2019', tmp, conf_thres=args.conf, split=args.split, imgsz=args.imgsz))
        append_jsonl(out_path, tmp)
        if tmp.exists():
            tmp.unlink()

    if not args.skip_ccpd2020:
        paths = iter_ccpd2020_green_paths(args.ccpd2020_root, args.split)
        if args.limit_ccpd2020 > 0:
            paths = paths[:args.limit_ccpd2020]
        tmp = out_path.with_name(out_path.stem + '.ccpd2020.tmp.jsonl')
        model = YOLO(ccpd2020_weights)
        summaries.append(export_paths_to_coarse_jsonl(paths, model, 'ccpd2020_green', tmp, conf_thres=args.conf, split=args.split, imgsz=args.imgsz))
        append_jsonl(out_path, tmp)
        if tmp.exists():
            tmp.unlink()

    if not args.skip_crpd:
        pairs = iter_crpd_pairs(args.crpd_root, args.split)
        if args.limit_crpd > 0:
            pairs = pairs[:args.limit_crpd]
        tmp = out_path.with_name(out_path.stem + '.crpd.tmp.jsonl')
        model = YOLO(crpd_weights)
        summaries.append(export_crpd_pairs_to_coarse_jsonl(pairs, model, tmp, conf_thres=args.conf, split=args.split, imgsz=args.imgsz, min_iou=args.crpd_min_iou))
        append_jsonl(out_path, tmp)
        if tmp.exists():
            tmp.unlink()

    summary = {
        'output_jsonl': str(out_path),
        'split': args.split,
        'parts': summaries,
        'total_count': sum(int(x.get('count', 0)) for x in summaries),
        'total_miss_count': sum(int(x.get('miss_count', 0)) for x in summaries),
    }
    out_path.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
