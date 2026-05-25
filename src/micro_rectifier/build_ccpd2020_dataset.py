#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_ccpd_coarse_map(path: Path):
    out = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get('sample_id', '')
            if not sid or not row.get('coarse_quad'):
                continue
            stem = sid.split(':', 1)[-1]
            out[stem] = row['coarse_quad']
    return out


def parse_ccpd_gt_quad(image_name: str):
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) < 4:
        return None
    pts = parts[3].split('_')
    if len(pts) != 4:
        return None
    quad = []
    for p in pts:
        x, y = p.split('&')
        quad.append([float(x), float(y)])
    return np.asarray(quad, dtype=np.float32)


def order_quad(quad):
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).reshape(-1)
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(diff)]
    bl = quad[np.argmax(diff)]
    return np.asarray([tl, tr, br, bl], dtype=np.float32)


def warp_with_quad(image, quad, out_w, out_h):
    quad = order_quad(quad)
    dst = np.asarray([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


def estimate_affine_params_from_pair(src_patch, tgt_patch):
    src_gray = cv2.cvtColor(src_patch, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(tgt_patch, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _cc, warp = cv2.findTransformECC(
            templateImage=tgt_gray,
            inputImage=src_gray,
            warpMatrix=warp,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=3,
        )
    except cv2.error:
        pass
    dx = float(warp[0, 2])
    dy = float(warp[1, 2])
    sx = float(warp[0, 0])
    sy = float(warp[1, 1])
    shx = float(warp[0, 1])
    return dx, dy, sx, sy, shx


def build_records(ccpd_root: Path, split: str, coarse_map, out_csv: Path, input_w: int, input_h: int):
    split_dir = ccpd_root / split
    rows = []
    for img_path in sorted(split_dir.glob('*.jpg')):
        stem = img_path.stem
        if stem not in coarse_map:
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        gt_quad = parse_ccpd_gt_quad(img_path.name)
        if gt_quad is None:
            continue
        coarse_quad = np.asarray(coarse_map[stem], dtype=np.float32)
        coarse_patch = warp_with_quad(image, coarse_quad, input_w, input_h)
        gt_patch = warp_with_quad(image, gt_quad, input_w, input_h)
        dx, dy, sx, sy, shx = estimate_affine_params_from_pair(coarse_patch, gt_patch)
        src_path = out_csv.parent / split / 'input' / f'{stem}.png'
        tgt_path = out_csv.parent / split / 'target' / f'{stem}.png'
        src_path.parent.mkdir(parents=True, exist_ok=True)
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(src_path), coarse_patch)
        cv2.imwrite(str(tgt_path), gt_patch)
        rows.append({
            'sample_id': stem,
            'input_path': str(src_path),
            'target_path': str(tgt_path),
            'text': '',
            'split': split,
            'source_name': 'ccpd2020_green_micro_rectifier',
            'dx': dx,
            'dy': dy,
            'sx': sx,
            'sy': sy,
            'shx': shx,
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['sample_id','input_path','target_path','text','split','source_name','dx','dy','sx','sy','shx'])
        writer.writeheader()
        writer.writerows(rows)
    return {'count': len(rows), 'out_csv': str(out_csv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ccpd2020-root', required=True)
    ap.add_argument('--coarse-jsonl', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--input-width', type=int, default=160)
    ap.add_argument('--input-height', type=int, default=48)
    args = ap.parse_args()

    coarse_map = read_ccpd_coarse_map(Path(args.coarse_jsonl))
    out_root = Path(args.output_root)
    summary = {}
    for split in ['train', 'val']:
        summary[split] = build_records(Path(args.ccpd2020_root), split, coarse_map, out_root / f'{split}.csv', args.input_width, args.input_height)
    (out_root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
