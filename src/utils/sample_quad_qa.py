#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
from pathlib import Path

import cv2
import numpy as np


def parse_ccpd_quad_from_name(image_name):
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) < 4:
        return None
    points_text = parts[3]
    points = []
    try:
        for item in points_text.split('_'):
            if '&' not in item:
                return None
            xs, ys = item.split('&', 1)
            points.append((float(xs), float(ys)))
    except ValueError:
        return None
    if len(points) != 4:
        return None
    return np.asarray(points, dtype=np.float32)


def order_quad_points(pts):
    quad = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = quad.sum(axis=1)
    diffs = np.diff(quad, axis=1).reshape(-1)
    ordered[0] = quad[np.argmin(sums)]
    ordered[2] = quad[np.argmax(sums)]
    ordered[1] = quad[np.argmin(diffs)]
    ordered[3] = quad[np.argmax(diffs)]
    return ordered


def draw_quad(img, quad, label):
    quad = order_quad_points(quad).astype(int)
    colors = [(0,0,255),(0,255,255),(0,255,0),(255,0,0)]
    out = img.copy()
    for i in range(4):
        p1 = tuple(quad[i])
        p2 = tuple(quad[(i+1)%4])
        cv2.line(out, p1, p2, colors[i], 2)
        cv2.circle(out, p1, 4, colors[i], -1)
        cv2.putText(out, str(i), (p1[0]+3, p1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1, cv2.LINE_AA)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return out


def sample_rows_from_csv(path, dataset_name, n, seed=20260327):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['dataset_name'] == dataset_name:
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def sample_rows_from_manifest(path, dataset_name, split, n, seed=20260327):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['dataset_name'] == dataset_name and r['split'] == split:
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main():
    out_root = Path('/home/wzzz/LPRNet/qa_quad_samples')
    out_root.mkdir(parents=True, exist_ok=True)

    items = []

    # git_plate pseudo geom from success records
    git_rows = sample_rows_from_csv('/home/wzzz/LPRNet/nonccpd_obb_autolabel_v1/success_records.csv', 'git_plate', 6)
    for i, r in enumerate(git_rows, 1):
        img_path = Path(r['output_image'])
        quad = parse_ccpd_quad_from_name(img_path.name)
        items.append(('git_plate', i, img_path, r['text'], quad))

    # CRPD from manifest v3, real ccpd-style crops
    crpd_rows = sample_rows_from_manifest('/home/wzzz/LPRNet/manifests/crpd_all_raw_board_v1_supported.csv', 'real', 'train', 6)
    for i, r in enumerate(crpd_rows, 1):
        img_path = Path(r['img_path'])
        quad = parse_ccpd_quad_from_name(img_path.name)
        items.append(('crpd', i, img_path, r['text'], quad))

    summary_lines = []
    for group, idx, img_path, text, quad in items:
        img = cv2.imread(str(img_path))
        if img is None:
            summary_lines.append(f'{group}\t{idx}\tMISSING\t{img_path}\t{text}')
            continue
        vis = draw_quad(img, quad, f'{group} #{idx} {text}') if quad is not None else img
        out_path = out_root / f'{group}_{idx:02d}_{text}.jpg'
        cv2.imwrite(str(out_path), vis)
        summary_lines.append(f'{group}\t{idx}\tOK\t{out_path}\t{img_path}\t{text}')

    (out_root / 'index.tsv').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(str(out_root))


if __name__ == '__main__':
    main()
