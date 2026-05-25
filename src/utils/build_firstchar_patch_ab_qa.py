#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT / 'src', ROOT / 'src' / 'training']:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from load_data import parse_ccpd_quad_from_name, prepare_board_ocr_input_from_quad_bgr888  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description='Build firstchar patch QA for A/B methods across datasets/families.')
    ap.add_argument('--main-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_v4_board_aligned_real_only_crpd_raw.csv')
    ap.add_argument('--crpd-manifest', default='/home/wzzz/LPRNet/manifests/crpd_all_raw_board_v1_supported.csv')
    ap.add_argument('--cblprd-cv-manifest', default='/home/wzzz/LPRNet/manifests/cblprd_cv_geom_manifest.csv')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/qa_firstchar_patch_ab')
    ap.add_argument('--seed', type=int, default=20260421)
    ap.add_argument('--samples-per-group', type=int, default=12)
    ap.add_argument('--patch-width-normal7', type=int, default=32)
    ap.add_argument('--patch-width-green8', type=int, default=34)
    return ap.parse_args()


DATASET_SPECS = [
    {'key': 'ccpd2019', 'family': 'normal7', 'source': 'main'},
    {'key': 'ccpd2020_green', 'family': 'green8', 'source': 'main'},
    {'key': 'crpd_all_raw', 'family': 'normal7', 'source': 'crpd'},
    {'key': 'cblprd_330k', 'family': 'normal7', 'source': 'cblprd_cv'},
    {'key': 'cblprd_330k', 'family': 'green8', 'source': 'cblprd_cv'},
]

PATCH_W = {
    'normal7': 32,
    'green8': 34,
}

PATCH_X_RATIO = {
    'normal7': (0.00, 0.34),
    'green8': (0.00, 0.36),
}


def imread_any(path: Path):
    path = Path(path)
    if path.suffix.lower() == '.ppm':
        with path.open('rb') as f:
            blob = f.read()
        arr = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    return cv2.imread(str(path))


def read_csv(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def resolve_source_image(row, source_kind):
    if source_kind == 'crpd':
        return Path(row['img_path'])
    if source_kind == 'cblprd_cv':
        return Path(row['img_path'])
    rel = (row.get('img_rel_path') or '').strip()
    if rel:
        return ROOT / rel
    return Path(row['img_path'])


def board_from_row(row, source_kind):
    img_path = resolve_source_image(row, source_kind)
    img = imread_any(img_path)
    if img is None:
        return None, {'error': f'imread_failed:{img_path}'}

    if source_kind == 'crpd':
        quad = parse_ccpd_quad_from_name(str(img_path))
        if quad is None:
            quad_txt = row.get('quad')
            if quad_txt:
                vals = [float(x) for x in quad_txt.replace('&', ',').replace('_', ',').split(',') if x.strip()]
                if len(vals) == 8:
                    quad = np.asarray(vals, dtype=np.float32).reshape(4, 2)
    else:
        quad = parse_ccpd_quad_from_name(str(img_path))

    if quad is None:
        return None, {'error': f'quad_missing:{img_path}'}

    prepared, occ, warped, _ordered_quad, _matrix = prepare_board_ocr_input_from_quad_bgr888(
        img, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0
    )
    meta = {
        'img_path': str(img_path),
        'occ_ratio': float(occ),
        'warped_w': int(warped.shape[1]),
        'warped_h': int(warped.shape[0]),
    }
    return prepared, meta


def crop_slot_prior(board, family):
    h, w = board.shape[:2]
    x1r, x2r = PATCH_X_RATIO[family]
    x1 = max(0, min(w - 1, int(round(w * x1r))))
    x2 = max(x1 + 1, min(w, int(round(w * x2r))))
    patch = board[:, x1:x2]
    patch = cv2.resize(patch, (PATCH_W[family], 24), interpolation=cv2.INTER_NEAREST)
    return patch, {'x1': x1, 'x2': x2, 'w': w, 'h': h}


def crop_projection(board, family):
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    col_energy = inv.sum(axis=0).astype(np.float32)
    if np.max(col_energy) <= 1e-6:
        return crop_slot_prior(board, family)

    smooth = cv2.GaussianBlur(col_energy.reshape(1, -1), (1, 1), 0).reshape(-1)
    peak = int(np.argmax(smooth))
    threshold = max(float(smooth[peak]) * 0.30, float(np.mean(smooth) * 1.10))

    left = peak
    right = peak
    while left > 0 and smooth[left - 1] >= threshold:
        left -= 1
    while right < len(smooth) - 1 and smooth[right + 1] >= threshold:
        right += 1

    pad_l = 4 if family == 'normal7' else 5
    pad_r = 3 if family == 'normal7' else 4
    x1 = max(0, left - pad_l)
    x2 = min(board.shape[1], right + 1 + pad_r)

    max_w = 36 if family == 'normal7' else 38
    if x2 - x1 > max_w:
        x2 = x1 + max_w
    if x2 <= x1:
        return crop_slot_prior(board, family)

    patch = board[:, x1:x2]
    patch = cv2.resize(patch, (PATCH_W[family], 24), interpolation=cv2.INTER_NEAREST)
    return patch, {
        'x1': int(x1),
        'x2': int(x2),
        'w': int(board.shape[1]),
        'h': int(board.shape[0]),
        'peak': int(peak),
        'thr': float(threshold),
    }


def add_label(img, text, scale=0.45, bg=18):
    canvas = np.full((img.shape[0] + 22, img.shape[1], 3), bg, dtype=np.uint8)
    canvas[22:, :] = img
    cv2.putText(canvas, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_card(board, slot_patch, proj_patch, row, dataset_key, family, slot_meta, proj_meta):
    board_big = cv2.resize(board, (94 * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
    slot_big = cv2.resize(slot_patch, (slot_patch.shape[1] * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
    proj_big = cv2.resize(proj_patch, (proj_patch.shape[1] * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)

    top = add_label(board_big, f"{dataset_key} {family} {row['text']}")
    mid = add_label(slot_big, f"A slot x=[{slot_meta['x1']},{slot_meta['x2']})")
    bot = add_label(proj_big, f"B proj x=[{proj_meta['x1']},{proj_meta['x2']})")

    w = max(top.shape[1], mid.shape[1], bot.shape[1])
    def pad(img):
        if img.shape[1] == w:
            return img
        out = np.full((img.shape[0], w, 3), 0, dtype=np.uint8)
        out[:, :img.shape[1]] = img
        return out

    return np.concatenate([pad(top), pad(mid), pad(bot)], axis=0)


def save_sheet(cards, out_path, cols=3, pad=10):
    if not cards:
        return
    cell_h = max(x.shape[0] for x in cards)
    cell_w = max(x.shape[1] for x in cards)
    rows = math.ceil(len(cards) / cols)
    sheet = np.full((rows * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3), 0, dtype=np.uint8)
    for i, img in enumerate(cards):
        r = i // cols
        c = i % cols
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def sample_rows(rows, n, seed):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:min(n, len(rows))]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    detail_dir = out_dir / 'details'
    qa_dir = out_dir / 'qa'
    patch_dir = out_dir / 'patches'
    for d in [out_dir, detail_dir, qa_dir, patch_dir]:
        d.mkdir(parents=True, exist_ok=True)

    main_rows = read_csv(Path(args.main_manifest))
    crpd_rows = read_csv(Path(args.crpd_manifest))
    cblprd_cv_rows = read_csv(Path(args.cblprd_cv_manifest))

    source_map = {
        'main': main_rows,
        'crpd': crpd_rows,
        'cblprd_cv': cblprd_cv_rows,
    }

    summary = {'groups': {}, 'errors': Counter()}

    for spec in DATASET_SPECS:
        dataset_key = spec['key']
        family = spec['family']
        source_kind = spec['source']
        rows = source_map[source_kind]
        picked = [r for r in rows if r.get('dataset_name') == dataset_key and r.get('family') == family]
        picked = sample_rows(picked, args.samples_per_group, args.seed + hash((dataset_key, family, source_kind)) % 100000)

        cards = []
        details = []
        for idx, row in enumerate(picked):
            board, meta = board_from_row(row, source_kind)
            if board is None:
                summary['errors'][meta['error']] += 1
                continue
            slot_patch, slot_meta = crop_slot_prior(board, family)
            proj_patch, proj_meta = crop_projection(board, family)

            tag = f'{dataset_key}__{family}__{idx:03d}'
            board_path = patch_dir / f'{tag}_board.png'
            slot_path = patch_dir / f'{tag}_A_slot.png'
            proj_path = patch_dir / f'{tag}_B_proj.png'
            cv2.imwrite(str(board_path), board)
            cv2.imwrite(str(slot_path), slot_patch)
            cv2.imwrite(str(proj_path), proj_patch)

            cards.append(make_card(board, slot_patch, proj_patch, row, dataset_key, family, slot_meta, proj_meta))
            details.append({
                'dataset_name': dataset_key,
                'family': family,
                'text': row.get('text', ''),
                'img_path': meta['img_path'],
                'occ_ratio': round(meta['occ_ratio'], 6),
                'warped_w': meta['warped_w'],
                'warped_h': meta['warped_h'],
                'board_path': str(board_path),
                'slot_patch_path': str(slot_path),
                'proj_patch_path': str(proj_path),
                'slot_x1': slot_meta['x1'],
                'slot_x2': slot_meta['x2'],
                'proj_x1': proj_meta['x1'],
                'proj_x2': proj_meta['x2'],
                'proj_peak': proj_meta.get('peak', ''),
                'proj_thr': proj_meta.get('thr', ''),
            })

        group_name = f'{dataset_key}__{family}'
        sheet_path = qa_dir / f'{group_name}_contact_sheet.jpg'
        save_sheet(cards, sheet_path, cols=3)

        csv_path = detail_dir / f'{group_name}_details.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            fieldnames = list(details[0].keys()) if details else ['dataset_name', 'family', 'text']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(details)

        summary['groups'][group_name] = {
            'dataset_name': dataset_key,
            'family': family,
            'requested': args.samples_per_group,
            'accepted': len(details),
            'contact_sheet': str(sheet_path),
            'details_csv': str(csv_path),
        }

    summary['errors'] = dict(summary['errors'])
    (detail_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
