#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description='High-res firstchar crop QA across CCPD/CRPD/CBLPRD/raw synthetic sources.')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/qa_firstchar_hires_rawcrop')
    ap.add_argument('--seed', type=int, default=20260421)
    ap.add_argument('--samples-per-group', type=int, default=12)
    ap.add_argument('--ccpd2019-root', default='/home/wzzz/LPRNet/datasets/CCPD2019')
    ap.add_argument('--ccpd2019-split-dir', default='/home/wzzz/LPRNet/CCPD2019/splits')
    ap.add_argument('--ccpd2020-root', default='/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green')
    ap.add_argument('--crpd-root', default='/home/wzzz/LPRNet/datasets/CRPD_all')
    ap.add_argument('--cblprd-root', default='/home/wzzz/LPRNet/datasets/CBLPRD-330k_v1')
    ap.add_argument('--green-synth-manifest', default='/home/wzzz/LPRNet/green_exact_quad_synthetic_v1/manifests/train_synthetic_labels_without_su_hu_holdout.txt')
    return ap.parse_args()


CCPD2020_PROVINCES = set(['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云'])
CCPD2019_PROVINCES = set(['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新'])


def parse_ccpd_quad_from_name(image_name: str):
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) < 4:
        return None
    pts = []
    try:
        for item in parts[3].split('_'):
            xs, ys = item.split('&', 1)
            pts.append((float(xs), float(ys)))
    except Exception:
        return None
    if len(pts) != 4:
        return None
    return np.asarray(pts, dtype=np.float32)


def warp_quad(image, quad):
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    s = q.sum(axis=1)
    d = np.diff(q, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = q[np.argmin(s)]
    ordered[2] = q[np.argmax(s)]
    ordered[1] = q[np.argmin(d)]
    ordered[3] = q[np.argmax(d)]
    width_top = float(np.linalg.norm(ordered[1] - ordered[0]))
    width_bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    height_left = float(np.linalg.norm(ordered[3] - ordered[0]))
    height_right = float(np.linalg.norm(ordered[2] - ordered[1]))
    dst_w = max(1, int(max(width_top, width_bottom) + 0.5))
    dst_h = max(1, int(max(height_left, height_right) + 0.5))
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, ordered


def decode_ccpd_plate(name: str):
    # repo local helper-less minimal parser: only used for sample labels where filename already embeds decoded text via prior code paths.
    stem = Path(name).stem
    parts = stem.split('-')
    if len(parts) >= 5 and '&' in parts[3]:
        # cannot decode raw indices here robustly without repo helper; caller should provide text when possible
        return None
    return None


def family_from_text(text: str):
    if len(text) == 7:
        return 'normal7'
    if len(text) == 8:
        return 'green8'
    return 'other'


def crop_slot_a(warped, family):
    h, w = warped.shape[:2]
    if family == 'normal7':
        x1 = int(round(w * 0.00))
        x2 = int(round(w * 0.17))
    else:
        x1 = int(round(w * 0.00))
        x2 = int(round(w * 0.15))
    x2 = max(x1 + 2, min(w, x2))
    return warped[:, x1:x2], {'x1': x1, 'x2': x2}


def crop_slot_c(warped, family):
    h, w = warped.shape[:2]
    if family == 'normal7':
        x1 = int(round(w * 0.00))
        x2 = int(round(w * 0.20))
    else:
        x1 = int(round(w * 0.00))
        x2 = int(round(w * 0.18))
    x2 = max(x1 + 2, min(w, x2))
    return warped[:, x1:x2], {'x1': x1, 'x2': x2}


def crop_slot_b(warped, family):
    h, w = warped.shape[:2]
    if family == 'normal7':
        x1 = int(round(w * 0.00))
        x2 = int(round(w * 0.23))
    else:
        x1 = int(round(w * 0.00))
        x2 = int(round(w * 0.21))
    x2 = max(x1 + 2, min(w, x2))
    return warped[:, x1:x2], {'x1': x1, 'x2': x2}


def fit_patch(patch, out_w=96, out_h=64):
    if patch is None or patch.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    h, w = patch.shape[:2]
    scale = min(out_w / max(1, w), out_h / max(1, h))
    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))
    resized = cv2.resize(patch, (rw, rh), interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x = (out_w - rw) // 2
    y = (out_h - rh) // 2
    canvas[y:y+rh, x:x+rw] = resized
    return canvas


def add_label(img, text, scale=0.45):
    canvas = np.full((img.shape[0] + 22, img.shape[1], 3), 18, dtype=np.uint8)
    canvas[22:, :] = img
    cv2.putText(canvas, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_sheet(cards, out_path, cols=3, pad=10):
    if not cards:
        return
    cell_h = max(i.shape[0] for i in cards)
    cell_w = max(i.shape[1] for i in cards)
    rows = math.ceil(len(cards) / cols)
    sheet = np.full((rows * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3), 0, dtype=np.uint8)
    for i, img in enumerate(cards):
        r = i // cols
        c = i % cols
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        sheet[y:y+img.shape[0], x:x+img.shape[1]] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def sample_rows(rows, n, seed):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:min(n, len(rows))]


def collect_ccpd2019(root, split_dir):
    out = []
    for split in ['train', 'val', 'test']:
        fp = Path(split_dir) / f'{split}.txt'
        if not fp.exists():
            continue
        for raw in fp.read_text(encoding='utf-8').splitlines():
            rel = raw.strip().replace('\\', '/')
            if not rel:
                continue
            img = Path(root) / rel
            if not img.exists():
                continue
            quad = parse_ccpd_quad_from_name(img.name)
            if quad is None:
                continue
            text = img.stem.split('-')[-3] if False else None
            out.append({'dataset_name': 'ccpd2019', 'family': 'normal7', 'text': '', 'img_path': str(img), 'quad': quad.tolist(), 'split': split})
    return out


def collect_ccpd2020(root):
    out = []
    for split in ['train', 'val', 'test']:
        d = Path(root) / split
        if not d.exists():
            continue
        for img in d.glob('*.jpg'):
            quad = parse_ccpd_quad_from_name(img.name)
            if quad is None:
                continue
            out.append({'dataset_name': 'ccpd2020_green', 'family': 'green8', 'text': '', 'img_path': str(img), 'quad': quad.tolist(), 'split': split})
    return out


def collect_crpd(root):
    out = []
    root = Path(root)
    for subset in ['CRPD_single', 'CRPD_double', 'CRPD_multi']:
        for split in ['train', 'val', 'test']:
            label_dir = root / subset / split / 'labels'
            img_dir = root / subset / split / 'images'
            if not label_dir.exists() or not img_dir.exists():
                continue
            for txt in label_dir.glob('*.txt'):
                img = img_dir / (txt.stem + '.jpg')
                if not img.exists():
                    continue
                for idx, line in enumerate(txt.read_text(encoding='utf-8').splitlines()):
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    vals = list(map(float, parts[:8]))
                    text = parts[9]
                    family = family_from_text(text)
                    if family not in {'normal7', 'green8'}:
                        continue
                    quad = np.asarray(vals, dtype=np.float32).reshape(4, 2)
                    out.append({'dataset_name': 'crpd_all_raw', 'family': family, 'text': text, 'img_path': str(img), 'quad': quad.tolist(), 'split': split, 'subset': subset, 'obj_idx': idx})
    return out


def collect_cblprd(root):
    out = []
    meta = {
        '普通蓝牌': 'normal7',
        '新能源小型车': 'green8',
        '新能源大型车': 'green8',
    }
    for txt_name in ['train.txt', 'val.txt']:
        fp = Path(root) / txt_name
        if not fp.exists():
            continue
        split = txt_name.replace('.txt', '')
        with fp.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                if len(parts) != 3:
                    continue
                rel, text, plate_type = parts
                family = meta.get(plate_type)
                if family not in {'normal7', 'green8'}:
                    continue
                img = Path(root) / rel
                if not img.exists():
                    continue
                out.append({'dataset_name': 'cblprd_330k', 'family': family, 'text': text, 'img_path': str(img), 'split': split, 'plate_type': plate_type})
    return out


def collect_green_synth(manifest_path):
    out = []
    fp = Path(manifest_path)
    if not fp.exists():
        return out
    with fp.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            rel, text = parts
            img = fp.parent.parent / rel
            if not img.exists():
                continue
            family = family_from_text(text)
            if family != 'green8':
                continue
            out.append({'dataset_name': 'green_exact_quad_synthetic_v1', 'family': 'green8', 'text': text, 'img_path': str(img), 'split': 'train'})
    return out


def process_group(rows, dataset_name, family, out_dir, seed, samples_per_group):
    picked = sample_rows(rows, samples_per_group, seed)
    cards = []
    details = []
    patch_dir = out_dir / 'patches'
    patch_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(picked):
        img = cv2.imread(row['img_path'])
        if img is None:
            continue
        if 'quad' in row:
            warped, ordered = warp_quad(img, row['quad'])
        else:
            warped = img.copy()
            ordered = None
        patch_a, meta_a = crop_slot_a(warped, family)
        patch_c, meta_c = crop_slot_c(warped, family)
        patch_b, meta_b = crop_slot_b(warped, family)
        board_vis = fit_patch(warped, 260, 96)
        if ordered is not None:
            raw_vis = img.copy()
            pts = np.asarray(ordered, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(raw_vis, [pts], True, (0, 255, 0), 2)
            raw_vis = fit_patch(raw_vis, 260, 140)
        else:
            raw_vis = fit_patch(img, 260, 140)
        pa = fit_patch(patch_a, 120, 96)
        pc = fit_patch(patch_c, 120, 96)
        pb = fit_patch(patch_b, 120, 96)
        tiles = [
            add_label(raw_vis, f"raw {dataset_name} {family} {row.get('text','')[:12]}"),
            add_label(board_vis, 'warped/raw-crop'),
            add_label(pa, f"A x=[{meta_a['x1']},{meta_a['x2']})"),
            add_label(pc, f"C x=[{meta_c['x1']},{meta_c['x2']})"),
            add_label(pb, f"B x=[{meta_b['x1']},{meta_b['x2']})"),
        ]
        target_h = max(t.shape[0] for t in tiles)
        padded = []
        for t in tiles:
            if t.shape[0] == target_h:
                padded.append(t)
                continue
            canvas = np.zeros((target_h, t.shape[1], 3), dtype=np.uint8)
            canvas[:t.shape[0], :, :] = t
            padded.append(canvas)
        card = np.concatenate(padded, axis=1)
        cards.append(card)
        details.append({
            'dataset_name': dataset_name,
            'family': family,
            'text': row.get('text', ''),
            'img_path': row['img_path'],
            'split': row.get('split', ''),
            'a_x1': meta_a['x1'], 'a_x2': meta_a['x2'],
            'b_x1': meta_b['x1'], 'b_x2': meta_b['x2'],
            'has_quad': int('quad' in row),
        })
    sheet = out_dir / 'qa' / f'{dataset_name}__{family}_contact_sheet.jpg'
    save_sheet(cards, sheet, cols=2)
    csv_path = out_dir / 'details' / f'{dataset_name}__{family}_details.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        fieldnames = list(details[0].keys()) if details else ['dataset_name', 'family', 'text']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(details)
    return {'accepted': len(details), 'contact_sheet': str(sheet), 'details_csv': str(csv_path)}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / 'qa').mkdir(parents=True, exist_ok=True)
    (out_dir / 'details').mkdir(parents=True, exist_ok=True)

    all_groups = defaultdict(list)

    for row in collect_ccpd2019(args.ccpd2019_root, args.ccpd2019_split_dir):
        all_groups[(row['dataset_name'], row['family'])].append(row)
    for row in collect_ccpd2020(args.ccpd2020_root):
        all_groups[(row['dataset_name'], row['family'])].append(row)
    for row in collect_crpd(args.crpd_root):
        all_groups[(row['dataset_name'], row['family'])].append(row)
    for row in collect_cblprd(args.cblprd_root):
        all_groups[(row['dataset_name'], row['family'])].append(row)
    for row in collect_green_synth(args.green_synth_manifest):
        all_groups[(row['dataset_name'], row['family'])].append(row)

    summary = {'groups': {}}
    for (dataset_name, family), rows in sorted(all_groups.items()):
        result = process_group(rows, dataset_name, family, out_dir, args.seed + abs(hash((dataset_name, family))) % 100000, args.samples_per_group)
        summary['groups'][f'{dataset_name}__{family}'] = {
            'available_rows': len(rows),
            **result,
        }

    (out_dir / 'details' / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
