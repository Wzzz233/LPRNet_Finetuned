#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import sys

REPO_ROOT = Path('/home/wzzz/LPRNet')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from load_data import prepare_board_ocr_input_from_quad_bgr888  # noqa: E402

FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type',
    'source', 'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad',
    'can_parse_ccpd_geom', 'can_perspective', 'bbox_source', 'quad_source',
    'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
]

ALL_PROVINCES = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新']
TARGETED_PROVINCES = {'闽', '苏', '沪', '浙', '粤', '豫', '赣', '皖'}
TARGET_PROVINCES_DEFAULT = ','.join(ALL_PROVINCES)
DIFF_RATIOS_DEFAULT = 'geometry_clean:0.15,board_mid_occ:0.40,board_low_occ:0.35,board_extreme_tail:0.10'
PROV_RATIOS_DEFAULT = ','.join(f"{p}:{4 if p in TARGETED_PROVINCES else 1}" for p in ALL_PROVINCES)


def parse_ratios(spec: str, allowed_keys=None):
    out = {}
    total = 0.0
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(':', 1)
        key = key.strip()
        val = float(value.strip())
        out[key] = val
        total += val
    if allowed_keys is not None:
        bad = [k for k in out if k not in allowed_keys]
        if bad:
            raise ValueError(f'unknown keys in ratio spec: {bad}')
    if total <= 0:
        raise ValueError('ratio total must be > 0')
    return {k: v / total for k, v in out.items()}


def allocate_counts(total: int, ratio_map: dict):
    raw = {k: total * v for k, v in ratio_map.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    remain = total - sum(base.values())
    order = sorted(ratio_map.keys(), key=lambda k: (raw[k] - base[k]), reverse=True)
    for k in order[:remain]:
        base[k] += 1
    return base


def parse_quad(text: str):
    arr = np.asarray(json.loads(text), dtype=np.float32)
    return arr.reshape(4, 2)


def gray_stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    q = max(1, w // 4)
    bw = max(1, w // 16)
    bh = max(1, h // 6)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mask = np.zeros_like(g, dtype=bool)
    mask[:, :bw] = True
    mask[:, -bw:] = True
    mask[:bh, :] = True
    mask[-bh:, :] = True
    return {
        'mean': float(g.mean()),
        'std': float(g.std()),
        'left_minus_right': float(g[:, :q].mean() - g[:, -q:].mean()),
        'border_dark_ratio': float((g[mask] < 25).mean()),
        'left_edge': float(mag[:, :q].mean()),
        'mid_edge': float(mag[:, q:-q].mean()) if w > 2 * q else float(mag.mean()),
    }


def brighten_to_target(img, target_mean):
    cur = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    out = img.astype(np.float32) + (target_mean - cur)
    return np.clip(out, 0, 255).astype(np.uint8)


def soften_mid_detail(img):
    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (3, 3), 0.9)
    x = np.linspace(0, 1, w, dtype=np.float32)
    mid = 1.0 - np.minimum(1.0, np.abs(x - 0.5) / 0.45)
    keep = (0.35 + 0.45 * (1.0 - mid))[None, :, None]
    out = img.astype(np.float32) * keep + blur.astype(np.float32) * (1.0 - keep)
    return np.clip(out, 0, 255).astype(np.uint8)


def left_dark_gradient(img, strength):
    h, w = img.shape[:2]
    ramp = np.linspace(1.0 - strength, 1.02, w, dtype=np.float32)
    out = img.astype(np.float32) * ramp[None, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def soften_black_border(img, rng, target_ratio):
    out = img.copy().astype(np.float32)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    bw = max(1, w // 16)
    bh = max(1, h // 6)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :bw] = True
    mask[:, -bw:] = True
    mask[:bh, :] = True
    mask[-bh:, :] = True
    dark = mask & (g < 25)
    if dark.mean() <= target_ratio:
        return img
    vals = out[dark]
    idx = np.arange(vals.shape[0])
    rng.shuffle(idx)
    keep_dark = int(vals.shape[0] * target_ratio / max(float(dark.mean()), 1e-6))
    brighten_idx = idx[keep_dark:]
    if brighten_idx.size > 0:
        vals[brighten_idx] = np.maximum(vals[brighten_idx], rng.uniform(35, 65, size=vals[brighten_idx].shape))
    out[dark] = vals
    edge_vals = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if float((edge_vals[mask] < 25).mean()) < 0.18:
        restore_idx = idx[:max(1, int(vals.shape[0] * 0.22))]
        vals[restore_idx] = np.minimum(vals[restore_idx], rng.uniform(0, 18, size=vals[restore_idx].shape))
        out[dark] = vals
    return np.clip(out, 0, 255).astype(np.uint8)


def dump_like_transform(prepared_bgr, rng):
    img = prepared_bgr.copy()
    img = brighten_to_target(img, rng.uniform(166, 182))
    img = soften_mid_detail(img)
    img = left_dark_gradient(img, strength=float(rng.uniform(0.06, 0.11)))
    img = soften_black_border(img, rng, target_ratio=float(rng.uniform(0.22, 0.32)))
    img = cv2.GaussianBlur(img, (3, 3), float(rng.uniform(0.35, 0.65)))
    return img


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_ppm(path: Path, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode('ascii'))
        f.write(rgb.tobytes())


def make_manifest_row(abs_path: Path, rel_path: str, text: str, dataset_name: str, source_name: str):
    return {
        'img_path': str(abs_path),
        'img_rel_path': rel_path,
        'dataset_name': dataset_name,
        'split': 'train',
        'text': text,
        'plate_len': len(text),
        'family': 'green8',
        'sub_type': 'green_small',
        'source': source_name,
        'is_real': 0,
        'need_tilt_aug': 0,
        'preprocess_group': 'board_dump',
        'has_bbox': 0,
        'has_quad': 0,
        'can_parse_ccpd_geom': 0,
        'can_perspective': 0,
        'bbox_source': 'none',
        'quad_source': 'none',
        'ocr_channel_order': 'bgr',
        'ocr_crop_mode': 'board_dump',
        'ocr_resize_mode': 'letterbox',
        'ocr_resize_kernel': 'nn',
        'ocr_preproc': 'none',
        'ocr_min_occ_ratio': 1.0,
        'ocr_quad_pad_ratio': 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description='Generate dumplike bright v1 board-dump style 94x24 training data and manifest rows.')
    ap.add_argument('--accepted-tsv', required=True)
    ap.add_argument('--source-root', required=True, help='root directory for rel_path in accepted.tsv')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--count', type=int, default=400)
    ap.add_argument('--target-provinces', default=TARGET_PROVINCES_DEFAULT)
    ap.add_argument('--difficulty-ratios', default=DIFF_RATIOS_DEFAULT)
    ap.add_argument('--province-ratios', default=PROV_RATIOS_DEFAULT)
    ap.add_argument('--seed', type=int, default=20260412)
    ap.add_argument('--dataset-name', default='green_dumplike_boarddump_bright_v1')
    ap.add_argument('--source-name', default='dumplike_board_bright_v1')
    ap.add_argument('--qa-per-bucket', type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images' / 'train'
    manifests_dir = out_dir / 'manifests'
    details_dir = out_dir / 'details'
    qa_dir = out_dir / 'qa_preview' / 'train'
    for d in [images_dir, manifests_dir, details_dir, qa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    target_provinces = [x.strip() for x in args.target_provinces.split(',') if x.strip()]
    if len(set(target_provinces)) != len(target_provinces):
        raise ValueError('target provinces contain duplicates')
    diff_ratios = parse_ratios(args.difficulty_ratios, allowed_keys={'geometry_clean', 'board_mid_occ', 'board_low_occ', 'board_extreme_tail'})
    prov_ratios = parse_ratios(args.province_ratios, allowed_keys=set(target_provinces))
    if any(p not in prov_ratios for p in target_provinces):
        missing = [p for p in target_provinces if p not in prov_ratios]
        raise ValueError(f'province ratios missing provinces: {missing}')
    diff_quota = allocate_counts(args.count, diff_ratios)
    prov_quota = allocate_counts(args.count, prov_ratios)
    too_small = [p for p in target_provinces if prov_quota.get(p, 0) <= 0]
    if too_small:
        raise ValueError(f'count={args.count} too small for full province coverage, zero quota provinces={too_small}')

    rows = read_rows(Path(args.accepted_tsv))
    candidates = []
    for row in rows:
        if row.get('split') != 'train':
            continue
        if row.get('province') not in target_provinces:
            continue
        if row.get('bucket') not in diff_quota:
            continue
        candidates.append(row)

    by_pair = defaultdict(list)
    for row in candidates:
        by_pair[(row['province'], row['bucket'])].append(row)
    for items in by_pair.values():
        py_rng.shuffle(items)

    remain_diff = dict(diff_quota)
    remain_prov = dict(prov_quota)
    selected = []
    pair_cursor = defaultdict(int)

    while len(selected) < args.count:
        valid_pairs = []
        for prov in target_provinces:
            if remain_prov.get(prov, 0) <= 0:
                continue
            for bucket in diff_quota:
                if remain_diff.get(bucket, 0) <= 0:
                    continue
                items = by_pair.get((prov, bucket), [])
                if items:
                    valid_pairs.append((prov, bucket))
        if not valid_pairs:
            break
        valid_pairs.sort(key=lambda x: (remain_prov[x[0]], remain_diff[x[1]]), reverse=True)
        prov, bucket = valid_pairs[0]
        items = by_pair[(prov, bucket)]
        row = items[pair_cursor[(prov, bucket)] % len(items)]
        pair_cursor[(prov, bucket)] += 1
        selected.append(dict(row))
        remain_prov[prov] -= 1
        remain_diff[bucket] -= 1

    if len(selected) != args.count:
        raise RuntimeError(f'could only select {len(selected)} rows, target={args.count}, remaining_diff={remain_diff}, remaining_prov={remain_prov}')

    manifest_rows = []
    detail_rows = []
    bucket_preview_count = Counter()
    bucket_counts = Counter()
    prov_counts = Counter()
    all_stats = []

    for idx, row in enumerate(selected):
        src_path = Path(args.source_root) / row['rel_path']
        image = cv2.imread(str(src_path))
        if image is None:
            raise RuntimeError(f'failed to read source image: {src_path}')
        quad = parse_quad(row['board_quad'])
        prepared, occ, warped, ordered_quad, matrix = prepare_board_ocr_input_from_quad_bgr888(
            image, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0
        )
        dumplike = dump_like_transform(prepared, rng)
        stats = gray_stats(dumplike)

        bucket = row['bucket']
        prov = row['province']
        text = row['text']
        bucket_counts[bucket] += 1
        prov_counts[prov] += 1
        all_stats.append(stats)

        rel_path = f'images/train/{bucket}/{prov}/{idx:04d}_{text}.ppm'
        abs_path = out_dir / rel_path
        write_ppm(abs_path, dumplike)
        manifest_rows.append(make_manifest_row(abs_path, rel_path, text, args.dataset_name, args.source_name))
        detail_rows.append({
            'idx': idx,
            'province': prov,
            'bucket': bucket,
            'text': text,
            'src_rel_path': row['rel_path'],
            'out_rel_path': rel_path,
            'mean': f"{stats['mean']:.4f}",
            'left_minus_right': f"{stats['left_minus_right']:.4f}",
            'border_dark_ratio': f"{stats['border_dark_ratio']:.6f}",
            'left_edge': f"{stats['left_edge']:.4f}",
            'mid_edge': f"{stats['mid_edge']:.4f}",
            'occ_ratio': f'{occ:.6f}',
        })

        if bucket_preview_count[bucket] < args.qa_per_bucket:
            bucket_preview_count[bucket] += 1
            warp_big = cv2.resize(warped, (220, 64), interpolation=cv2.INTER_NEAREST)
            prep_big = cv2.resize(prepared, (220, 64), interpolation=cv2.INTER_NEAREST)
            dump_big = cv2.resize(dumplike, (220, 64), interpolation=cv2.INTER_NEAREST)
            card = np.full((150, 680, 3), 255, np.uint8)
            card[8:72, 8:228] = warp_big
            card[8:72, 230:450] = prep_big
            card[8:72, 452:672] = dump_big
            cv2.putText(card, f'{bucket} {prov} {text}', (8, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(card, f"mean={stats['mean']:.1f} lr={stats['left_minus_right']:.1f} bd={stats['border_dark_ratio']:.3f}", (8, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
            cv2.putText(card, f"mid_edge={stats['mid_edge']:.1f} left_edge={stats['left_edge']:.1f}", (8, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
            preview_path = qa_dir / bucket / f'{idx:04d}_{prov}_{text}.jpg'
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), card)

    manifest_path = manifests_dir / 'train_manifest_dumplike_boarddump_v1.csv'
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(manifest_rows)

    details_path = details_dir / 'accepted.tsv'
    with details_path.open('w', encoding='utf-8', newline='') as f:
        fieldnames = list(detail_rows[0].keys()) if detail_rows else ['idx']
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(detail_rows)

    aggregate = {}
    if all_stats:
        for key in all_stats[0].keys():
            aggregate[f'avg_{key}'] = float(np.mean([x[key] for x in all_stats]))
            aggregate[f'min_{key}'] = float(np.min([x[key] for x in all_stats]))
            aggregate[f'max_{key}'] = float(np.max([x[key] for x in all_stats]))

    report = {
        'total': len(manifest_rows),
        'bucket_counts': dict(bucket_counts),
        'province_counts': dict(prov_counts),
        'difficulty_quota': diff_quota,
        'province_quota': prov_quota,
        'manifest_path': str(manifest_path),
        'details_path': str(details_path),
        'aggregate_stats': aggregate,
        'dataset_name': args.dataset_name,
        'source_name': args.source_name,
        'qa_dir': str(qa_dir),
    }
    (out_dir / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
