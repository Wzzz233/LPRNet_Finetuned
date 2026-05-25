#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import generate_green_e16a_nonanhui_ad_balance as e16a
from generate_green_boarddump_exact_templates import boarddump_manifest_row, fieldnames_for_manifest_rows, write_ppm
from prepare_ccpd_splits import decode_ccpd_plate

CCPD2020_PROVINCES = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云']
CCPD2019_PROVINCES = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新']


def parse_args():
    ap = argparse.ArgumentParser(description='Build first-character province patch boarddump dataset from CCPD/cluster dumps/optional CRPD.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--dataset-name', default='green_firstchar_patch_v1')
    ap.add_argument('--source-name', default='firstchar_patch_v1')
    ap.add_argument('--repo-root', default='/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/repo_license_plate_generator')
    ap.add_argument('--lpr-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--seed', type=int, default=20260420)
    ap.add_argument('--max-per-province-ccpd2020', type=int, default=160)
    ap.add_argument('--max-per-province-ccpd2019', type=int, default=24)
    ap.add_argument('--max-cluster-texts-per-cluster', type=int, default=12)
    ap.add_argument('--cluster-repeat-per-text', type=int, default=8)
    ap.add_argument('--max-crpd-per-province', type=int, default=12)
    ap.add_argument('--ccpd2020-root', default='/home/wzzz/LPRNet/datasets/CCPD2020/ccpd_green')
    ap.add_argument('--ccpd2019-root', default='/home/wzzz/LPRNet/CCPD2019')
    ap.add_argument('--ccpd2019-split-dir', default='/home/wzzz/LPRNet/CCPD2019/splits')
    ap.add_argument('--crpd-manifest', default='/home/wzzz/LPRNet/manifests/crpd_all_raw_board_v1_supported.csv')
    ap.add_argument('--cluster-csv', action='append', default=[])
    ap.add_argument('--contact-limit', type=int, default=64)
    return ap.parse_args()


def load_prepare_board(lpr_root: str):
    lpr_root = str(Path(lpr_root).resolve())
    for p in [lpr_root, str(Path(lpr_root) / 'src')]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from load_data import prepare_board_ocr_input_from_quad_bgr888  # type: ignore
        return prepare_board_ocr_input_from_quad_bgr888
    except Exception:
        def _order_quad_points(pts):
            q = np.asarray(pts, dtype=np.float32).reshape(4, 2)
            s = q.sum(axis=1)
            d = np.diff(q, axis=1).reshape(-1)
            out = np.zeros((4, 2), dtype=np.float32)
            out[0] = q[np.argmin(s)]
            out[2] = q[np.argmax(s)]
            out[1] = q[np.argmin(d)]
            out[3] = q[np.argmax(d)]
            return out

        def _clip_quad_to_image(pts, img_w, img_h):
            q = _order_quad_points(pts).copy()
            q[:, 0] = np.clip(q[:, 0], 0, max(0, img_w - 1))
            q[:, 1] = np.clip(q[:, 1], 0, max(0, img_h - 1))
            return q.astype(np.float32)

        def _quad_edge_lengths(quad):
            width_top = float(np.linalg.norm(quad[1] - quad[0]))
            width_bottom = float(np.linalg.norm(quad[2] - quad[3]))
            height_left = float(np.linalg.norm(quad[3] - quad[0]))
            height_right = float(np.linalg.norm(quad[2] - quad[1]))
            return width_top, width_bottom, height_left, height_right

        def _normalize_vec2(v):
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                return np.asarray([1.0, 0.0], dtype=np.float32)
            return (v / n).astype(np.float32)

        def _calc_warp_padding_ratio(quad):
            width_top, width_bottom, height_left, height_right = _quad_edge_lengths(quad)
            mean_w = max(1.0, 0.5 * (width_top + width_bottom))
            mean_h = max(1.0, 0.5 * (height_left + height_right))
            aspect = mean_w / mean_h
            pad_w = 0.06 if aspect >= 3.0 else 0.10
            pad_h = 0.18 if aspect >= 3.0 else 0.22
            return float(pad_w), float(pad_h)

        def _expand_quad_for_ocr(pts, img_w, img_h, pad_w_ratio, pad_h_ratio):
            q = _order_quad_points(pts).astype(np.float32)
            center = np.mean(q, axis=0)
            ux = _normalize_vec2((q[1] - q[0]) + (q[2] - q[3]))
            vx = _normalize_vec2((q[3] - q[0]) + (q[2] - q[1]))
            w_top, w_bottom, h_left, h_right = _quad_edge_lengths(q)
            half_w = max(1.0, 0.25 * (w_top + w_bottom) * (1.0 + float(pad_w_ratio)))
            half_h = max(1.0, 0.25 * (h_left + h_right) * (1.0 + float(pad_h_ratio)))
            out = np.zeros((4, 2), dtype=np.float32)
            out[0] = center - ux * half_w - vx * half_h
            out[1] = center + ux * half_w - vx * half_h
            out[2] = center + ux * half_w + vx * half_h
            out[3] = center - ux * half_w + vx * half_h
            return _clip_quad_to_image(out, img_w, img_h)

        def _warp_quad_to_rect(image, pts, pad_ratio=0.0, dynamic_pad=False):
            img_h, img_w = image.shape[:2]
            quad = _clip_quad_to_image(pts, img_w, img_h)
            if dynamic_pad:
                pad_w, pad_h = _calc_warp_padding_ratio(quad)
                if pad_ratio > 0.0:
                    pad_w += float(pad_ratio)
                    pad_h += float(pad_ratio)
                quad = _expand_quad_for_ocr(quad, img_w, img_h, pad_w, pad_h)
            width_top, width_bottom, height_left, height_right = _quad_edge_lengths(quad)
            dst_w = max(1, int(max(width_top, width_bottom) + 0.5))
            dst_h = max(1, int(max(height_left, height_right) + 0.5))
            dst = np.array([[0.0, 0.0], [dst_w - 1.0, 0.0], [dst_w - 1.0, dst_h - 1.0], [0.0, dst_h - 1.0]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
            warped = cv2.warpPerspective(image, matrix, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return warped, quad, matrix

        def _fallback_prepare_board(image, quad, in_w, in_h, resize_mode, resize_kernel, preproc_mode, channel_order, quad_pad_ratio=0.0):
            warped, ordered_quad, matrix = _warp_quad_to_rect(
                image,
                quad,
                pad_ratio=quad_pad_ratio,
                dynamic_pad=True,
            )
            if resize_mode == 'letterbox':
                interp = cv2.INTER_NEAREST if resize_kernel == 'nn' else cv2.INTER_LINEAR
                src_h, src_w = warped.shape[:2]
                scale = min(in_w / max(1, src_w), in_h / max(1, src_h))
                scaled_w = max(1, min(in_w, int(src_w * scale + 0.5)))
                scaled_h = max(1, min(in_h, int(src_h * scale + 0.5)))
                resized = cv2.resize(warped, (scaled_w, scaled_h), interpolation=interp)
                out = np.zeros((in_h, in_w, 3), dtype=warped.dtype)
                x0 = (in_w - scaled_w) // 2
                y0 = (in_h - scaled_h) // 2
                out[y0:y0 + scaled_h, x0:x0 + scaled_w] = resized
                occ = scaled_w / float(in_w)
            else:
                interp = cv2.INTER_NEAREST if resize_kernel == 'nn' else cv2.INTER_LINEAR
                out = cv2.resize(warped, (in_w, in_h), interpolation=interp)
                occ = 1.0
            if channel_order == 'rgb':
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            return out, occ, warped, ordered_quad, matrix

        return _fallback_prepare_board


def parse_quad_string(raw: str):
    text = str(raw).strip().replace('&', ',').replace('_', ',').replace(';', ',')
    vals = [float(x) for x in text.split(',') if str(x).strip()]
    if len(vals) != 8:
        raise ValueError(f'bad quad string: {raw!r}')
    return np.asarray(vals, dtype=np.float32).reshape(4, 2)


def patch_crop_from_board(board_img: np.ndarray, patch_w: int = 32, out_h: int = 24):
    if board_img is None or board_img.size == 0:
        return None
    h, w = board_img.shape[:2]
    patch_w = min(w, patch_w)
    x2 = patch_w
    patch = board_img[:, :x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (patch_w, out_h), interpolation=cv2.INTER_NEAREST)


def add_label(img: np.ndarray, text: str, scale: float = 0.45):
    canvas = np.full((img.shape[0] + 24, img.shape[1], 3), 20, dtype=np.uint8)
    canvas[24:, :] = img
    cv2.putText(canvas, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_contact_sheet(cards, out_path: Path, cols: int = 4, pad: int = 8):
    if not cards:
        return
    cell_h = max(img.shape[0] for img in cards)
    cell_w = max(img.shape[1] for img in cards)
    rows_n = int(math.ceil(len(cards) / cols))
    sheet = np.full((rows_n * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3), 0, dtype=np.uint8)
    for i, img in enumerate(cards):
        r = i // cols
        c = i % cols
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def dedupe_keep_order(items, key_fn):
    out = []
    seen = set()
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def collect_ccpd2020_samples(root: Path):
    out = []
    for split in ['train', 'val', 'test']:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for img_path in sorted(split_dir.glob('*.jpg')):
            try:
                text = decode_ccpd_plate(img_path.name)
            except Exception:
                continue
            if len(text) != 8 or text[0] not in CCPD2020_PROVINCES:
                continue
            out.append({'source_kind': 'ccpd2020_green', 'split': split, 'img_path': img_path, 'text': text})
    return out


def collect_ccpd2019_samples(dataset_root: Path, split_dir: Path):
    out = []
    for split in ['train', 'val', 'test']:
        split_file = split_dir / f'{split}.txt'
        if not split_file.exists():
            continue
        for raw in split_file.read_text(encoding='utf-8').splitlines():
            rel = raw.strip().replace('\\', '/')
            if not rel:
                continue
            img_path = dataset_root / rel
            if not img_path.exists():
                continue
            try:
                text = decode_ccpd_plate(rel)
            except Exception:
                continue
            if len(text) != 7 or text[0] not in CCPD2019_PROVINCES:
                continue
            out.append({'source_kind': 'ccpd2019', 'split': split, 'img_path': img_path, 'text': text})
    return out


def collect_cluster_texts(csv_paths, max_texts_per_cluster):
    grouped = defaultdict(list)
    for csv_path in csv_paths:
        p = Path(csv_path)
        if not p.exists():
            continue
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                text = (row.get('gt_text') or '').strip()
                cluster = (row.get('cluster') or p.stem).strip() or p.stem
                if len(text) != 8:
                    continue
                grouped[cluster].append({'text': text, 'csv_path': str(p), 'failure_type': (row.get('failure_type') or '').strip()})
    out = []
    for cluster, rows in grouped.items():
        uniq = dedupe_keep_order(rows, key_fn=lambda x: x['text'])
        out.extend([dict(item, cluster=cluster) for item in uniq[:max_texts_per_cluster]])
    return out


def collect_crpd_texts(manifest_path: Path):
    out = []
    if not manifest_path.exists():
        return out
    with manifest_path.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            text = (row.get('text') or '').strip()
            if len(text) not in (7, 8):
                continue
            img_path = Path(row.get('img_path') or '')
            quad = row.get('quad') or ''
            if not img_path.exists() or not quad:
                continue
            out.append({'source_kind': 'crpd', 'img_path': img_path, 'text': text, 'quad': quad, 'split': row.get('split', 'train')})
    return out


def sample_by_province(items, max_per_province, seed):
    groups = defaultdict(list)
    for item in items:
        groups[item['text'][0]].append(item)
    rng = random.Random(seed)
    sampled = []
    for prov, rows in sorted(groups.items()):
        rows = list(rows)
        rng.shuffle(rows)
        sampled.extend(rows[:max_per_province])
    return sampled


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_root = Path(args.out_dir)
    images_dir = out_root / 'images' / 'train'
    details_dir = out_root / 'details'
    manifests_dir = out_root / 'manifests'
    qa_dir = out_root / 'qa'
    patch_dir = qa_dir / 'patches'
    for d in [images_dir, details_dir, manifests_dir, qa_dir, patch_dir]:
        d.mkdir(parents=True, exist_ok=True)

    prepare_board = load_prepare_board(args.lpr_root)

    ccpd2020_samples = sample_by_province(collect_ccpd2020_samples(Path(args.ccpd2020_root)), args.max_per_province_ccpd2020, args.seed + 11)
    ccpd2019_samples = sample_by_province(collect_ccpd2019_samples(Path(args.ccpd2019_root), Path(args.ccpd2019_split_dir)), args.max-per-province-ccpd2019 if False else args.max_per_province_ccpd2019, args.seed + 13)
    cluster_texts = collect_cluster_texts(args.cluster_csv, args.max_cluster_texts_per_cluster)
    crpd_samples = sample_by_province(collect_crpd_texts(Path(args.crpd_manifest)), args.max_crpd_per_province, args.seed + 17)

    chars_gen, augmenter = e16a.ensure_repo_imports(args.repo_root)
    augmenter = e16a.setup_augmenter(augmenter, e16a.GEOMETRY_CFG)
    canonical_quad = e16a.detect_canonical_plate_quad(augmenter.template_image)

    manifest_rows = []
    accepted_details = []
    cards = []

    def emit(board_img, text, source_kind, source_ref, split='train', extra=None):
        nonlocal manifest_rows, accepted_details, cards
        board_img = np.clip(board_img, 0, 255).astype(np.uint8)
        prov = text[0]
        sample_id = f'{source_kind}-{prov}-{len(accepted_details):05d}'
        rel_path = f'images/train/p{ord(prov):05d}/{sample_id}.ppm'
        abs_path = out_root / rel_path
        write_ppm(abs_path, board_img)
        manifest_rows.append(boarddump_manifest_row(abs_path, rel_path, text, args.dataset_name, args.source_name))
        patch = patch_crop_from_board(board_img)
        patch_rel = f'qa/patches/{sample_id}_patch.png'
        patch_abs = out_root / patch_rel
        if patch is not None:
            cv2.imwrite(str(patch_abs), patch)
        detail = {
            'sample_id': sample_id,
            'text': text,
            'province': prov,
            'source_kind': source_kind,
            'source_ref': source_ref,
            'split': split,
            'img_rel_path': rel_path,
            'img_abs_path': str(abs_path),
            'patch_rel_path': patch_rel if patch is not None else '',
            'patch_abs_path': str(patch_abs) if patch is not None else '',
            'board_w': board_img.shape[1],
            'board_h': board_img.shape[0],
            'patch_w': patch.shape[1] if patch is not None else 0,
            'patch_h': patch.shape[0] if patch is not None else 0,
        }
        if extra:
            detail.update(extra)
        accepted_details.append(detail)
        if len(cards) < args.contact_limit and patch is not None:
            board_vis = cv2.resize(board_img, (94 * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
            patch_vis = cv2.resize(patch, (32 * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
            card = np.concatenate([add_label(board_vis, f'{text} {source_kind}'), add_label(patch_vis, prov)], axis=1)
            cards.append(card)

    for item in ccpd2020_samples:
        img = cv2.imread(str(item['img_path']))
        if img is None:
            continue
        stem_parts = item['img_path'].stem.split('-')
        if len(stem_parts) < 4:
            continue
        quad = parse_quad_string(stem_parts[3].replace('&', ','))
        board_img, occ_ratio, _, _, _ = prepare_board(img, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)
        emit(board_img, item['text'], item['source_kind'], str(item['img_path']), split=item['split'], extra={'occ_ratio': round(float(occ_ratio), 6), 'family': 'green8'})

    for item in ccpd2019_samples:
        img = cv2.imread(str(item['img_path']))
        if img is None:
            continue
        stem_parts = item['img_path'].stem.split('-')
        if len(stem_parts) < 4:
            continue
        quad = parse_quad_string(stem_parts[3].replace('&', ','))
        board_img, occ_ratio, _, _, _ = prepare_board(img, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)
        emit(board_img, item['text'], item['source_kind'], str(item['img_path']), split=item['split'], extra={'occ_ratio': round(float(occ_ratio), 6), 'family': 'blue7'})

    for item in cluster_texts:
        for rep in range(args.cluster_repeat_per_text):
            local_rng = random.Random(args.seed + len(accepted_details) * 101 + rep)
            render = e16a.render_standard_exact_quad(item['text'], chars_gen, augmenter, canonical_quad, prepare_board, local_rng)
            if render['occ_ratio'] < float(e16a.GEOMETRY_CFG['min_occ_ratio']):
                continue
            emit(render['prepared94'], item['text'], f'dump_{item["cluster"]}', item['csv_path'], extra={
                'cluster': item['cluster'],
                'failure_type': item['failure_type'],
                'occ_ratio': round(float(render['occ_ratio']), 6),
                'max_char_angle_error_deg': round(float(render['max_char_angle_error_deg']), 6),
                'family': 'green8',
                'synthetic_from_dump_text': 1,
            })

    for item in crpd_samples:
        img = cv2.imread(str(item['img_path']))
        if img is None:
            continue
        try:
            quad = parse_quad_string(item['quad'])
        except Exception:
            continue
        board_img, occ_ratio, _, _, _ = prepare_board(img, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)
        emit(board_img, item['text'], item['source_kind'], str(item['img_path']), split=item['split'], extra={'occ_ratio': round(float(occ_ratio), 6), 'family': 'real'})

    if not manifest_rows:
        raise RuntimeError('no rows emitted')

    manifest_local = manifests_dir / f'train_manifest_{args.dataset_name}.csv'
    with manifest_local.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_for_manifest_rows(manifest_rows))
        w.writeheader()
        w.writerows(manifest_rows)

    detail_csv = details_dir / 'accepted.csv'
    detail_fields = sorted({k for row in accepted_details for k in row.keys()})
    with detail_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        w.writerows(accepted_details)

    base_count, merged_count = e16a.append_manifest(Path(args.base_manifest), manifest_rows, Path(args.out_manifest))
    save_contact_sheet(cards, qa_dir / 'contact_sheet.jpg', cols=2)

    summary = {
        'dataset_name': args.dataset_name,
        'source_name': args.source_name,
        'accepted_count': len(accepted_details),
        'base_manifest_count': base_count,
        'merged_manifest_count': merged_count,
        'source_kind_counts': dict(sorted(Counter(x['source_kind'] for x in accepted_details).items())),
        'province_counts': dict(sorted(Counter(x['province'] for x in accepted_details).items())),
        'cluster_counts': dict(sorted(Counter(x.get('cluster', '') for x in accepted_details if x.get('cluster')).items())),
        'paths': {
            'out_dir': str(out_root),
            'manifest_local': str(manifest_local),
            'manifest_merged': str(args.out_manifest),
            'accepted_csv': str(detail_csv),
            'contact_sheet': str(qa_dir / 'contact_sheet.jpg'),
            'patch_dir': str(patch_dir),
        },
    }
    (details_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
