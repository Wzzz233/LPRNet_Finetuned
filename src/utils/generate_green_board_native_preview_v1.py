#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


METRIC_KEYS = [
    'occ_ratio',
    'mean',
    'border_dark_ratio',
    'left_minus_right',
    'left_edge',
    'mid_edge',
]


def ensure_94x24(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError('img is None')
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if (h, w) == (24, 94):
        return img.copy()
    return cv2.resize(img, (94, 24), interpolation=cv2.INTER_LINEAR)


def gray_stats(img_bgr: np.ndarray) -> dict:
    img_bgr = ensure_94x24(img_bgr)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
    occ_ratio = float((g.mean(axis=0) > 5).mean())
    return {
        'occ_ratio': occ_ratio,
        'mean': float(g.mean()),
        'std': float(g.std()),
        'left_minus_right': float(g[:, :q].mean() - g[:, -q:].mean()),
        'border_dark_ratio': float((g[mask] < 25).mean()),
        'left_edge': float(mag[:, :q].mean()),
        'mid_edge': float(mag[:, q:-q].mean()) if w > 2 * q else float(mag.mean()),
    }


def load_qspec(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'metrics' in data:
        data = data['metrics']
    return data


def within_spec(stats: dict, qspec: dict, metric_keys=None) -> bool:
    metric_keys = metric_keys or METRIC_KEYS
    for key in metric_keys:
        if stats[key] < float(qspec[key]['q10']) or stats[key] > float(qspec[key]['q90']):
            return False
    return True


def spec_score(stats: dict, qspec: dict, metric_keys=None) -> float:
    metric_keys = metric_keys or METRIC_KEYS
    vals = []
    for key in metric_keys:
        q10 = float(qspec[key]['q10'])
        q50 = float(qspec[key].get('q50', (q10 + float(qspec[key]['q90'])) / 2.0))
        q90 = float(qspec[key]['q90'])
        scale = max((q90 - q10) / 2.0, 1e-6)
        vals.append(abs(float(stats[key]) - q50) / scale)
    return float(np.mean(vals))


def sample_params(rng: np.random.Generator) -> dict:
    return {
        'base_shift': float(rng.uniform(8.0, 20.0)),
        'inner_shift': float(rng.uniform(16.0, 38.0)),
        'target_mean': float(rng.uniform(180.0, 182.6)),
        'target_lmr': float(rng.uniform(-12.8, -6.0)),
        'mid_sigma': float(rng.uniform(0.35, 1.15)),
        'mid_width': int(rng.integers(28, 51)),
        'mid_blend': float(rng.uniform(0.35, 0.85)),
        'left_lift': float(rng.uniform(3.0, 10.0)),
        'right_lift': float(rng.uniform(-1.0, 3.0)),
        'left_blur_sigma': float(rng.uniform(0.3, 1.3)),
        'left_blur_mix': float(rng.uniform(0.2, 0.9)),
        'right_blur_sigma': float(rng.uniform(0.0, 0.8)),
        'right_blur_mix': float(rng.uniform(0.0, 0.5)),
        'left_val_final': float(rng.uniform(0.0, 8.0)),
        'right_val_final': float(rng.uniform(0.0, 8.0)),
        'contrast': float(rng.uniform(0.96, 1.08)),
        'left_inner_dark': float(rng.uniform(1.0, 6.0)),
        'right_inner_bright': float(rng.uniform(-1.0, 3.0)),
    }


def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _blend_blur_region(arr: np.ndarray, x1: int, x2: int, sigma: float, mix: float) -> np.ndarray:
    if sigma <= 0 or mix <= 0 or x2 <= x1:
        return arr
    blur = cv2.GaussianBlur(arr, (3, 3), sigma)
    arr[:, x1:x2] = arr[:, x1:x2] * (1.0 - mix) + blur[:, x1:x2] * mix
    return arr


def _adjust_inner_mean(arr: np.ndarray, target_mean: float, border: int = 5) -> np.ndarray:
    g = cv2.cvtColor(_clip_u8(arr), cv2.COLOR_BGR2GRAY)
    cur = float(g.mean())
    delta = target_mean - cur
    if abs(delta) < 1e-3:
        return arr
    arr[:, border:arr.shape[1] - border] += delta * 1.08
    return arr


def transform_dumplike_to_board_native(img: np.ndarray, params: dict) -> np.ndarray:
    x = ensure_94x24(img).astype(np.float32)
    h, w = x.shape[:2]
    border = 5
    q = w // 4

    x += params['base_shift']
    x[:, border:w - border] += params['inner_shift']
    x[:, border:w - border] = (x[:, border:w - border] - 128.0) * params['contrast'] + 128.0

    inner_left_end = min(border + 24, w - border)
    inner_right_start = max(border, w - 24)
    x[:, border:inner_left_end] += params['left_inner_dark']
    x[:, inner_right_start:w - border] += params['right_inner_bright']

    x[:, :q] += params['left_lift']
    x[:, w - q:] += params['right_lift']

    mid_width = int(np.clip(params['mid_width'], 8, w - 2 * border))
    x1 = max(border, (w - mid_width) // 2)
    x2 = min(w - border, x1 + mid_width)
    x = _blend_blur_region(x, x1, x2, params['mid_sigma'], params['mid_blend'])

    left_end = min(24, w - border)
    right_start = max(border, w - 24)
    x = _blend_blur_region(x, 0, left_end, params['left_blur_sigma'], params['left_blur_mix'])
    x = _blend_blur_region(x, right_start, w, params['right_blur_sigma'], params['right_blur_mix'])

    # lock both side borders to the board-native dark-border regime first
    x[:, :border] = params['left_val_final']
    x[:, -border:] = params['right_val_final']

    # steer left-minus-right toward the target by compensating only the quarter interiors
    for _ in range(2):
        tmp = _clip_u8(x)
        g = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        cur_lmr = float(g[:, :q].mean() - g[:, -q:].mean())
        diff = params['target_lmr'] - cur_lmr
        x[:, border:q] += diff * 1.18
        x[:, w - q:w - border] -= diff * 0.18
        x[:, :border] = params['left_val_final']
        x[:, -border:] = params['right_val_final']

    x = _adjust_inner_mean(x, params['target_mean'], border=border)

    # one last mild lmr correction after mean adjustment
    tmp = _clip_u8(x)
    g = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    cur_lmr = float(g[:, :q].mean() - g[:, -q:].mean())
    diff = params['target_lmr'] - cur_lmr
    x[:, border:q] += diff * 0.70
    x[:, w - q:w - border] -= diff * 0.10

    x[:, :border] = params['left_val_final']
    x[:, -border:] = params['right_val_final']
    return _clip_u8(x)


def make_card(base_img: np.ndarray, gen_img: np.ndarray, label: str, stats: dict, score: float) -> np.ndarray:
    scale = 4
    b1 = cv2.resize(ensure_94x24(base_img), (94 * scale, 24 * scale), interpolation=cv2.INTER_NEAREST)
    b2 = cv2.resize(ensure_94x24(gen_img), (94 * scale, 24 * scale), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((max(b1.shape[0], b2.shape[0]) + 48, b1.shape[1] + b2.shape[1] + 12, 3), 255, dtype=np.uint8)
    canvas[2:2 + b1.shape[0], 2:2 + b1.shape[1]] = b1
    canvas[2:2 + b2.shape[0], 8 + b1.shape[1]:8 + b1.shape[1] + b2.shape[1]] = b2
    line1 = f'{label} score={score:.2f}'
    line2 = f"m={stats['mean']:.1f} lmr={stats['left_minus_right']:.1f} bd={stats['border_dark_ratio']:.3f}"
    cv2.putText(canvas, line1, (4, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (4, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def write_contact_sheet(cards, out_path: Path, cols: int = 4):
    if not cards:
        return
    pad = 8
    cell_h = max(x.shape[0] for x in cards)
    cell_w = max(x.shape[1] for x in cards)
    rows = int(np.ceil(len(cards) / cols))
    sheet = np.full((rows * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3), 255, dtype=np.uint8)
    for i, img in enumerate(cards):
        r = i // cols
        c = i % cols
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def read_csv_rows(path: Path, delimiter=','):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def join_manifest_and_details(manifest_rows, detail_rows):
    detail_map = {row['out_rel_path']: row for row in detail_rows}
    joined = []
    for row in manifest_rows:
        d = detail_map.get(row['img_rel_path'])
        if d is not None:
            joined.append((row, d))
    return joined


def _candidate_distance(detail_row, qspec):
    vals = {
        'mean': float(detail_row['mean']),
        'left_edge': float(detail_row['left_edge']),
        'mid_edge': float(detail_row['mid_edge']),
        'left_minus_right': float(detail_row['left_minus_right']),
    }
    score = 0.0
    for key, v in vals.items():
        q10 = float(qspec[key]['q10'])
        q50 = float(qspec[key].get('q50', (q10 + float(qspec[key]['q90'])) / 2.0))
        q90 = float(qspec[key]['q90'])
        scale = max((q90 - q10) / 2.0, 1e-6)
        score += abs(v - q50) / scale
    return score / len(vals)


def source_filter(detail_row, args):
    if detail_row['bucket'] != args.required_bucket:
        return False
    mean = float(detail_row['mean'])
    left_edge = float(detail_row['left_edge'])
    mid_edge = float(detail_row['mid_edge'])
    return (
        args.mean_min <= mean <= args.mean_max
        and args.left_edge_min <= left_edge <= args.left_edge_max
        and args.mid_edge_min <= mid_edge <= args.mid_edge_max
    )


def main():
    ap = argparse.ArgumentParser(description='Generate accepted-only board-native green preview from dumplike 94x24 seeds.')
    ap.add_argument('--input-manifest', default='/home/wzzz/LPRNet/tmp/green_dumplike_boarddump_bright_v1_20260412_a3100/manifests/train_manifest_dumplike_boarddump_v1.csv')
    ap.add_argument('--details-tsv', default='/home/wzzz/LPRNet/tmp/green_dumplike_boarddump_bright_v1_20260412_a3100/details/accepted.tsv')
    ap.add_argument('--spec-json', default='/home/wzzz/LPRNet/tmp/green_board_domain_spec_v1.json')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--target-count', type=int, default=300)
    ap.add_argument('--required-bucket', default='geometry_clean')
    ap.add_argument('--mean-min', type=float, default=160.0)
    ap.add_argument('--mean-max', type=float, default=182.0)
    ap.add_argument('--left-edge-min', type=float, default=90.0)
    ap.add_argument('--left-edge-max', type=float, default=135.0)
    ap.add_argument('--mid-edge-min', type=float, default=90.0)
    ap.add_argument('--mid-edge-max', type=float, default=140.0)
    ap.add_argument('--candidate-limit', type=int, default=160)
    ap.add_argument('--per-source-target', type=int, default=4)
    ap.add_argument('--max-tries-per-source', type=int, default=600)
    ap.add_argument('--seed', type=int, default=20260413)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images'
    cards_dir = out_dir / 'cards'
    details_dir = out_dir / 'details'
    for d in (images_dir, cards_dir, details_dir):
        d.mkdir(parents=True, exist_ok=True)

    qspec = load_qspec(Path(args.spec_json))
    manifest_rows = read_csv_rows(Path(args.input_manifest))
    detail_rows = read_csv_rows(Path(args.details_tsv), delimiter='\t')
    joined = join_manifest_and_details(manifest_rows, detail_rows)
    candidates = [(m, d) for (m, d) in joined if source_filter(d, args)]
    candidates.sort(key=lambda x: _candidate_distance(x[1], qspec))
    if args.candidate_limit > 0:
        candidates = candidates[:args.candidate_limit]

    accepted_rows = []
    accepted_cards = []
    source_success_counts = Counter()
    source_try_counts = {}
    province_counts = Counter()
    all_stats = []

    for source_idx, (src_manifest_row, src_detail_row) in enumerate(candidates):
        if len(accepted_rows) >= args.target_count:
            break
        src_img = cv2.imread(src_manifest_row['img_path'], cv2.IMREAD_COLOR)
        if src_img is None:
            continue
        src_img = ensure_94x24(src_img)
        tries = 0
        accepted = 0
        while tries < args.max_tries_per_source and accepted < args.per_source_target and len(accepted_rows) < args.target_count:
            tries += 1
            params = sample_params(rng)
            gen = transform_dumplike_to_board_native(src_img, params)
            stats = gray_stats(gen)
            if not within_spec(stats, qspec):
                continue
            score = spec_score(stats, qspec)
            text = src_manifest_row['text']
            prov = text[0]
            out_name = f'preview_s{source_idx:03d}_v{accepted:02d}_{text}.ppm'
            rel_path = f'images/{args.required_bucket}/{prov}/{out_name}'
            abs_path = out_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(abs_path), gen)
            accepted_rows.append({
                'source_idx': source_idx,
                'variant_idx': accepted,
                'text': text,
                'province': prov,
                'bucket': src_detail_row['bucket'],
                'src_img_path': src_manifest_row['img_path'],
                'src_img_rel_path': src_manifest_row['img_rel_path'],
                'out_img_path': str(abs_path),
                'out_img_rel_path': rel_path,
                'score': f'{score:.6f}',
                'mean': f"{stats['mean']:.6f}",
                'left_minus_right': f"{stats['left_minus_right']:.6f}",
                'border_dark_ratio': f"{stats['border_dark_ratio']:.6f}",
                'left_edge': f"{stats['left_edge']:.6f}",
                'mid_edge': f"{stats['mid_edge']:.6f}",
                'occ_ratio': f"{stats['occ_ratio']:.6f}",
                'params_json': json.dumps(params, ensure_ascii=False),
            })
            all_stats.append(stats)
            province_counts[prov] += 1
            source_success_counts[src_manifest_row['img_rel_path']] += 1
            if len(accepted_cards) < 60:
                card = make_card(src_img, gen, f'{text} v{accepted:02d}', stats, score)
                accepted_cards.append(card)
                card_path = cards_dir / f'card_s{source_idx:03d}_v{accepted:02d}_{text}.jpg'
                cv2.imwrite(str(card_path), card)
            accepted += 1
        source_try_counts[src_manifest_row['img_rel_path']] = {'tries': tries, 'accepted': accepted}

    if not accepted_rows:
        raise RuntimeError('no accepted samples generated; widen candidate filter or increase tries')

    details_path = details_dir / 'accepted.tsv'
    with details_path.open('w', encoding='utf-8', newline='') as f:
        fieldnames = list(accepted_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(accepted_rows)

    contact_sheet_path = out_dir / 'contact_sheet.png'
    write_contact_sheet(accepted_cards, contact_sheet_path, cols=3)

    summary = {
        'experiment_name': 'green_board_native_preview_v1',
        'spec_json': args.spec_json,
        'input_manifest': args.input_manifest,
        'details_tsv': args.details_tsv,
        'target_count': args.target_count,
        'accepted_count': len(accepted_rows),
        'candidate_source_count': len(candidates),
        'successful_source_count': int(sum(1 for v in source_try_counts.values() if v['accepted'] > 0)),
        'per_source_target': args.per_source_target,
        'max_tries_per_source': args.max_tries_per_source,
        'filter': {
            'bucket': args.required_bucket,
            'mean': [args.mean_min, args.mean_max],
            'left_edge': [args.left_edge_min, args.left_edge_max],
            'mid_edge': [args.mid_edge_min, args.mid_edge_max],
            'candidate_limit': args.candidate_limit,
        },
        'province_counts': dict(province_counts),
        'source_success_counts_top20': dict(sorted(source_success_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]),
        'contact_sheet': str(contact_sheet_path),
        'accepted_tsv': str(details_path),
        'aggregate': {},
    }
    for key in METRIC_KEYS:
        vals = np.asarray([float(r[key]) for r in accepted_rows], dtype=np.float32)
        summary['aggregate'][key] = {
            'mean': float(vals.mean()),
            'min': float(vals.min()),
            'q10': float(np.quantile(vals, 0.10)),
            'q50': float(np.quantile(vals, 0.50)),
            'q90': float(np.quantile(vals, 0.90)),
            'max': float(vals.max()),
            'target_q10': float(qspec[key]['q10']),
            'target_q90': float(qspec[key]['q90']),
        }

    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
