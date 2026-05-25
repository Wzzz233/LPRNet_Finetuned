#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate accepted-only green exact-template data directly in final 94x24 board_dump form.

Core idea:
- reuse E9 exact-template text / geometry chain
- render board-like crop with legal overflow templates
- prepare to final 94x24 OCR input
- search board-native transform params until cluster1 spec is satisfied
- save final images as board_dump rows (no bbox / quad re-warp at train time)
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import generate_green_edgefit_v4_boardlike_equalprov as base
from generate_green_board_native_preview_v1 import (
    ensure_94x24,
    gray_stats,
    load_qspec,
    make_card,
    sample_params,
    spec_score,
    transform_dumplike_to_board_native,
    within_spec,
    write_contact_sheet,
)

DEFAULT_BUCKET_WEIGHTS = {
    'geometry_clean': 0.40,
    'board_mid_occ': 0.30,
    'board_low_occ': 0.20,
    'board_extreme_tail': 0.10,
}

TEMPLATE_MODES = {
    # Renormalized from E9 broad template mix after removing D/F templates.
    'broad_overflow': {
        'AA0': 0.333333,
        'AA1': 0.166667,
        'AA2': 0.166667,
        'AB0': 0.088889,
        'AC0': 0.088889,
        'AE0': 0.088889,
        'AG0': 0.022222,
        'AH0': 0.022222,
        'AJ0': 0.011111,
        'AK0': 0.011111,
    },
    'cluster1_focus': {
        'AA0': 0.50,
        'AB0': 0.10,
        'AC0': 0.10,
        'AE0': 0.10,
        'AA1': 0.10,
        'AA2': 0.10,
    },
    # E12: explicitly add D/F hard negatives back in the same board_dump domain,
    # while still centering AA0/AA1/AA2 and overflow letters around the cluster1 failure mode.
    'anti_collapse_balanced': {
        'AA0': 0.25,
        'AD0': 0.15,
        'AF0': 0.15,
        'AB0': 0.10,
        'AC0': 0.10,
        'AE0': 0.10,
        'AA1': 0.075,
        'AA2': 0.075,
    },
    # E13A: smallest append-only slot-alignment probe.
    # Keep one dominant AA0 template, a small AD0 hard-negative branch to measure AA0->ADA drift,
    # and a light AA1 branch to avoid collapsing all appended mass onto a single exact text family.
    'slotalign_aa0_probe': {
        'AA0': 0.80,
        'AD0': 0.10,
        'AA1': 0.10,
    },
}

TAIL_MODES = {
    'random': 'Fully random 4-digit tails.',
    'anti_collapse': 'Bias tails toward 0/2-heavy motifs like 0222/0022/0202 to target cluster1 alignment collapse.',
    # E13A: hold text mass near the AA02222 neighborhood while varying which one of the last four slots changes.
    'slotalign_aa0': 'Bias tails toward AA0x222 / AA02x22 / AA022x2 / AA0222x style neighborhoods to probe slot alignment instead of only template priors.',
}

FIVE_PROVS = ['陕', '苏', '沪', '浙', '粤']
ALL_PROVS = list(base.ALL_PROVINCES)
DIGITS = list('0123456789')
TARGETISH_TAILS = [
    '0222', '0022', '0202', '0220', '2022', '2202', '2002', '2222',
    '0228', '0282', '0822', '0200', '0020', '2200', '2000', '0226',
]


def apportion(total, weight_map, ordered_keys):
    raw = {k: total * float(weight_map[k]) for k in ordered_keys}
    counts = {k: int(np.floor(raw[k])) for k in ordered_keys}
    rem = total - sum(counts.values())
    if rem > 0:
        order = sorted(ordered_keys, key=lambda k: (raw[k] - counts[k], weight_map[k], k), reverse=True)
        for k in order[:rem]:
            counts[k] += 1
    return counts


def distribute_total_even(total, provinces):
    base_n = total // len(provinces)
    extra = total % len(provinces)
    out = {p: base_n for p in provinces}
    for p in provinces[:extra]:
        out[p] += 1
    return out


def write_ppm(path: Path, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode('ascii'))
        f.write(rgb.tobytes())


def sample_tail(rng, tail_mode):
    if tail_mode == 'random':
        return ''.join(rng.choice(DIGITS) for _ in range(4))
    if tail_mode == 'anti_collapse':
        u = rng.random()
        if u < 0.22:
            return rng.choice(TARGETISH_TAILS)
        if u < 0.52:
            digits = ['0', '2', rng.choice(DIGITS), rng.choice(DIGITS)]
            rng.shuffle(digits)
            if digits[0] not in {'0', '2'}:
                digits[0] = rng.choice(['0', '2'])
            return ''.join(digits)
        if u < 0.72:
            digits = [rng.choice(['0', '2']) for _ in range(3)] + [rng.choice(DIGITS)]
            rng.shuffle(digits)
            return ''.join(digits)
        return ''.join(rng.choice(DIGITS) for _ in range(4))
    if tail_mode == 'slotalign_aa0':
        u = rng.random()
        if u < 0.65:
            slot = rng.randrange(4)
            digits = ['0', '2', '2', '2']
            digits[slot] = rng.choice(DIGITS)
            return ''.join(digits)
        if u < 0.85:
            return rng.choice(TARGETISH_TAILS)
        if u < 0.93:
            digits = ['0', '2', '2', rng.choice(DIGITS)]
            rng.shuffle(digits)
            return ''.join(digits)
        return ''.join(rng.choice(DIGITS) for _ in range(4))
    raise ValueError(f'unknown tail_mode: {tail_mode}')


def make_text_from_pattern(province, pattern, used_texts, rng, tail_mode='random'):
    city = 'A'
    serial1 = pattern[1]
    serial2 = pattern[2]
    while True:
        tail = sample_tail(rng, tail_mode)
        text = province + city + serial1 + serial2 + tail
        if text not in used_texts:
            used_texts.add(text)
            return text


def boarddump_manifest_row(abs_path: Path, rel_path: str, text: str, dataset_name: str, source_name: str):
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


def fieldnames_for_manifest_rows(rows):
    base_fields = [
        'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type', 'source',
        'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad', 'can_parse_ccpd_geom',
        'can_perspective', 'bbox_source', 'quad_source', 'ocr_channel_order', 'ocr_crop_mode',
        'ocr_resize_mode', 'ocr_resize_kernel', 'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
    ]
    if not rows:
        return base_fields
    keys = list(rows[0].keys())
    extras = [k for k in keys if k not in base_fields]
    return base_fields + extras


def parse_bucket_weights(text):
    out = {}
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f'invalid bucket weight item: {part}')
        key, value = part.split('=', 1)
        key = key.strip()
        if key not in DEFAULT_BUCKET_WEIGHTS:
            raise ValueError(f'unknown bucket key: {key}')
        out[key] = float(value)
    missing = [k for k in DEFAULT_BUCKET_WEIGHTS if k not in out]
    if missing:
        raise ValueError(f'missing bucket weights for: {missing}')
    total = sum(out.values())
    if total <= 0:
        raise ValueError('bucket weights must sum to > 0')
    return {k: v / total for k, v in out.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--spec_json', default='/home/wzzz/LPRNet/tmp/green_board_domain_spec_v1.json')
    ap.add_argument('--province_mode', choices=['five', 'all', 'non_anhui'], default='five')
    ap.add_argument('--template_mode', choices=sorted(TEMPLATE_MODES.keys()), required=True)
    ap.add_argument('--total_count', type=int, required=True)
    ap.add_argument('--seed', type=int, default=20260416)
    ap.add_argument('--preview_limit', type=int, default=40)
    ap.add_argument('--param_tries_per_seed', type=int, default=80)
    ap.add_argument('--extra_geom_retry', type=int, default=4)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--source_name', required=True)
    ap.add_argument('--bucket_weights', default='geometry_clean=0.4,board_mid_occ=0.3,board_low_occ=0.2,board_extreme_tail=0.1')
    ap.add_argument('--tail_mode', choices=sorted(TAIL_MODES.keys()), default='random')
    ap.add_argument('--avoid_text_files', nargs='*', default=[])
    return ap.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_dir)
    images_dir = out_root / 'images' / 'train'
    manifests_dir = out_root / 'manifests'
    details_dir = out_root / 'details'
    cards_dir = out_root / 'cards'
    for d in [images_dir, manifests_dir, details_dir, cards_dir]:
        d.mkdir(parents=True, exist_ok=True)

    qspec = load_qspec(Path(args.spec_json))
    bucket_weights = parse_bucket_weights(args.bucket_weights)
    used_texts = base.load_used_texts(args.avoid_text_files)
    chars_gen, augmenter, prepare_fn = base.ensure_repo_imports(args.repo_root)
    if args.province_mode == 'five':
        provinces = FIVE_PROVS
    elif args.province_mode == 'non_anhui':
        provinces = list(base.NON_ANHUI_PROVINCES)
    else:
        provinces = ALL_PROVS
    province_targets = distribute_total_even(args.total_count, provinces)
    template_weights = TEMPLATE_MODES[args.template_mode]
    template_keys = list(template_weights.keys())
    bucket_keys = [k for k, v in bucket_weights.items() if v > 0]

    manifest_rows = []
    detail_rows = []
    plan_rows = []
    cards = []
    rejects = Counter()
    split_texts = defaultdict(set)
    all_stats = []

    for province in provinces:
        template_targets = apportion(province_targets[province], template_weights, template_keys)
        for template in template_keys:
            bucket_targets = apportion(template_targets[template], bucket_weights, bucket_keys)
            for bucket in bucket_keys:
                target = bucket_targets[bucket]
                if target <= 0:
                    continue
                made = 0
                attempts = 0
                max_attempts = base.attempt_budget_for_bucket(bucket, target) * max(1, args.extra_geom_retry)
                while made < target and attempts < max_attempts:
                    attempts += 1
                    text = make_text_from_pattern(province, template, used_texts, rng, tail_mode=args.tail_mode)
                    success = False
                    try:
                        plate_base = base.build_base_plate(text, chars_gen, augmenter)
                        exact_quad_seed = base.make_exact_quad(bucket, rng)
                        _, exact_quad, board_img_raw, board_quad, asym_mode = base.render_plate_with_exact_and_board(
                            plate_base, exact_quad_seed, bucket, rng
                        )
                        board_img, appearance_meta = base.apply_appearance_by_bucket(board_img_raw, bucket, rng)
                        prepared, occ, warped, ordered_quad, matrix = prepare_fn(
                            board_img,
                            board_quad,
                            base.IN_W,
                            base.IN_H,
                            'letterbox',
                            'nn',
                            'none',
                            'bgr',
                            quad_pad_ratio=0.0,
                        )
                        prepared = ensure_94x24(prepared)
                        if not base.accept_bucket(bucket, float(occ)):
                            rejects[(province, template, bucket, 'occ_reject')] += 1
                            used_texts.discard(text)
                            continue

                        best_img = None
                        best_stats = None
                        best_score = None
                        best_params = None
                        for _ in range(args.param_tries_per_seed):
                            params = sample_params(np_rng)
                            gen = transform_dumplike_to_board_native(prepared, params)
                            stats = gray_stats(gen)
                            if not within_spec(stats, qspec):
                                continue
                            score = spec_score(stats, qspec)
                            if best_score is None or score < best_score:
                                best_img = gen
                                best_stats = stats
                                best_score = score
                                best_params = params
                        if best_img is None:
                            rejects[(province, template, bucket, 'spec_reject')] += 1
                            used_texts.discard(text)
                            continue

                        out_name = f'boarddump-{bucket}-{province}-{template}-{made:04d}-{text}.ppm'
                        rel_path = f'images/train/{bucket}/{province}/{out_name}'
                        abs_path = out_root / rel_path
                        write_ppm(abs_path, best_img)
                        manifest_rows.append(boarddump_manifest_row(abs_path, rel_path, text, args.dataset_name, args.source_name))
                        detail_rows.append({
                            'split': 'train',
                            'bucket': bucket,
                            'province': province,
                            'template_key': template,
                            'text': text,
                            'rel_path': rel_path,
                            'occ_ratio_seed': f'{float(occ):.6f}',
                            'mean': f"{best_stats['mean']:.6f}",
                            'left_minus_right': f"{best_stats['left_minus_right']:.6f}",
                            'border_dark_ratio': f"{best_stats['border_dark_ratio']:.6f}",
                            'left_edge': f"{best_stats['left_edge']:.6f}",
                            'mid_edge': f"{best_stats['mid_edge']:.6f}",
                            'occ_ratio': f"{best_stats['occ_ratio']:.6f}",
                            'score': f'{best_score:.6f}',
                            'params_json': json.dumps(best_params, ensure_ascii=False),
                            'exact_quad': json.dumps(np.asarray(exact_quad, dtype=np.float32).tolist(), ensure_ascii=False),
                            'board_quad': json.dumps(np.asarray(board_quad, dtype=np.float32).tolist(), ensure_ascii=False),
                            'appearance_mode': appearance_meta.get('appearance_mode', bucket),
                            'blur_strength': appearance_meta.get('blur_strength', 0.0),
                            'jpeg_quality': appearance_meta.get('jpeg_quality', 95),
                        })
                        split_texts['train'].add(text)
                        all_stats.append(best_stats)
                        if len(cards) < args.preview_limit:
                            cards.append(make_card(prepared, best_img, f'{text} {template}', best_stats, best_score))
                        made += 1
                        success = True
                    finally:
                        if not success:
                            pass
                if made < target:
                    raise RuntimeError(
                        f'Failed to reach target province={province} template={template} bucket={bucket}: made={made} target={target} attempts={attempts}'
                    )
                plan_rows.append({
                    'province': province,
                    'template_key': template,
                    'bucket': bucket,
                    'target': target,
                    'made': made,
                })

    manifest_path = manifests_dir / 'train_manifest_boarddump_exact_template.csv'
    manifest_fields = fieldnames_for_manifest_rows(manifest_rows)
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest_rows)

    details_path = details_dir / 'accepted.tsv'
    detail_fields = list(detail_rows[0].keys())
    with details_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=detail_fields, delimiter='\t')
        w.writeheader()
        w.writerows(detail_rows)

    plan_path = details_dir / 'generation_plan.tsv'
    with plan_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['province', 'template_key', 'bucket', 'target', 'made'], delimiter='\t')
        w.writeheader()
        w.writerows(plan_rows)

    contact_sheet_path = out_root / 'contact_sheet.png'
    write_contact_sheet(cards, contact_sheet_path, cols=3)

    report = {
        'total': len(manifest_rows),
        'split_counts': {'train': len(manifest_rows)},
        'bucket_counts': {
            f"train/{bucket}": count for bucket, count in Counter(r['bucket'] for r in detail_rows).items()
        },
        'split_text_overlap': {
            'train_val': 0,
            'train_test': 0,
            'val_test': 0,
        },
        'accepted_tsv': str(details_path),
        'train_manifest': str(manifest_path),
        'preview_paths': [str(contact_sheet_path)],
        'province_mode': args.province_mode,
        'province_targets': province_targets,
        'template_mode': args.template_mode,
        'tail_mode': args.tail_mode,
        'bucket_weights': bucket_weights,
        'template_counts': dict(Counter(r['template_key'] for r in detail_rows)),
        'template_bucket_counts': {
            f"{tpl}/{bucket}": c for (tpl, bucket), c in sorted(Counter((r['template_key'], r['bucket']) for r in detail_rows).items())
        },
        'province_template_counts': {
            p: dict(Counter(r['template_key'] for r in detail_rows if r['province'] == p)) for p in provinces
        },
        'rejects': {f'{p}/{t}/{b}/{kind}': c for (p, t, b, kind), c in sorted(rejects.items())},
        'aggregate': {},
    }
    if all_stats:
        for key in all_stats[0].keys():
            vals = np.asarray([x[key] for x in all_stats], dtype=np.float32)
            report['aggregate'][key] = {
                'mean': float(vals.mean()),
                'min': float(vals.min()),
                'max': float(vals.max()),
            }

    report_path = out_root / 'build_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
