#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E9 exact-template green8 board-like generator
- reuse E2-compatible v4 boardlike geometry chain
- keep four final difficulty buckets from E2-style training
- inject legal backup-letter templates (A/B/C/E/G/H/J/K etc.)
- apply fixed board-like bright transform on final rendered crop
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import generate_green_edgefit_v4_boardlike_equalprov as base

DIFF_WEIGHTS = {
    'geometry_clean': 0.40,
    'board_mid_occ': 0.30,
    'board_low_occ': 0.20,
    'board_extreme_tail': 0.10,
}

PATTERN_WEIGHTS = {
    'AA0': 0.30,
    'AA1': 0.15,
    'AA2': 0.15,
    'AB0': 0.08,
    'AC0': 0.08,
    'AE0': 0.08,
    'AD0': 0.05,
    'AF0': 0.05,
    'AG0': 0.02,
    'AH0': 0.02,
    'AJ0': 0.01,
    'AK0': 0.01,
}

FIVE_PROVS = ['陕', '苏', '沪', '浙', '粤']
ALL_PROVS = list(base.ALL_PROVINCES)
DIGITS = list('0123456789')


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


def brighten_boardlike(img, bucket, rng):
    out = img.astype(np.float32)
    target_mean = {
        'geometry_clean': (150, 168),
        'board_mid_occ': (162, 178),
        'board_low_occ': (170, 186),
        'board_extreme_tail': (174, 190),
    }[bucket]
    gamma = {
        'geometry_clean': (0.92, 1.00),
        'board_mid_occ': (0.88, 0.96),
        'board_low_occ': (0.84, 0.92),
        'board_extreme_tail': (0.82, 0.90),
    }[bucket]
    grad_left = {
        'geometry_clean': (0.92, 0.98),
        'board_mid_occ': (0.88, 0.96),
        'board_low_occ': (0.84, 0.94),
        'board_extreme_tail': (0.82, 0.92),
    }[bucket]
    grad_right = {
        'geometry_clean': (1.00, 1.06),
        'board_mid_occ': (1.02, 1.08),
        'board_low_occ': (1.03, 1.10),
        'board_extreme_tail': (1.04, 1.12),
    }[bucket]

    g = rng.uniform(*gamma)
    out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), g)

    cur = float(out.mean())
    target = rng.uniform(*target_mean)
    out += (target - cur)

    h, w = out.shape[:2]
    left_gain = rng.uniform(*grad_left)
    right_gain = rng.uniform(*grad_right)
    grad = np.linspace(left_gain, right_gain, w, dtype=np.float32).reshape(1, w, 1)
    out *= grad

    alpha = rng.uniform(0.92, 0.99)
    center = 160.0
    out = (out - center) * alpha + center

    if rng.random() < 0.55:
        sigma = rng.uniform(0.6, 1.8)
        noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
        out += noise

    return np.clip(out, 0, 255).astype(np.uint8)


def make_text_from_pattern(province, pattern, used_texts, rng):
    city = 'A'
    serial1 = pattern[1]
    serial2 = pattern[2]
    while True:
        tail = ''.join(rng.choice(DIGITS) for _ in range(4))
        text = province + city + serial1 + serial2 + tail
        if text not in used_texts:
            used_texts.add(text)
            return text


def generate_rows(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'images').mkdir(exist_ok=True)
    (out_root / 'manifests').mkdir(exist_ok=True)
    (out_root / 'details').mkdir(exist_ok=True)

    used_texts = base.load_used_texts(args.avoid_text_files)
    chars_gen, augmenter, prepare_fn = base.ensure_repo_imports(args.repo_root)

    provinces = FIVE_PROVS if args.province_mode == 'five' else ALL_PROVS
    province_targets = distribute_total_even(args.total_count, provinces)
    template_keys = list(PATTERN_WEIGHTS.keys())
    bucket_keys = list(DIFF_WEIGHTS.keys())

    rows = []
    split_texts = defaultdict(set)
    rejects = Counter()
    plan_rows = []

    for province in provinces:
        template_targets = apportion(province_targets[province], PATTERN_WEIGHTS, template_keys)
        for template in template_keys:
            bucket_targets = apportion(template_targets[template], DIFF_WEIGHTS, bucket_keys)
            for bucket in bucket_keys:
                target = bucket_targets[bucket]
                if target <= 0:
                    continue
                made = 0
                attempts = 0
                max_attempts = base.attempt_budget_for_bucket(bucket, target)
                while made < target and attempts < max_attempts:
                    attempts += 1
                    text = make_text_from_pattern(province, template, used_texts, rng)
                    plate_base = base.build_base_plate(text, chars_gen, augmenter)
                    exact_quad_seed = base.make_exact_quad(bucket, rng)
                    _, exact_quad, board_img_raw, board_quad, asym_mode = base.render_plate_with_exact_and_board(
                        plate_base, exact_quad_seed, bucket, rng
                    )
                    board_img, appearance_meta = base.apply_appearance_by_bucket(board_img_raw, bucket, rng)
                    board_img = brighten_boardlike(board_img, bucket, rng)
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
                    if not base.accept_bucket(bucket, float(occ)):
                        rejects[(bucket, province, template)] += 1
                        used_texts.discard(text)
                        continue
                    uid = f'train-{bucket}-{province}-{template}-{made:04d}-{text}'
                    record = base.build_sample_record(
                        img=board_img,
                        split='train',
                        bucket=bucket,
                        province=province,
                        text=text,
                        exact_quad=exact_quad,
                        board_quad=board_quad,
                        asym_mode=asym_mode,
                        appearance_meta=appearance_meta,
                        out_root=out_root,
                        uid=uid,
                        prepare_fn=prepare_fn,
                        dataset_name=args.dataset_name,
                        source=args.source_name,
                    )
                    record['template_key'] = template
                    rows.append(record)
                    split_texts['train'].add(text)
                    made += 1
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
    return rows, split_texts, rejects, provinces, province_targets, plan_rows, prepare_fn


def write_outputs(rows, split_texts, rejects, provinces, province_targets, plan_rows, out_dir, prepare_fn, preview_per_bucket):
    sanitized_rows = []
    for r in rows:
        rr = dict(r)
        rr.pop('template_key', None)
        sanitized_rows.append(rr)
    report = base.write_outputs(sanitized_rows, split_texts, out_dir, prepare_fn=prepare_fn, preview_per_bucket=preview_per_bucket)
    out_root = Path(out_dir)

    # enrich accepted.tsv with template key
    tsv_path = out_root / 'details' / 'accepted.tsv'
    # rewrite with template_key included
    fieldnames = [
        'split', 'bucket', 'province', 'template_key', 'text', 'rel_path', 'exact_quad', 'board_quad', 'quad_mode',
        'occ_ratio', 'warped_w', 'warped_h', 'warped_aspect', 'left_right_width_ratio',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'asym_mode',
        'blur_strength', 'jpeg_quality', 'appearance_mode', 'source_tag'
    ]
    import csv
    with tsv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr['exact_quad'] = json.dumps(rr['exact_quad'], ensure_ascii=False)
            rr['board_quad'] = json.dumps(rr['board_quad'], ensure_ascii=False)
            rr.pop('abs_path', None)
            rr.pop('manifest_row', None)
            w.writerow({k: rr.get(k) for k in fieldnames})

    import csv
    plan_path = out_root / 'details' / 'generation_plan.tsv'
    with plan_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['province', 'template_key', 'bucket', 'target', 'made'], delimiter='\t')
        w.writeheader()
        w.writerows(plan_rows)

    report['province_mode'] = 'five' if provinces == FIVE_PROVS else 'all'
    report['province_targets'] = province_targets
    report['template_counts'] = dict(Counter(r['template_key'] for r in rows))
    report['template_bucket_counts'] = {
        f'{tpl}/{bucket}': c for (tpl, bucket), c in sorted(Counter((r['template_key'], r['bucket']) for r in rows).items())
    }
    report['province_template_counts'] = {
        p: dict(Counter(r['template_key'] for r in rows if r['province'] == p)) for p in provinces
    }
    report['rejects'] = {f'{b}/{p}/{t}': c for (b, p, t), c in sorted(rejects.items())}
    report['generation_plan_tsv'] = str(plan_path)
    (out_root / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--province_mode', choices=['five', 'all'], required=True)
    ap.add_argument('--total_count', type=int, required=True)
    ap.add_argument('--seed', type=int, default=20260416)
    ap.add_argument('--dataset_name', default='green_edgefit_v4_boardlike_exact_template_v1')
    ap.add_argument('--source_name', default='v4_boardlike_exact_template_v1')
    ap.add_argument('--preview_per_bucket', type=int, default=2)
    ap.add_argument('--avoid_text_files', nargs='*', default=[])
    return ap.parse_args()


def main():
    args = parse_args()
    rows, split_texts, rejects, provinces, province_targets, plan_rows, prepare_fn = generate_rows(args)
    write_outputs(rows, split_texts, rejects, provinces, province_targets, plan_rows, args.out_dir, prepare_fn, args.preview_per_bucket)


if __name__ == '__main__':
    main()
