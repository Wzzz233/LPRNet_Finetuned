#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import generate_green_edgefit_v4_boardlike_equalprov as base
import generate_green_boarddump_exact_templates as boarddump
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

DEFAULT_BUCKET_PLAN = [
    ('geometry_clean', 6),
    ('board_mid_occ', 14),
    ('board_low_occ', 10),
]


def parse_bucket_plan(text):
    out = []
    total = 0
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = int(value.strip())
        if key not in {'geometry_clean', 'board_mid_occ', 'board_low_occ', 'board_extreme_tail'}:
            raise ValueError(f'unknown bucket: {key}')
        if value < 0:
            raise ValueError(f'negative bucket count: {part}')
        if value == 0:
            continue
        out.append((key, value))
        total += value
    if total <= 0:
        raise ValueError('bucket plan total must be > 0')
    return out


def append_manifest(base_manifest: Path, append_rows, out_manifest: Path):
    with base_manifest.open('r', encoding='utf-8', newline='') as f:
        base_rows = list(csv.DictReader(f))
    fieldnames = list(base_rows[0].keys())
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(base_rows)
        w.writerows(append_rows)
    return len(base_rows), len(base_rows) + len(append_rows)


def validate_generated_set(manifest_rows):
    bad = []
    for row in manifest_rows[: min(20, len(manifest_rows))]:
        path = Path(row['img_path'])
        img = cv2.imread(str(path))
        if img is None:
            bad.append({'img_path': str(path), 'reason': 'read_fail'})
            continue
        if img.ndim != 3:
            bad.append({'img_path': str(path), 'reason': f'bad_ndim:{img.ndim}'})
    summary = {'checked_rows': min(20, len(manifest_rows)), 'bad_count': len(bad), 'bad_examples': bad[:10]}
    if bad:
        raise RuntimeError(f'generated set validation failed: {summary}')
    return summary


def load_bank(bank_json: Path):
    data = json.loads(bank_json.read_text(encoding='utf-8'))
    provinces = data.get('provinces') or list(base.NON_ANHUI_PROVINCES)
    suffixes = data.get('suffixes') or []
    if not suffixes:
        raise RuntimeError(f'empty suffix bank: {bank_json}')
    return provinces, suffixes


def main():
    ap = argparse.ArgumentParser(description='Generate E17A cluster2 shared-suffix board_dump append set.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--bank-json', required=True)
    ap.add_argument('--repo-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--spec-json', default='/home/wzzz/LPRNet/tmp/green_board_domain_spec_v1.json')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--dataset-name', default='green_e17a_cluster2_suffixbank_900')
    ap.add_argument('--source-name', default='e17a_cluster2_suffixbank_900')
    ap.add_argument('--seed', type=int, default=20260417)
    ap.add_argument('--bucket-plan', default='geometry_clean=6,board_mid_occ=14,board_low_occ=10')
    ap.add_argument('--max-attempts', type=int, default=60)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    details_dir = out_root / 'details'
    manifests_dir = out_root / 'manifests'
    qa_dir = out_root / 'qa'
    for d in [out_root, details_dir, manifests_dir, qa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    provinces, suffixes = load_bank(Path(args.bank_json))
    bucket_plan = parse_bucket_plan(args.bucket_plan)
    expected = sum(v for _, v in bucket_plan)
    if len(suffixes) < expected:
        raise RuntimeError(f'suffix bank size mismatch: need at least {expected}, got {len(suffixes)}')
    suffixes = suffixes[:expected]

    qspec = load_qspec(Path(args.spec_json))
    chars_gen, augmenter, prepare_fn = base.ensure_repo_imports(args.repo_root)
    used_texts = base.load_used_texts([args.base_manifest])

    manifest_rows = []
    accepted_details = []
    cards = []

    bucket_for_suffix = []
    idx = 0
    for bucket, count in bucket_plan:
        for _ in range(count):
            bucket_for_suffix.append((suffixes[idx], bucket))
            idx += 1

    for prov_idx, province in enumerate(provinces):
        for item_idx, (suffix, bucket) in enumerate(bucket_for_suffix):
            text = province + suffix
            if text in used_texts:
                text = province + suffix[:-1] + str((item_idx + prov_idx) % 10)
            item_seed_base = args.seed + prov_idx * 10_000_000 + item_idx * 1009
            accepted = False
            rejects = []
            for attempt in range(args.max_attempts):
                attempt_seed = item_seed_base + attempt * 101
                rng = random.Random(attempt_seed)
                np_rng = np.random.default_rng(attempt_seed)
                exact_quad_seed = base.make_exact_quad(bucket, rng)
                plate_base = base.build_base_plate(text, chars_gen, augmenter)
                _, exact_quad, board_img_raw, board_quad, asym_mode = base.render_plate_with_exact_and_board(plate_base, exact_quad_seed, bucket, rng)
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
                    rejects.append({'reason': 'occ_reject', 'attempt': attempt, 'occ': float(occ)})
                    continue

                best_img = None
                best_stats = None
                best_score = None
                best_params = None
                best_within_spec = False
                for _ in range(100):
                    params = sample_params(np_rng)
                    gen = transform_dumplike_to_board_native(prepared, params)
                    stats = gray_stats(gen)
                    is_within = within_spec(stats, qspec)
                    score = spec_score(stats, qspec)
                    if not is_within:
                        score += 1000.0
                    if best_score is None or score < best_score:
                        best_img = gen
                        best_stats = stats
                        best_score = score
                        best_params = params
                        best_within_spec = is_within
                if best_img is None:
                    rejects.append({'reason': 'spec_reject', 'attempt': attempt})
                    continue

                sample_id = f'e17a-{province}-{bucket}-{item_idx:03d}-{suffix}'
                out_name = f'{sample_id}.ppm'
                rel_path = f'images/train/{bucket}/{province}/{out_name}'
                abs_path = out_root / rel_path
                boarddump.write_ppm(abs_path, best_img)
                row = boarddump.boarddump_manifest_row(abs_path, rel_path, text, args.dataset_name, args.source_name)
                manifest_rows.append(row)
                accepted_details.append({
                    'sample_id': sample_id,
                    'province': province,
                    'suffix': suffix,
                    'bucket': bucket,
                    'text': text,
                    'out_rel_path': rel_path,
                    'out_abs_path': str(abs_path),
                    'occ_ratio_seed': round(float(occ), 6),
                    'mean': round(float(best_stats['mean']), 6),
                    'left_minus_right': round(float(best_stats['left_minus_right']), 6),
                    'border_dark_ratio': round(float(best_stats['border_dark_ratio']), 6),
                    'left_edge': round(float(best_stats['left_edge']), 6),
                    'mid_edge': round(float(best_stats['mid_edge']), 6),
                    'occ_ratio': round(float(best_stats['occ_ratio']), 6),
                    'score': round(float(best_score), 6),
                    'params_json': json.dumps(best_params, ensure_ascii=False),
                    'asym_mode': asym_mode,
                    'appearance_mode': appearance_meta.get('appearance_mode', bucket),
                })
                if len(cards) < 48:
                    cards.append(make_card(prepared, best_img, f'{text} {bucket}', best_stats, best_score))
                used_texts.add(text)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(f'failed province={province} suffix={suffix} bucket={bucket}; rejects={rejects[:5]}')

    manifest_local = manifests_dir / f'train_manifest_{args.dataset_name}.csv'
    manifest_fields = boarddump.fieldnames_for_manifest_rows(manifest_rows)
    with manifest_local.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest_rows)

    with (details_dir / 'accepted.tsv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = list(accepted_details[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(accepted_details)

    base_count, merged_count = append_manifest(Path(args.base_manifest), manifest_rows, Path(args.out_manifest))
    validation = validate_generated_set(manifest_rows)
    write_contact_sheet(cards, qa_dir / 'contact_sheet.png', cols=3)

    summary = {
        'dataset_name': args.dataset_name,
        'source_name': args.source_name,
        'accepted_count': len(accepted_details),
        'base_manifest_count': base_count,
        'merged_manifest_count': merged_count,
        'target_provinces': provinces,
        'per_province_count': len(suffixes),
        'suffix_count': len(suffixes),
        'bucket_plan': {k: v for k, v in bucket_plan},
        'province_counts': dict(sorted(Counter(x['province'] for x in accepted_details).items())),
        'bucket_counts': dict(sorted(Counter(x['bucket'] for x in accepted_details).items())),
        'validation': validation,
        'paths': {
            'out_dir': str(out_root),
            'manifest_local': str(manifest_local),
            'manifest_merged': str(args.out_manifest),
            'accepted_tsv': str(details_dir / 'accepted.tsv'),
            'contact_sheet': str(qa_dir / 'contact_sheet.png'),
        },
    }
    (details_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
