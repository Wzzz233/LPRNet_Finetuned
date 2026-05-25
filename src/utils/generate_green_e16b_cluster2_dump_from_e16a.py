#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
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

TARGET_BUCKET_PLAN = [
    ('geometry_clean', 60),
    ('board_mid_occ', 45),
    ('board_low_occ', 30),
    ('board_extreme_tail', 15),
]


def load_dump_targets(csv_paths):
    merged = {}
    for path in csv_paths:
        with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt = (row.get('gt_text') or '').strip()
                if len(gt) != 8:
                    continue
                item = merged.setdefault(gt, {
                    'gt_text': gt,
                    'clusters': set(),
                    'failure_types': set(),
                    'sources': set(),
                    'rows': 0,
                })
                item['clusters'].add((row.get('cluster') or '').strip())
                item['failure_types'].add((row.get('failure_type') or '').strip())
                item['sources'].add(str(path))
                item['rows'] += 1
    targets = []
    for gt, item in sorted(merged.items()):
        targets.append({
            'gt_text': gt,
            'clusters': sorted(x for x in item['clusters'] if x),
            'failure_types': sorted(x for x in item['failure_types'] if x),
            'sources': sorted(item['sources']),
            'rows': item['rows'],
        })
    return targets


def main():
    ap = argparse.ArgumentParser(description='Generate E16B targeted dump-text append set on top of E16A manifest.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--dataset-name', default='green_e16b_nonanhui_ad_balance_12k_dump_v1')
    ap.add_argument('--source-name', default='e16b_nonanhui_ad_balance_12k_dump_v1')
    ap.add_argument('--repo-root', default='/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/repo_license_plate_generator')
    ap.add_argument('--lpr-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--seed', type=int, default=20260418)
    ap.add_argument('--smoke-count-per-text', type=int, default=0)
    ap.add_argument('--dump-csv', action='append', required=True)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    details_dir = out_root / 'details'
    manifests_dir = out_root / 'manifests'
    qa_dir = out_root / 'qa'
    for d in [out_root, details_dir, manifests_dir, qa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    targets = load_dump_targets(args.dump_csv)
    if not targets:
        raise RuntimeError('no valid 8-char gt_text found in dump csvs')

    chars_gen, augmenter = e16a.ensure_repo_imports(args.repo_root)
    augmenter = e16a.setup_augmenter(augmenter, e16a.GEOMETRY_CFG)
    canonical_quad = e16a.detect_canonical_plate_quad(augmenter.template_image)
    prepare_board = e16a.load_prepare_board(args.lpr_root)

    manifest_rows = []
    accepted_details = []
    cards = []

    per_text_count_override = None
    if args.smoke_count_per_text > 0:
        per_text_count_override = int(args.smoke_count_per_text)

    for text_idx, target in enumerate(targets):
        text = target['gt_text']
        bucket_plan = []
        if per_text_count_override is None:
            for bucket, count in TARGET_BUCKET_PLAN:
                bucket_plan.extend([bucket] * count)
        else:
            bucket_plan.extend(['geometry_clean'] * per_text_count_override)
        for item_idx, bucket in enumerate(bucket_plan):
            item_seed_base = args.seed + text_idx * 10_000_000 + item_idx * 1009
            accepted = None
            rejects = []
            for attempt in range(40):
                attempt_seed = item_seed_base + attempt * 101
                rng = random.Random(attempt_seed)
                np.random.seed(attempt_seed % (2**32 - 1))
                render = e16a.render_standard_exact_quad(text, chars_gen, augmenter, canonical_quad, prepare_board, rng)
                if render['occ_ratio'] < float(e16a.GEOMETRY_CFG['min_occ_ratio']):
                    rejects.append({'reason': 'occ_ratio', 'value': render['occ_ratio'], 'attempt': attempt})
                    continue
                if render['max_char_angle_error_deg'] > float(e16a.GEOMETRY_CFG['max_char_angle_error_deg']):
                    rejects.append({'reason': 'char_angle_error', 'value': render['max_char_angle_error_deg'], 'attempt': attempt})
                    continue
                sample_id = f'e16b-{text}-{bucket}-{item_idx:03d}'
                record = e16a.base.build_sample_record(
                    img=render['realized'],
                    split='train',
                    bucket=bucket,
                    province=text[0],
                    text=text,
                    exact_quad=render['quad'],
                    board_quad=render['quad'],
                    asym_mode='exact',
                    appearance_meta={
                        'blur_strength': 0.0,
                        'jpeg_quality': 95,
                        'appearance_mode': 'targeted_dump_text',
                        'clusters': '|'.join(target['clusters']),
                        'failure_types': '|'.join(target['failure_types']),
                    },
                    out_root=out_root,
                    uid=sample_id,
                    prepare_fn=prepare_board,
                    dataset_name=args.dataset_name,
                    source=args.source_name,
                )
                manifest_rows.append(record['manifest_row'])
                accepted_details.append({
                    'sample_id': sample_id,
                    'text': text,
                    'bucket': bucket,
                    'clusters': '|'.join(target['clusters']),
                    'failure_types': '|'.join(target['failure_types']),
                    'source_csvs': '|'.join(target['sources']),
                    'out_rel_path': record['rel_path'],
                    'out_abs_path': record['abs_path'],
                    'occ_ratio': round(float(record['occ_ratio']), 6),
                    'warped_aspect': round(float(record['warped_aspect']), 6),
                    'left_right_width_ratio': round(float(record['left_right_width_ratio']), 6),
                    'max_char_angle_error_deg': round(float(render['max_char_angle_error_deg']), 6),
                    'horizontal_sight_direction': render['horizontal_sight_direction'],
                    'vertical_sight_direction': render['vertical_sight_direction'],
                })
                if len(cards) < 32:
                    preview = cv2.resize(render['prepared94'], (94 * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(preview, f'{text} {bucket}', (4, preview.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    cards.append(preview)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(f'failed to generate text={text} bucket={bucket}; rejects={rejects[:5]}')

    manifest_local = manifests_dir / f'train_manifest_{args.dataset_name}.csv'
    with manifest_local.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=e16a.MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest_rows)

    with (details_dir / 'accepted.tsv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = list(accepted_details[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(accepted_details)

    base_count, merged_count = e16a.append_manifest(Path(args.base_manifest), manifest_rows, Path(args.out_manifest))
    validation = e16a.validate_generated_set(manifest_rows)

    if cards:
        cols = 4
        pad = 8
        cell_h = max(x.shape[0] for x in cards)
        cell_w = max(x.shape[1] for x in cards)
        rows_n = int(np.ceil(len(cards) / cols))
        sheet = np.full((rows_n * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3), 0, dtype=np.uint8)
        for i, img in enumerate(cards):
            r = i // cols
            c = i % cols
            y = pad + r * (cell_h + pad)
            x = pad + c * (cell_w + pad)
            sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.imwrite(str(qa_dir / 'contact_sheet.jpg'), sheet)

    summary = {
        'dataset_name': args.dataset_name,
        'source_name': args.source_name,
        'accepted_count': len(accepted_details),
        'base_manifest_count': base_count,
        'merged_manifest_count': merged_count,
        'target_texts': [t['gt_text'] for t in targets],
        'target_rows': {t['gt_text']: t['rows'] for t in targets},
        'bucket_counts': dict(sorted(Counter(x['bucket'] for x in accepted_details).items())),
        'text_counts': dict(sorted(Counter(x['text'] for x in accepted_details).items())),
        'validation': validation,
        'paths': {
            'out_dir': str(out_root),
            'manifest_local': str(manifest_local),
            'manifest_merged': str(args.out_manifest),
            'accepted_tsv': str(details_dir / 'accepted.tsv'),
            'contact_sheet': str(qa_dir / 'contact_sheet.jpg'),
        },
    }
    (details_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
