#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import generate_green_e16a_nonanhui_ad_balance as e16a

ALLOWED_BUCKETS = ('geometry_clean', 'board_mid_occ', 'board_low_occ', 'board_extreme_tail')
DEFAULT_BUCKET_RATIOS = 'board_mid_occ=0.5,board_low_occ=0.35,board_extreme_tail=0.15'


def parse_bucket_ratios(text: str):
    ratios = {}
    total = 0.0
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = float(value.strip())
        if key not in ALLOWED_BUCKETS:
            raise ValueError(f'unknown bucket in --bucket-ratios: {key}')
        if value < 0:
            raise ValueError(f'negative bucket ratio in --bucket-ratios: {part}')
        if value == 0:
            continue
        ratios[key] = ratios.get(key, 0.0) + value
        total += value
    if total <= 0:
        raise ValueError('bucket ratios total must be > 0')
    ordered = [(k, ratios[k] / total) for k in ALLOWED_BUCKETS if k in ratios]
    return ordered


def allocate_bucket_counts(total_count: int, bucket_ratios):
    if total_count <= 0:
        raise ValueError(f'total_count must be > 0, got {total_count}')
    raw = []
    assigned = 0
    for bucket, ratio in bucket_ratios:
        exact = total_count * ratio
        base = int(math.floor(exact))
        raw.append([bucket, base, exact - base])
        assigned += base
    remain = total_count - assigned
    raw.sort(key=lambda x: (-x[2], x[0]))
    for i in range(remain):
        raw[i % len(raw)][1] += 1
    raw.sort(key=lambda x: ALLOWED_BUCKETS.index(x[0]))
    return [(bucket, count) for bucket, count, _ in raw if count > 0]


def load_bank(bank_json: Path):
    data = json.loads(bank_json.read_text(encoding='utf-8'))
    items = data.get('items') if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise RuntimeError(f'invalid bank json: {bank_json}')
    out = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(f'bank item #{idx} is not an object')
        text = str(item.get('text') or '').strip()
        count = int(item.get('count') or 0)
        group = str(item.get('group') or '').strip()
        note = str(item.get('note') or '').strip()
        if len(text) != 8:
            raise RuntimeError(f'bank item #{idx} invalid text length: {text!r}')
        if count <= 0:
            raise RuntimeError(f'bank item #{idx} invalid count: {count}')
        out.append({'text': text, 'count': count, 'group': group, 'note': note})
    return out


def main():
    ap = argparse.ArgumentParser(description='Generate E20A cluster2 province-contrast board-like append set on top of E12C manifest.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--bank-json', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--dataset-name', default='green_e20a_cluster2_beijing_prefix_contrast_1200')
    ap.add_argument('--source-name', default='e20a_cluster2_beijing_prefix_contrast_1200')
    ap.add_argument('--repo-root', default='/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/repo_license_plate_generator')
    ap.add_argument('--lpr-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--seed', type=int, default=20260417)
    ap.add_argument('--bucket-ratios', default=DEFAULT_BUCKET_RATIOS)
    ap.add_argument('--max-attempts', type=int, default=60)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    details_dir = out_root / 'details'
    manifests_dir = out_root / 'manifests'
    qa_dir = out_root / 'qa'
    for d in [out_root, details_dir, manifests_dir, qa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    bank_items = load_bank(Path(args.bank_json))
    bucket_ratios = parse_bucket_ratios(args.bucket_ratios)

    chars_gen, augmenter = e16a.ensure_repo_imports(args.repo_root)
    augmenter = e16a.setup_augmenter(augmenter, e16a.GEOMETRY_CFG)
    canonical_quad = e16a.detect_canonical_plate_quad(augmenter.template_image)
    prepare_board = e16a.load_prepare_board(args.lpr_root)

    manifest_rows = []
    accepted_details = []
    cards = []

    for text_idx, item in enumerate(bank_items):
        text = item['text']
        bucket_counts = allocate_bucket_counts(item['count'], bucket_ratios)
        bucket_plan = []
        for bucket, count in bucket_counts:
            bucket_plan.extend([bucket] * count)
        province = text[0]

        for item_idx, bucket in enumerate(bucket_plan):
            item_seed_base = args.seed + text_idx * 10_000_000 + item_idx * 1009
            accepted = False
            rejects = []
            for attempt in range(args.max_attempts):
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

                sample_id = f'e20a-{text}-{bucket}-{item_idx:03d}'
                record = e16a.base.build_sample_record(
                    img=render['realized'],
                    split='train',
                    bucket=bucket,
                    province=province,
                    text=text,
                    exact_quad=render['quad'],
                    board_quad=render['quad'],
                    asym_mode='exact',
                    appearance_meta={
                        'blur_strength': 0.0,
                        'jpeg_quality': 95,
                        'appearance_mode': 'cluster2_prefix_contrast',
                        'bank_group': item['group'],
                        'bank_note': item['note'],
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
                    'province': province,
                    'group': item['group'],
                    'note': item['note'],
                    'bucket': bucket,
                    'target_count_for_text': item['count'],
                    'out_rel_path': record['rel_path'],
                    'out_abs_path': record['abs_path'],
                    'occ_ratio': round(float(record['occ_ratio']), 6),
                    'warped_aspect': round(float(record['warped_aspect']), 6),
                    'left_right_width_ratio': round(float(record['left_right_width_ratio']), 6),
                    'max_char_angle_error_deg': round(float(render['max_char_angle_error_deg']), 6),
                    'horizontal_sight_direction': render['horizontal_sight_direction'],
                    'vertical_sight_direction': render['vertical_sight_direction'],
                })
                if len(cards) < 48:
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
        'bank_json': str(args.bank_json),
        'bank_item_count': len(bank_items),
        'bucket_ratios': {k: v for k, v in bucket_ratios},
        'province_counts': dict(sorted(Counter(x['province'] for x in accepted_details).items())),
        'bucket_counts': dict(sorted(Counter(x['bucket'] for x in accepted_details).items())),
        'text_counts': dict(sorted(Counter(x['text'] for x in accepted_details).items())),
        'group_counts': dict(sorted(Counter(x['group'] for x in accepted_details).items())),
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
