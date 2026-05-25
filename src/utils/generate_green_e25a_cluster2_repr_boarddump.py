#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
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
from generate_green_board_native_preview_v1 import gray_stats, sample_params, transform_dumplike_to_board_native
from generate_green_boarddump_exact_templates import boarddump_manifest_row, fieldnames_for_manifest_rows

ALLOWED_APPEARANCE = {'prepared_raw', 'board_native', 'mixed'}


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


def save_contact_sheet(cards, out_path: Path, cols: int = 4, pad: int = 8):
    if not cards:
        return
    cell_h = max(img.shape[0] for img in cards)
    cell_w = max(img.shape[1] for img in cards)
    rows_n = int(np.ceil(len(cards) / cols))
    sheet = np.full((rows_n * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3), 0, dtype=np.uint8)
    for i, img in enumerate(cards):
        r = i // cols
        c = i % cols
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.imwrite(str(out_path), sheet)


def main():
    ap = argparse.ArgumentParser(description='Generate E25A cluster2 representation-rebuild data directly in board_dump form.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--bank-json', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--dataset-name', default='green_e25a_cluster2_repr_boarddump_6000')
    ap.add_argument('--source-name', default='e25a_cluster2_repr_boarddump_6000')
    ap.add_argument('--repo-root', default='/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/repo_license_plate_generator')
    ap.add_argument('--lpr-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--seed', type=int, default=20260419)
    ap.add_argument('--max-attempts', type=int, default=60)
    ap.add_argument('--appearance-mode', choices=sorted(ALLOWED_APPEARANCE), default='board_native')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_dir)
    images_dir = out_root / 'images' / 'train'
    manifests_dir = out_root / 'manifests'
    details_dir = out_root / 'details'
    qa_dir = out_root / 'qa'
    for d in [images_dir, manifests_dir, details_dir, qa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    bank_items = load_bank(Path(args.bank_json))
    chars_gen, augmenter = e16a.ensure_repo_imports(args.repo_root)
    augmenter = e16a.setup_augmenter(augmenter, e16a.GEOMETRY_CFG)
    canonical_quad = e16a.detect_canonical_plate_quad(augmenter.template_image)
    prepare_board = e16a.load_prepare_board(args.lpr_root)

    manifest_rows = []
    accepted_details = []
    cards = []
    stats_acc = []

    for text_idx, item in enumerate(bank_items):
        text = item['text']
        province = text[0]
        for item_idx in range(item['count']):
            item_seed_base = args.seed + text_idx * 10_000_000 + item_idx * 1009
            accepted = False
            rejects = []
            for attempt in range(args.max_attempts):
                attempt_seed = item_seed_base + attempt * 101
                attempt_rng = random.Random(attempt_seed)
                np.random.seed(attempt_seed % (2**32 - 1))
                render = e16a.render_standard_exact_quad(text, chars_gen, augmenter, canonical_quad, prepare_board, attempt_rng)
                if render['occ_ratio'] < float(e16a.GEOMETRY_CFG['min_occ_ratio']):
                    rejects.append({'reason': 'occ_ratio', 'value': render['occ_ratio'], 'attempt': attempt})
                    continue
                if render['max_char_angle_error_deg'] > float(e16a.GEOMETRY_CFG['max_char_angle_error_deg']):
                    rejects.append({'reason': 'char_angle_error', 'value': render['max_char_angle_error_deg'], 'attempt': attempt})
                    continue

                prepared94 = render['prepared94']
                local_np_rng = np.random.default_rng(attempt_seed)
                if args.appearance_mode == 'prepared_raw':
                    final_img = prepared94
                    appearance_tag = 'prepared_raw'
                elif args.appearance_mode == 'board_native':
                    final_img = transform_dumplike_to_board_native(prepared94, sample_params(local_np_rng))
                    appearance_tag = 'board_native'
                else:
                    if local_np_rng.random() < 0.2:
                        final_img = prepared94
                        appearance_tag = 'prepared_raw'
                    else:
                        final_img = transform_dumplike_to_board_native(prepared94, sample_params(local_np_rng))
                        appearance_tag = 'board_native'
                final_img = np.clip(final_img, 0, 255).astype(np.uint8)
                stats = gray_stats(final_img)

                sample_id = f'e25a-{text}-{appearance_tag}-{item_idx:04d}'
                rel_path = f'images/train/p{ord(province):05d}/{sample_id}.ppm'
                abs_path = out_root / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(abs_path), final_img):
                    raise RuntimeError(f'failed to write image: {abs_path}')

                manifest_rows.append(boarddump_manifest_row(abs_path, rel_path, text, args.dataset_name, args.source_name))
                accepted_details.append({
                    'sample_id': sample_id,
                    'text': text,
                    'province': province,
                    'group': item['group'],
                    'note': item['note'],
                    'appearance_tag': appearance_tag,
                    'target_count_for_text': item['count'],
                    'out_rel_path': rel_path,
                    'out_abs_path': str(abs_path),
                    'occ_ratio': round(float(stats['occ_ratio']), 6),
                    'mean': round(float(stats['mean']), 6),
                    'border_dark_ratio': round(float(stats['border_dark_ratio']), 6),
                    'left_minus_right': round(float(stats['left_minus_right']), 6),
                    'left_edge': round(float(stats['left_edge']), 6),
                    'mid_edge': round(float(stats['mid_edge']), 6),
                    'max_char_angle_error_deg': round(float(render['max_char_angle_error_deg']), 6),
                })
                stats_acc.append(stats)
                if len(cards) < 48:
                    preview = cv2.resize(final_img, (94 * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(preview, f'{text} {appearance_tag}', (4, preview.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    cards.append(preview)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(f'failed to generate text={text}; rejects={rejects[:5]}')

    manifest_local = manifests_dir / f'train_manifest_{args.dataset_name}.csv'
    with manifest_local.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_for_manifest_rows(manifest_rows))
        w.writeheader()
        w.writerows(manifest_rows)

    with (details_dir / 'accepted.tsv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = list(accepted_details[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(accepted_details)

    base_count, merged_count = e16a.append_manifest(Path(args.base_manifest), manifest_rows, Path(args.out_manifest))
    save_contact_sheet(cards, qa_dir / 'contact_sheet.jpg')

    def q(values, key, qv):
        arr = np.array([float(x[key]) for x in values], dtype=np.float32)
        return round(float(np.quantile(arr, qv)), 6)

    summary = {
        'dataset_name': args.dataset_name,
        'source_name': args.source_name,
        'accepted_count': len(accepted_details),
        'base_manifest_count': base_count,
        'merged_manifest_count': merged_count,
        'bank_json': str(args.bank_json),
        'bank_item_count': len(bank_items),
        'appearance_mode': args.appearance_mode,
        'province_counts': dict(sorted(Counter(x['province'] for x in accepted_details).items())),
        'group_counts': dict(sorted(Counter(x['group'] for x in accepted_details).items())),
        'text_counts': dict(sorted(Counter(x['text'] for x in accepted_details).items())),
        'appearance_counts': dict(sorted(Counter(x['appearance_tag'] for x in accepted_details).items())),
        'gray_stats_quantiles': {
            key: {'q10': q(stats_acc, key, 0.1), 'q50': q(stats_acc, key, 0.5), 'q90': q(stats_acc, key, 0.9)}
            for key in ['occ_ratio', 'mean', 'border_dark_ratio', 'left_minus_right', 'left_edge', 'mid_edge']
        },
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
