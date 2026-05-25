#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_ccpd_splits import decode_ccpd_plate
from data.load_data import parse_ccpd_bbox_from_name


ALL_PROVINCES = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
]


def normalize_rel_path(rel_path: str) -> str:
    return Path(rel_path.strip().replace('\\', '/')).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare CCPD pseudo-anchor train/val label files from samples outside existing splits.')
    parser.add_argument('--dataset_root', default='./CCPD2019')
    parser.add_argument('--existing_txts', nargs='+', default=[
        './prepared_labels/ccpd2019/train_labels.txt',
        './prepared_labels/ccpd2019/val_labels.txt',
        './prepared_labels/ccpd2019/test_labels.txt',
    ])
    parser.add_argument('--output_dir', default='./prepared_labels/ccpd2019')
    parser.add_argument('--train_output_name', default='pseudo_anchor_train_labels.txt')
    parser.add_argument('--val_output_name', default='pseudo_anchor_val_labels.txt')
    parser.add_argument('--stats_output_name', default='pseudo_anchor_stats.json')
    parser.add_argument('--focused_provinces', default='粤,晋,黑,苏,浙')
    parser.add_argument('--focused_train_cap', type=int, default=40)
    parser.add_argument('--focused_val_cap', type=int, default=10)
    parser.add_argument('--other_train_cap', type=int, default=20)
    parser.add_argument('--other_val_cap', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260316)
    return parser.parse_args()


def read_used_paths(txt_files):
    used = set()
    for txt_file in txt_files:
        path = Path(txt_file)
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                used.add(normalize_rel_path(line.split(maxsplit=1)[0]))
    return used


def collect_candidates(dataset_root: Path, used_paths):
    groups = defaultdict(list)
    invalid_decode = 0
    invalid_bbox = 0
    skipped_used = 0
    for path in sorted(dataset_root.rglob('*')):
        if not path.is_file():
            continue
        rel_path = path.relative_to(dataset_root).as_posix()
        if rel_path in used_paths:
            skipped_used += 1
            continue
        try:
            plate = decode_ccpd_plate(rel_path)
        except Exception:
            invalid_decode += 1
            continue
        if parse_ccpd_bbox_from_name(rel_path) is None:
            invalid_bbox += 1
            continue
        groups[plate[0]].append((rel_path, plate))
    return groups, {
        'used_paths_skipped': skipped_used,
        'invalid_decode_skipped': invalid_decode,
        'invalid_bbox_skipped': invalid_bbox,
    }


def choose_split_counts(total_available, train_cap, val_cap):
    if total_available <= 0:
        return 0, 0
    if total_available == 1:
        return 1, 0
    val_count = min(val_cap, max(1, int(round(total_available * 0.2))))
    val_count = min(val_count, total_available - 1)
    train_count = min(train_cap, total_available - val_count)
    if train_count <= 0:
        train_count = 1
        val_count = max(0, total_available - train_count)
    return train_count, val_count


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    focused = {item.strip() for item in args.focused_provinces.split(',') if item.strip()}
    rng = random.Random(args.seed)

    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    used_paths = read_used_paths(args.existing_txts)
    groups, skip_stats = collect_candidates(dataset_root, used_paths)

    train_rows = []
    val_rows = []
    province_stats = {}

    for province in ALL_PROVINCES:
        rows = list(groups.get(province, []))
        rng.shuffle(rows)
        if province in focused:
            train_cap = args.focused_train_cap
            val_cap = args.focused_val_cap
        else:
            train_cap = args.other_train_cap
            val_cap = args.other_val_cap

        train_count, val_count = choose_split_counts(len(rows), train_cap, val_cap)
        chosen_train = rows[:train_count]
        chosen_val = rows[train_count:train_count + val_count]
        train_rows.extend(chosen_train)
        val_rows.extend(chosen_val)

        province_stats[province] = {
            'available': len(rows),
            'focused': province in focused,
            'train_target_cap': train_cap,
            'val_target_cap': val_cap,
            'train_selected': len(chosen_train),
            'val_selected': len(chosen_val),
            'train_shortfall': max(0, train_cap - len(chosen_train)),
            'val_shortfall': max(0, val_cap - len(chosen_val)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / args.train_output_name
    val_path = output_dir / args.val_output_name
    stats_path = output_dir / args.stats_output_name

    train_path.write_text(''.join(f'{rel} {plate}\n' for rel, plate in train_rows), encoding='utf-8')
    val_path.write_text(''.join(f'{rel} {plate}\n' for rel, plate in val_rows), encoding='utf-8')

    train_set = {rel for rel, _ in train_rows}
    val_set = {rel for rel, _ in val_rows}
    report = {
        'dataset_root': str(dataset_root.resolve()),
        'existing_txts': [str(Path(p)) for p in args.existing_txts],
        'focused_provinces': sorted(focused),
        'seed': args.seed,
        'skip_stats': skip_stats,
        'overlap': {
            'train_val': len(train_set & val_set),
            'train_existing': len(train_set & used_paths),
            'val_existing': len(val_set & used_paths),
        },
        'train': {
            'sample_count': len(train_rows),
            'province_distribution': dict(sorted(Counter(plate[0] for _, plate in train_rows).items())),
        },
        'val': {
            'sample_count': len(val_rows),
            'province_distribution': dict(sorted(Counter(plate[0] for _, plate in val_rows).items())),
        },
        'province_stats': province_stats,
    }
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'[OK] train labels: {train_path}')
    print(f'[OK] val labels: {val_path}')
    print(f'[OK] stats: {stats_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
