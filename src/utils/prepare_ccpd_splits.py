#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

PROVINCES = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
]

ADS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate LPRNet label txt files from CCPD split files.')
    parser.add_argument('--dataset_root', default='./CCPD2019', help='CCPD dataset root directory.')
    parser.add_argument('--split_dir', default=None, help='Directory that contains train.txt/val.txt/test.txt.')
    parser.add_argument('--output_dir', default='./prepared_labels/ccpd2019', help='Directory to write generated label txt files.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='Split names to generate.')
    return parser.parse_args()


def decode_ccpd_plate(rel_path: str) -> str:
    stem = Path(rel_path).stem
    parts = stem.split('-')
    if len(parts) < 5:
        raise ValueError(f'Unexpected CCPD filename format: {rel_path}')

    plate_codes = [int(item) for item in parts[4].split('_')]
    if len(plate_codes) != 7:
        raise ValueError(f'Unexpected plate code length in: {rel_path}')

    province_idx = plate_codes[0]
    if not 0 <= province_idx < len(PROVINCES):
        raise ValueError(f'Invalid province index {province_idx} in: {rel_path}')

    plate = [PROVINCES[province_idx]]
    for code in plate_codes[1:]:
        if not 0 <= code < len(ADS):
            raise ValueError(f'Invalid character index {code} in: {rel_path}')
        plate.append(ADS[code])
    return ''.join(plate)


def normalize_rel_path(rel_path: str) -> str:
    return Path(rel_path.strip().replace('\\', '/')).as_posix()


def generate_split(dataset_root: Path, split_file: Path, output_file: Path) -> tuple[int, int]:
    lines = split_file.read_text(encoding='utf-8').splitlines()
    valid_count = 0
    skipped_count = 0
    output_lines = []

    for raw_line in lines:
        rel_path = normalize_rel_path(raw_line)
        if not rel_path:
            continue
        img_path = dataset_root / rel_path
        if not img_path.exists():
            skipped_count += 1
            continue
        plate = decode_ccpd_plate(rel_path)
        output_lines.append(f'{rel_path} {plate}')
        valid_count += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('\n'.join(output_lines) + ('\n' if output_lines else ''), encoding='utf-8')
    return valid_count, skipped_count


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    split_dir = Path(args.split_dir) if args.split_dir else dataset_root / 'splits'
    output_dir = Path(args.output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset root not found: {dataset_root}')
    if not split_dir.exists():
        raise FileNotFoundError(f'Split dir not found: {split_dir}')

    for split_name in args.splits:
        split_file = split_dir / f'{split_name}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f'Split file not found: {split_file}')
        output_file = output_dir / f'{split_name}_labels.txt'
        valid_count, skipped_count = generate_split(dataset_root, split_file, output_file)
        print(f'[OK] {split_name}: wrote {valid_count} labels to {output_file} (missing files skipped: {skipped_count})')

    print(f'[Done] dataset_root={dataset_root.resolve()} output_dir={output_dir.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
