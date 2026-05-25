#!/usr/bin/env python3
"""Build fixed evaluation subsets for front-end preprocess ablation study.

Samples N images from each CCPD2019 subset (base, tilt, weather, db, challenge)
with a fixed random seed. Outputs a CSV manifest per subset and a combined one.
"""

import os, sys, csv, random, argparse
from pathlib import Path

sys.path.insert(0, '/home/wzzz/LPRNet/src')
from load_data import CHARS

# CCPD label encoding: separate province list and ADS (letters + digits) list
CCPD_PROVINCES = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
]
CCPD_ADS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9',
]

CCPD_ROOT = '/home/wzzz/LPRNet/datasets/CCPD2019'
SUBSETS = ['ccpd_base', 'ccpd_tilt', 'ccpd_weather', 'ccpd_db', 'ccpd_challenge']
SUBSET_ALIASES = {
    'ccpd_base': 'Base',
    'ccpd_tilt': 'Tilt',
    'ccpd_weather': 'Weather',
    'ccpd_db': 'DB',
    'ccpd_challenge': 'Challenge',
}
# Subset sizes: Base is huge, Challenge is large, others smaller
SUBSET_SAMPLES = {
    'ccpd_base': 1000,
    'ccpd_tilt': 1000,
    'ccpd_weather': 1000,
    'ccpd_db': 1000,
    'ccpd_challenge': 1000,
}

def parse_label_from_ccpd_name(image_name):
    """Extract plate text from CCPD filename using CCPD's own PROVINCES+ADS encoding."""
    stem = os.path.splitext(os.path.basename(image_name))[0]
    parts = stem.split('-')
    if len(parts) < 5:
        return None
    # CCPD format: ...-chars-brightness-blur.jpg, chars field at index 4
    plate_codes_str = parts[4]
    try:
        plate_codes = [int(x) for x in plate_codes_str.split('_')]
    except ValueError:
        return None
    if len(plate_codes) < 1:
        return None
    # First index -> PROVINCES, rest -> ADS
    if not (0 <= plate_codes[0] < len(CCPD_PROVINCES)):
        return None
    label = CCPD_PROVINCES[plate_codes[0]]
    for code in plate_codes[1:]:
        if not (0 <= code < len(CCPD_ADS)):
            return None
        label += CCPD_ADS[code]
    return label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='eval_reports/front_preprocess_ablation')
    parser.add_argument('--seed', type=int, default=20260507)
    parser.add_argument('--samples_per_subset', type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for subset in SUBSETS:
        subset_dir = Path(CCPD_ROOT) / subset
        jpgs = sorted(subset_dir.glob('*.jpg'))
        print(f'{subset}: {len(jpgs)} total images')

        n = min(args.samples_per_subset, len(jpgs))
        sampled = random.sample(jpgs, n) if n < len(jpgs) else jpgs

        records = []
        for jpg in sampled:
            label = parse_label_from_ccpd_name(jpg.name)
            if label is None:
                continue
            records.append({
                'image_path': str(jpg.resolve()),
                'rel_path': f'datasets/CCPD2019/{subset}/{jpg.name}',
                'subset': subset,
                'subset_alias': SUBSET_ALIASES[subset],
                'label': label,
                'filename': jpg.name,
            })

        all_records.extend(records)
        print(f'  Sampled {len(records)} (valid labels)')

        # Write per-subset CSV
        csv_path = out_dir / f'eval_subset_{subset}.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['image_path','rel_path','subset','subset_alias','label','filename'])
            w.writeheader()
            for r in records:
                w.writerow({k: r[k] for k in ['image_path','rel_path','subset','subset_alias','label','filename']})
        print(f'  -> {csv_path}')

    # Write combined CSV
    combined_path = out_dir / 'eval_subset_all.csv'
    with open(combined_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_path','rel_path','subset','subset_alias','label','filename'])
        w.writeheader()
        for r in all_records:
            w.writerow({k: r[k] for k in ['image_path','rel_path','subset','subset_alias','label','filename']})
    print(f'\nCombined: {combined_path} ({len(all_records)} total)')

if __name__ == '__main__':
    main()
