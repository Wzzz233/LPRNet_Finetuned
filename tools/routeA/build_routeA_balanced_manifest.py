#!/usr/bin/env python3
"""Phase 5A: Create per-province balanced first-char manifest from R50 data pool.

Balances at a target per-province count. Only uses the current R50 source pool
(real ccppd2020 + replace data), not old CBLPRD data.

Output:
  manifests_rebased/routeA_firstchar_r50_20260512/train_r50_bal31_v1.csv

Usage:
  python tools/routeA/build_routeA_balanced_manifest.py
"""
import csv, json, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'manifests_rebased/routeA_firstchar_r50_20260512'
OUT_DIR.mkdir(parents=True, exist_ok=True)

R50_TRAIN_CSV = OUT_DIR / 'train_r50_raw_v1.csv'
PROVINCES_ORDER = ['京', '津', '冀', '晋', '蒙', '辽', '吉', '黑',
                   '沪', '苏', '浙', '皖', '闽', '赣', '鲁', '豫',
                   '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云',
                   '藏', '陕', '甘', '青', '宁', '新', '渝']

def main():
    random.seed(20260512)
    print('=' * 60)
    print('Phase 5A: Build per-province balanced manifest from R50 pool')
    print('=' * 60)

    # Read all rows
    rows_by_province = defaultdict(list)
    source_counts = Counter()
    total_rows = 0
    fieldnames = None

    with open(R50_TRAIN_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            text = r.get('text', '').strip()
            if not text:
                continue
            prov = text[0]
            rows_by_province[prov].append(r)
            source_counts[(prov, r.get('source', '').strip())] += 1
            total_rows += 1

    print(f'\nTotal rows: {total_rows}')
    print(f'Provinces found: {len(rows_by_province)}')

    # Determine min count
    prov_counts = {p: len(rows_by_province[p]) for p in PROVINCES_ORDER if p in rows_by_province}
    min_count = min(prov_counts.values())
    max_count = max(prov_counts.values())
    print(f'Min per-province: {min_count} ({min(prov_counts, key=prov_counts.get)})')
    print(f'Max per-province: {max_count} ({max(prov_counts, key=prov_counts.get)})')

    # Target: balance at min_count (per-province equal)
    target = min_count
    print(f'\nBalancing at target={target} per province')

    balanced_rows = []
    prov_selected = {}
    for prov in PROVINCES_ORDER:
        pool = rows_by_province.get(prov, [])
        if len(pool) <= target:
            selected = pool[:]
        else:
            selected = random.sample(pool, target)
        balanced_rows.extend(selected)
        prov_selected[prov] = len(selected)

    random.shuffle(balanced_rows)
    print(f'Balanced total: {len(balanced_rows)} rows')

    # Write balanced manifest
    out_path = OUT_DIR / 'train_r50_bal31_v1.csv'
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(balanced_rows)
    print(f'Saved: {out_path}')

    # Summary
    summary = {
        'name': 'train_r50_bal31_v1',
        'total': len(balanced_rows),
        'province_count': len(prov_selected),
        'target_per_province': target,
        'province_counts': prov_selected,
        'source_pool': str(R50_TRAIN_CSV),
        'note': f'Balanced from original {total_rows} rows (min={min_count})',
    }
    sum_path = OUT_DIR / 'summary_bal31.json'
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Summary: {sum_path}')

    for prov, cnt in prov_selected.items():
        print(f'  {prov}: {cnt}')


if __name__ == '__main__':
    main()
