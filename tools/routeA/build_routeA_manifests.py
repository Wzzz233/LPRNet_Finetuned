#!/usr/bin/env python3
"""Route A Phase 1: Build first-char manifests from R50 data pipeline.

Outputs (under manifests_rebased/routeA_firstchar_r50_20260512/):
  train_r50_raw_v1.csv      - R50 training data, first-char labeled
  test_real_holdout_v1.csv  - CCPD2020 green val/test as holdout
  test_province_stress_v1.csv - province-stress pose val set
  summary.json              - per-file statistics

Usage:
  python tools/routeA/build_routeA_manifests.py
"""
import csv, json, sys, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'manifests_rebased/routeA_firstchar_r50_20260512'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
R50_TRAIN_CSV = ROOT / 'manifests_rebased/a_ratio_sweep_20260510/train_A_ratio_r50.csv'
CCPD2020_GREEN_REAL = ROOT / 'manifests_rebased/ccpd2020_green_real_20260509/train_ccpd2020_green_real.csv'
CURRICULUM_GREEN_VAL = ROOT / 'manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv'
PROVINCE_STRESS_CSV = ROOT / 'manifests_rebased/province_stress_pose_val_v1/province_stress_pose_val_v1.csv'

FIRST_CHAR_PROVINCES = set('京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新渝')

def load_csv_rows(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    return rows, fieldnames

def compute_summary(rows, name):
    total = len(rows)
    province_counts = Counter()
    source_counts = Counter()
    is_real = 0
    is_replace = 0
    text_counts = Counter()
    missing_paths = 0
    missing_cols = set()

    for r in rows:
        text = (r.get('text') or '').strip()
        src = (r.get('source') or '').strip()
        img_path = (r.get('img_path') or '').strip()
        family = (r.get('family') or '').strip()

        if text:
            province_counts[text[0]] += 1
            text_counts[text] += 1
        if src:
            source_counts[src] += 1
        if 'ccpd2020' in src.lower() or 'real' in src.lower():
            is_real += 1
        else:
            is_replace += 1
        if img_path and not Path(img_path).exists():
            missing_paths += 1

    # Check required columns
    required = ['img_path', 'text', 'family', 'source']
    for col in required:
        if col not in rows[0] if rows else []:
            missing_cols.add(col)

    result = {
        'name': name,
        'total': total,
        'province_count': len(province_counts),
        'province_counts': dict(province_counts.most_common()),
        'source_top10': dict(source_counts.most_common(10)),
        'real_est': is_real,
        'replace_est': is_replace,
        'unique_texts': len(text_counts),
        'missing_paths': missing_paths,
        'missing_columns': sorted(missing_cols),
        'families': dict(Counter(r.get('family','').strip() for r in rows).most_common()),
        'splits': dict(Counter(r.get('split','').strip() for r in rows).most_common()),
    }
    return result

def write_manifest(rows, fieldnames, path, split_val=None):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if split_val and r.get('split','').strip() != split_val:
                if split_val == 'train' and r.get('split','').strip() in ('val', 'test'):
                    continue
                elif split_val == 'test' and r.get('split','').strip() not in ('val', 'test'):
                    continue
            writer.writerow(r)
    return len(rows)


def main():
    print('=' * 60)
    print('Route A Phase 1: Build first-char manifests from R50 pipeline')
    print('=' * 60)

    all_summaries = {}

    # --- 1. Train: R50 raw data ---
    print('\n[1] Loading R50 train manifest...')
    r50_rows, r50_fields = load_csv_rows(R50_TRAIN_CSV)
    print(f'  R50 train total: {len(r50_rows)} rows')

    # Filter to green8 family only (should already be all green8)
    green_rows = [r for r in r50_rows if r.get('family','').strip() == 'green8']
    print(f'  green8 filtered: {len(green_rows)} rows')

    # Write train_r50_raw_v1
    train_path = OUT_DIR / 'train_r50_raw_v1.csv'
    write_manifest(green_rows, r50_fields, train_path, split_val=None)
    print(f'  Saved: {train_path}')
    all_summaries['train_r50_raw_v1'] = compute_summary(green_rows, 'train_r50_raw_v1')

    # --- 2. Test real holdout: use CCPD2020 green val ---
    print('\n[2] Building real holdout test set...')
    val_rows, val_fields = load_csv_rows(CURRICULUM_GREEN_VAL)
    print(f'  Curriculum green val total: {len(val_rows)} rows')

    # Filter to only test/val split, keep green8
    holdout_rows = [r for r in val_rows if r.get('family','').strip() == 'green8']
    print(f'  green8 filtered: {len(holdout_rows)} rows')

    # Also need to add training_manifest field for consistency
    # Check whether val_fields has all needed columns
    need_cols = set(r50_fields) - set(val_fields)
    if need_cols:
        print(f'  Adding missing columns: {need_cols}')
        for r in holdout_rows:
            for c in need_cols:
                r[c] = ''
        holdout_fields = r50_fields
    else:
        holdout_fields = val_fields

    # Reorder to match r50_fields
    holdout_out = []
    for r in holdout_rows:
        new_r = {k: r.get(k, '') for k in r50_fields}
        holdout_out.append(new_r)

    test_path = OUT_DIR / 'test_real_holdout_v1.csv'
    with open(test_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=r50_fields)
        writer.writeheader()
        writer.writerows(holdout_out)
    print(f'  Saved: {test_path}')
    all_summaries['test_real_holdout_v1'] = compute_summary(holdout_out, 'test_real_holdout_v1')

    # --- 3. Province stress test set ---
    print('\n[3] Building province stress test set...')
    stress_rows, stress_fields = load_csv_rows(PROVINCE_STRESS_CSV)
    print(f'  Province stress total: {len(stress_rows)} rows')

    # Align columns to r50_fields
    stress_out = []
    for r in stress_rows:
        new_r = {k: r.get(k, '') for k in r50_fields}
        stress_out.append(new_r)

    stress_path = OUT_DIR / 'test_province_stress_v1.csv'
    with open(stress_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=r50_fields)
        writer.writeheader()
        writer.writerows(stress_out)
    print(f'  Saved: {stress_path}')
    all_summaries['test_province_stress_v1'] = compute_summary(stress_out, 'test_province_stress_v1')

    # --- Save summary ---
    summary_path = OUT_DIR / 'summary.json'
    final_summary = {
        'generated': '2026-05-12',
        'route': 'A_firstchar_r50',
        'base_manifest': str(R50_TRAIN_CSV),
        'manifests': all_summaries,
    }
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\n  Summary saved: {summary_path}')

    # Print key stats
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for name, s in all_summaries.items():
        print(f'\n  {name}:')
        print(f'    total: {s["total"]}')
        print(f'    provinces: {s["province_count"]}')
        top_prov = list(s['province_counts'].items())[:5]
        print(f'    top provinces: {top_prov}')
        print(f'    real/replace: {s["real_est"]}/{s["replace_est"]}')
        print(f'    unique texts: {s["unique_texts"]}')
        print(f'    missing paths: {s["missing_paths"]}')
        if s['missing_columns']:
            print(f'    WARNING missing columns: {s["missing_columns"]}')

    print('\nDone.')


if __name__ == '__main__':
    main()
