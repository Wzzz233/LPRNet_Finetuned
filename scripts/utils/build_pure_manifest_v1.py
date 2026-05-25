#!/usr/bin/env python3
"""Build a pure dataset v1 compliant manifest from verified data sources.

Usage:
    python build_pure_manifest_v1.py --out manifests/my_experiment.csv \
        --preproc none --green8_ratio 0.40
"""
import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

random.seed(20260423)

def load_existing_manifest(path, target_split, family_filter=None):
    """Load rows from an existing manifest, filtering by split and family."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split') != target_split:
                continue
            if family_filter and row.get('family') != family_filter:
                continue
            rows.append(row)
    return rows

def scan_image_directory(root_dir, family, difficulty_map, source_tag):
    """Scan a directory tree for images and generate manifest rows.
    
    difficulty_map: dict of {dirname_suffix: difficulty_bucket}
    e.g. {'simple': 'simple', 'hard': 'hard', 'extreme': 'extreme'}
    """
    rows = []
    root = Path(root_dir)
    if not root.exists():
        return rows
    for dirpath, dirnames, filenames in os.walk(root):
        bucket = None
        for suffix, diff in difficulty_map.items():
            if suffix in dirpath:
                bucket = diff
                break
        if bucket is None:
            continue
        for fname in filenames:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.ppm')):
                continue
            img_path = os.path.join(dirpath, fname)
            # Attempt to extract text from filename (format-dependent)
            text = ""
            parts = fname.replace('.jpg', '').split('-')
            # heuristic: look for Chinese province + alphanumeric pattern
            for p in parts:
                if len(p) >= 7 and any('\u4e00' <= c <= '\u9fff' for c in p):
                    text = p
                    break
            rows.append({
                'img_path': img_path,
                'text': text,
                'family': family,
                'difficulty_bucket': bucket,
                'split': 'train',
                'source': source_tag,
                'sample_weight': '1.0',
            })
    return rows

def apply_anhui_downweight(rows, real_factor=0.60, synth_factor=0.90):
    """Gently downweight Anhui samples while preserving sequence info."""
    for row in rows:
        text = row.get('text', '')
        if not text or text[0] != '皖':
            continue
        try:
            w = float(row.get('sample_weight', '1.0'))
        except:
            w = 1.0
        bucket = row.get('difficulty_bucket', '')
        if bucket == 'real':
            row['sample_weight'] = str(round(w * real_factor, 4))
        else:
            row['sample_weight'] = str(round(w * synth_factor, 4))
    return rows

def balance_provinces(rows, target_ratio_per_prov=0.04, max_clip=3.0):
    """Compute sample weights to balance provinces. 皖 is handled separately."""
    prov_counts = Counter()
    for row in rows:
        text = row.get('text', '')
        if text:
            prov_counts[text[0]] += 1
    
    if not prov_counts:
        return rows
    
    median_count = sorted(prov_counts.values())[len(prov_counts)//2]
    for row in rows:
        text = row.get('text', '')
        if not text:
            continue
        prov = text[0]
        if prov == '皖':
            continue  # Already handled
        count = prov_counts[prov]
        if count == 0:
            continue
        try:
            w = float(row.get('sample_weight', '1.0'))
        except:
            w = 1.0
        # Upweight minority provinces
        factor = min(median_count / count, max_clip)
        row['sample_weight'] = str(round(w * factor, 4))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='Output manifest path')
    ap.add_argument('--preproc', default='none', choices=['none', 'gray3'])
    ap.add_argument('--green8_ratio', type=float, default=0.40, help='Target green8 ratio in train')
    ap.add_argument('--anhui_real_factor', type=float, default=0.60)
    ap.add_argument('--anhui_synth_factor', type=float, default=0.90)
    ap.add_argument('--seed', type=int, default=20260423)
    args = ap.parse_args()
    
    random.seed(args.seed)
    
    # --- CONFIGURE YOUR DATA SOURCES HERE ---
    # This is a template; fill in actual paths and manifest references
    
    all_rows = []
    
    # Example: Load from existing reliable manifests
    # normal7_real = load_existing_manifest(
    #     '/home/wzzz/LPRNet/manifests/normal7_test_only_v1.csv',
    #     target_split='train', family_filter='normal7'
    # )
    # all_rows.extend(normal7_real)
    
    # Example: Scan synthetic directories
    # tier3_rows = scan_image_directory(
    #     '/home/wzzz/LPRNet/datasets/green_edgefit_tier3_full_v2/images/train',
    #     family='green8',
    #     difficulty_map={'simple': 'simple', 'hard': 'hard', 'extreme': 'extreme'},
    #     source_tag='edgefit_tier3_v2'
    # )
    # all_rows.extend(tier3_rows)
    
    # boardlike_rows = scan_image_directory(
    #     '/home/wzzz/LPRNet/datasets/green_edgefit_v4_boardlike_a3000/images/train',
    #     family='green8',
    #     difficulty_map={
    #         'geometry_clean': 'simple',
    #         'board_low_occ': 'hard',
    #         'board_mid_occ': 'hard',
    #         'board_extreme_tail': 'extreme'
    #     },
    #     source_tag='boardlike_v4'
    # )
    # all_rows.extend(boardlike_rows)
    
    if not all_rows:
        print("[WARN] No data sources configured. This is a template script.")
        print("       Edit the script to add your actual data sources.")
        return
    
    # Apply weights
    all_rows = apply_anhui_downweight(all_rows, args.anhui_real_factor, args.anhui_synth_factor)
    all_rows = balance_provinces(all_rows)
    
    # Write manifest
    fieldnames = ['img_path', 'text', 'family', 'difficulty_bucket', 'split', 'source', 'sample_weight']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    # Summary
    train_counts = Counter()
    for row in all_rows:
        if row.get('split') == 'train':
            train_counts[row.get('family')] += 1
            train_counts[row.get('difficulty_bucket')] += 1
    
    summary = {
        'preproc': args.preproc,
        'total_rows': len(all_rows),
        'train_counts': dict(train_counts),
        'anhui_real_factor': args.anhui_real_factor,
        'anhui_synth_factor': args.anhui_synth_factor,
    }
    
    summary_path = str(args.out).replace('.csv', '.summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] Manifest: {args.out}")
    print(f"[DONE] Summary:  {summary_path}")
    print(f"       Total rows: {len(all_rows)}")
    print(f"       Train family counts: {dict(train_counts)}")

if __name__ == '__main__':
    main()
