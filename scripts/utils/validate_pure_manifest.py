#!/usr/bin/env python3
"""Validate manifest against pure dataset v1 rules."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

def validate(manifest_path, expected_preproc, expected_families=None):
    errors = []
    warnings = []
    stats = {
        'total_rows': 0,
        'train_rows': 0,
        'test_rows': 0,
        'family_counts': Counter(),
        'difficulty_counts': Counter(),
        'province_counts': Counter(),
        'missing_img': 0,
    }
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required_cols = {'img_path', 'text', 'family', 'difficulty_bucket', 'split'}
        if not required_cols.issubset(reader.fieldnames or []):
            missing = required_cols - set(reader.fieldnames or [])
            errors.append(f"Missing required columns: {missing}")
            return errors, warnings, stats
        
        for row in reader:
            stats['total_rows'] += 1
            split = row.get('split', '')
            if split == 'train':
                stats['train_rows'] += 1
            elif split == 'test':
                stats['test_rows'] += 1
            
            family = row.get('family', '')
            stats['family_counts'][family] += 1
            
            diff = row.get('difficulty_bucket', '')
            stats['difficulty_counts'][diff] += 1
            
            text = row.get('text', '')
            if text:
                stats['province_counts'][text[0]] += 1
            
            img_path = row.get('img_path', '')
            if img_path and not Path(img_path).exists():
                stats['missing_img'] += 1
            
            # Check family consistency
            if expected_families and family not in expected_families:
                errors.append(f"Row {stats['total_rows']}: family='{family}' not in expected {expected_families}")
            
            # Check green8 length
            if family == 'green8' and len(text) != 8:
                warnings.append(f"Row {stats['total_rows']}: green8 text length != 8: '{text}'")
            
            # Check normal7 length
            if family == 'normal7' and len(text) != 7:
                warnings.append(f"Row {stats['total_rows']}: normal7 text length != 7: '{text}'")
    
    # Anhui dominance check
    if '皖' in stats['province_counts']:
        anhui_ratio = stats['province_counts']['皖'] / max(stats['train_rows'], 1)
        if anhui_ratio > 0.5:
            warnings.append(f"Anhui dominates train set: {anhui_ratio:.1%}. Consider down-weighting.")
    
    # Extreme check
    total_train = stats['train_rows']
    extreme_count = stats['difficulty_counts'].get('extreme', 0)
    if total_train > 0 and extreme_count / total_train < 0.01:
        warnings.append(f"Extreme data only {extreme_count}/{total_train} ({extreme_count/total_train:.2%}). Consider adding more.")
    
    return errors, warnings, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest', help='Path to manifest CSV')
    ap.add_argument('--expected_preproc', default='none', choices=['none', 'gray3'])
    ap.add_argument('--expected_families', default='', help='Comma-separated list')
    args = ap.parse_args()
    
    families = set(args.expected_families.split(',')) if args.expected_families else None
    errors, warnings, stats = validate(args.manifest, args.expected_preproc, families)
    
    print(f"=== Manifest Validation: {args.manifest} ===")
    print(f"Expected preproc: {args.expected_preproc}")
    print(f"Expected families: {families or 'any'}")
    print()
    
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  [E] {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
    
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  [W] {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings)-20} more")
    
    print()
    print("STATS:")
    print(f"  Total rows: {stats['total_rows']}")
    print(f"  Train: {stats['train_rows']}, Test: {stats['test_rows']}")
    print(f"  Families: {dict(stats['family_counts'])}")
    print(f"  Difficulty: {dict(stats['difficulty_counts'])}")
    print(f"  Top provinces: {stats['province_counts'].most_common(10)}")
    print(f"  Missing images: {stats['missing_img']}")
    
    if errors:
        sys.exit(1)
    print("\n[PASS] Manifest validation passed.")

if __name__ == '__main__':
    main()
