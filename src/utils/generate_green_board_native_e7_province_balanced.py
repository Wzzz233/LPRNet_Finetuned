#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E7: Province-balanced board-native data generation
- Target: 3100+ samples (100 per province)
- Per-province filter strategy instead of single global filter
- Relaxed filters to ensure all provinces can contribute
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from generate_green_board_native_preview_v1 import (
    gray_stats,
    load_qspec,
    within_spec,
    spec_score,
    sample_params,
    transform_dumplike_to_board_native,
    ensure_94x24,
    make_card,
    write_contact_sheet,
)


def write_ppm(path: Path, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode('ascii'))
        f.write(rgb.tobytes())


def read_csv_rows(path: Path, delimiter=','):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def join_manifest_and_details(manifest_rows, detail_rows):
    detail_map = {row['out_rel_path']: row for row in detail_rows}
    joined = []
    for row in manifest_rows:
        d = detail_map.get(row['img_rel_path'])
        if d is None:
            continue
        joined.append((row, d))
    return joined


def get_province(text):
    """Extract province character from plate text"""
    if not text:
        return None
    return text[0]


def source_filter_by_province(detail_row, province_filters, prov):
    """Apply province-specific filter"""
    filt = province_filters.get(prov, province_filters['default'])
    
    mean = float(detail_row['mean'])
    left_edge = float(detail_row['left_edge'])
    mid_edge = float(detail_row['mid_edge'])
    
    return (
        filt['mean_min'] <= mean <= filt['mean_max']
        and filt['left_edge_min'] <= left_edge <= filt['left_edge_max']
        and filt['mid_edge_min'] <= mid_edge <= filt['mid_edge_max']
    )


def append_manifest(base_manifest: Path, append_rows, out_manifest: Path):
    base_rows = read_csv_rows(base_manifest)
    fieldnames = list(base_rows[0].keys())
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(base_rows)
        w.writerows(append_rows)
    return len(base_rows), len(base_rows) + len(append_rows)


def main():
    ap = argparse.ArgumentParser(description='E7: Province-balanced board-native data generation')
    ap.add_argument('--input-manifest', default='/home/wzzz/LPRNet/tmp/green_dumplike_boarddump_bright_v1_20260412_a3100/manifests/train_manifest_dumplike_boarddump_v1.csv')
    ap.add_argument('--details-tsv', default='/home/wzzz/LPRNet/tmp/green_dumplike_boarddump_bright_v1_20260412_a3100/details/accepted.tsv')
    ap.add_argument('--spec-json', default='/home/wzzz/LPRNet/tmp/green_board_domain_spec_v1.json')
    ap.add_argument('--base-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/tmp/green_board_native_e7_province_balanced')
    ap.add_argument('--out-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e7_boardnative_provbal.csv')
    ap.add_argument('--required-bucket', default='geometry_clean')
    
    # Relaxed global defaults (will be overridden per-province if needed)
    ap.add_argument('--mean-min', type=float, default=155.0)
    ap.add_argument('--mean-max', type=float, default=185.0)
    ap.add_argument('--left-edge-min', type=float, default=60.0)  # Relaxed from 98
    ap.add_argument('--left-edge-max', type=float, default=180.0)  # Relaxed from 122
    ap.add_argument('--mid-edge-min', type=float, default=80.0)    # Relaxed from 102
    ap.add_argument('--mid-edge-max', type=float, default=200.0)   # Relaxed from 128
    
    ap.add_argument('--per-province-target', type=int, default=100, help='Target samples per province')
    ap.add_argument('--max-variants-per-source', type=int, default=20, help='Max variants per source image')
    ap.add_argument('--max-tries-per-variant', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260414)
    ap.add_argument('--dataset-name', default='green_board_native_e7_provbal')
    ap.add_argument('--source-name', default='board_native_e7_provbal')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images' / 'train'
    cards_dir = out_dir / 'cards'
    details_dir = out_dir / 'details'
    manifests_dir = out_dir / 'manifests'
    for d in [images_dir, cards_dir, details_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load and join data
    manifest_rows = read_csv_rows(Path(args.input_manifest))
    detail_rows = read_csv_rows(Path(args.details_tsv), delimiter='\t')
    joined = join_manifest_and_details(manifest_rows, detail_rows)
    
    # Filter to geometry_clean bucket only
    joined = [(m, d) for (m, d) in joined if d.get('bucket') == args.required_bucket]
    
    print(f"Total geometry_clean samples: {len(joined)}")
    
    # Group by province
    province_candidates = defaultdict(list)
    for m, d in joined:
        prov = get_province(m.get('text', ''))
        if prov:
            province_candidates[prov].append((m, d))
    
    all_provinces = sorted(province_candidates.keys())
    print(f"Found {len(all_provinces)} provinces: {all_provinces}")
    
    # Print candidate counts per province
    for prov in all_provinces:
        print(f"  {prov}: {len(province_candidates[prov])} candidates")
    
    # Define province-specific filters (relaxed for provinces with low left_edge)
    # Based on analysis: some provinces have lower left_edge naturally
    province_filters = {}
    for prov in all_provinces:
        province_filters[prov] = {
            'mean_min': args.mean_min,
            'mean_max': args.mean_max,
            'left_edge_min': args.left_edge_min,  # Relaxed to 60
            'left_edge_max': args.left_edge_max,  # Relaxed to 180
            'mid_edge_min': args.mid_edge_min,    # Relaxed to 80
            'mid_edge_max': args.mid_edge_max,    # Relaxed to 200
        }
    
    province_filters['default'] = {
        'mean_min': args.mean_min,
        'mean_max': args.mean_max,
        'left_edge_min': args.left_edge_min,
        'left_edge_max': args.left_edge_max,
        'mid_edge_min': args.mid_edge_min,
        'mid_edge_max': args.mid_edge_max,
    }

    qspec = load_qspec(Path(args.spec_json))
    
    # Generate per-province
    all_manifest_rows = []
    all_detail_rows = []
    all_cards = []
    province_counts = Counter()
    province_source_counts = defaultdict(Counter)
    all_stats = []
    
    for prov in all_provinces:
        candidates = province_candidates[prov]
        
        # Sort by score (lower is better match to spec)
        candidates.sort(key=lambda x: float(x[1].get('score', '9999')))
        
        generated_for_province = 0
        source_idx_for_province = 0
        
        print(f"\nProcessing {prov} (target: {args.per_province_target}, candidates: {len(candidates)})")
        
        for src_manifest_row, src_detail_row in candidates:
            if generated_for_province >= args.per_province_target:
                break
            
            # Check if this source passes province filter
            if not source_filter_by_province(src_detail_row, province_filters, prov):
                continue
            
            src_img = cv2.imread(src_manifest_row['img_path'], cv2.IMREAD_COLOR)
            if src_img is None:
                continue
            src_img = ensure_94x24(src_img)
            
            variants_for_this_source = 0
            tries = 0
            
            while (tries < args.max_tries_per_variant and 
                   variants_for_this_source < args.max_variants_per_source and
                   generated_for_province < args.per_province_target):
                tries += 1
                params = sample_params(rng)
                gen = transform_dumplike_to_board_native(src_img, params)
                stats = gray_stats(gen)
                
                if not within_spec(stats, qspec):
                    continue
                
                score = spec_score(stats, qspec)
                text = src_manifest_row['text']
                out_name = f'{prov}_s{source_idx_for_province:04d}_v{variants_for_this_source:02d}_{text}.ppm'
                rel_path = f'images/train/{args.required_bucket}/{prov}/{out_name}'
                abs_path = out_dir / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                write_ppm(abs_path, gen)
                
                new_row = dict(src_manifest_row)
                new_row['img_path'] = str(abs_path)
                new_row['img_rel_path'] = rel_path
                new_row['dataset_name'] = args.dataset_name
                new_row['source'] = args.source_name
                all_manifest_rows.append(new_row)
                
                all_detail_rows.append({
                    'source_idx': source_idx_for_province,
                    'variant_idx': variants_for_this_source,
                    'text': text,
                    'province': prov,
                    'src_img_path': src_manifest_row['img_path'],
                    'src_img_rel_path': src_manifest_row['img_rel_path'],
                    'out_img_path': str(abs_path),
                    'out_img_rel_path': rel_path,
                    'score': f'{score:.6f}',
                    'mean': f"{stats['mean']:.6f}",
                    'left_minus_right': f"{stats['left_minus_right']:.6f}",
                    'border_dark_ratio': f"{stats['border_dark_ratio']:.6f}",
                    'left_edge': f"{stats['left_edge']:.6f}",
                    'mid_edge': f"{stats['mid_edge']:.6f}",
                    'occ_ratio': f"{stats['occ_ratio']:.6f}",
                    'params_json': json.dumps(params, ensure_ascii=False),
                })
                
                province_counts[prov] += 1
                province_source_counts[prov][src_manifest_row['img_rel_path']] += 1
                all_stats.append(stats)
                
                # Save card for first few samples
                if generated_for_province < 6 and variants_for_this_source < 2:
                    all_cards.append(make_card(src_img, gen, f'{prov} {text} v{variants_for_this_source:02d}', stats, score))
                
                variants_for_this_source += 1
                generated_for_province += 1
            
            source_idx_for_province += 1
        
        print(f"  Generated {generated_for_province} samples for {prov}")
    
    # Write outputs
    append_manifest_path = manifests_dir / 'train_manifest_green_board_native_e7.csv'
    if all_manifest_rows:
        fieldnames = list(all_manifest_rows[0].keys())
        with append_manifest_path.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_manifest_rows)
    
    details_path = details_dir / 'accepted.tsv'
    if all_detail_rows:
        with details_path.open('w', encoding='utf-8', newline='') as f:
            fieldnames = list(all_detail_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            w.writeheader()
            w.writerows(all_detail_rows)
    
    contact_sheet_path = out_dir / 'contact_sheet.png'
    write_contact_sheet(all_cards, contact_sheet_path, cols=4)
    
    base_total, new_total = append_manifest(Path(args.base_manifest), all_manifest_rows, Path(args.out_manifest))
    
    summary = {
        'experiment_name': 'E7_board_native_province_balanced',
        'base_manifest': args.base_manifest,
        'out_manifest': args.out_manifest,
        'append_manifest': str(append_manifest_path),
        'details_path': str(details_path),
        'contact_sheet_path': str(contact_sheet_path),
        'candidate_source_count': len(joined),
        'append_count': len(all_manifest_rows),
        'per_province_target': args.per_province_target,
        'province_counts': dict(province_counts),
        'base_total': base_total,
        'new_total': new_total,
        'filter_settings': {
            'mean': [args.mean_min, args.mean_max],
            'left_edge': [args.left_edge_min, args.left_edge_max],
            'mid_edge': [args.mid_edge_min, args.mid_edge_max],
        },
        'aggregate': {},
    }
    
    if all_stats:
        for key in all_stats[0].keys():
            vals = np.asarray([x[key] for x in all_stats], dtype=np.float32)
            summary['aggregate'][key] = {
                'mean': float(vals.mean()),
                'min': float(vals.min()),
                'max': float(vals.max()),
            }
    
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n{'='*60}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
