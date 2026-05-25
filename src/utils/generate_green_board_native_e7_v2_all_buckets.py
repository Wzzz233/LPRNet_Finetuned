#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E7-v2: Province-balanced board-native using ALL buckets
- Use all 3100 candidate sources (not just geometry_clean)
- Relaxed spec constraints for transform output
- Target: 3100 samples (100 per province)
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


def relaxed_within_spec(stats, qspec, relax_factor=2.0):
    """Relaxed spec check - allow values outside q10-q90 but penalize"""
    # Use wider bounds: q10 - relax_factor*(q50-q10) to q90 + relax_factor*(q90-q50)
    checks = [
        'occ_ratio', 'mean', 'std', 'left_minus_right',
        'border_dark_ratio', 'left_edge', 'mid_edge'
    ]
    
    for key in checks:
        if key not in stats or key not in qspec:
            continue
        val = stats[key]
        q10 = qspec[key]['q10']
        q50 = qspec[key]['q50']
        q90 = qspec[key]['q90']
        
        # Wider bounds
        lower = q10 - relax_factor * (q50 - q10)
        upper = q90 + relax_factor * (q90 - q50)
        
        if not (lower <= val <= upper):
            return False
    return True


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
    ap = argparse.ArgumentParser(description='E7-v2: Province-balanced using all buckets')
    ap.add_argument('--input-manifest', default='/home/wzzz/LPRNet/tmp/green_dumplike_boarddump_bright_v1_20260412_a3100/manifests/train_manifest_dumplike_boarddump_v1.csv')
    ap.add_argument('--details-tsv', default='/home/wzzz/LPRNet/tmp/green_dumplike_boarddump_bright_v1_20260412_a3100/details/accepted.tsv')
    ap.add_argument('--spec-json', default='/home/wzzz/LPRNet/tmp/green_board_domain_spec_v1.json')
    ap.add_argument('--base-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative_fixed_e2_v4_20260411.csv')
    ap.add_argument('--out-dir', default='/home/wzzz/LPRNet/tmp/green_board_native_e7_v2')
    ap.add_argument('--out-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_e7_boardnative_v2.csv')
    
    # Use ALL buckets, not just geometry_clean
    ap.add_argument('--buckets', default='geometry_clean,board_mid_occ,board_low_occ,board_extreme_tail')
    
    ap.add_argument('--per-province-target', type=int, default=100)
    ap.add_argument('--max-variants-per-source', type=int, default=10)
    ap.add_argument('--max-tries-per-variant', type=int, default=500)
    ap.add_argument('--spec-relax-factor', type=float, default=2.0, help='Relax spec bounds by this factor')
    ap.add_argument('--seed', type=int, default=20260414)
    ap.add_argument('--dataset-name', default='green_board_native_e7_v2')
    ap.add_argument('--source-name', default='board_native_e7_v2')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images' / 'train'
    cards_dir = out_dir / 'cards'
    details_dir = out_dir / 'details'
    manifests_dir = out_dir / 'manifests'
    for d in [images_dir, cards_dir, details_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    allowed_buckets = set(args.buckets.split(','))
    print(f"Using buckets: {allowed_buckets}")

    # Load and join data
    manifest_rows = read_csv_rows(Path(args.input_manifest))
    detail_rows = read_csv_rows(Path(args.details_tsv), delimiter='\t')
    joined = join_manifest_and_details(manifest_rows, detail_rows)
    
    # Filter to allowed buckets
    joined = [(m, d) for (m, d) in joined if d.get('bucket') in allowed_buckets]
    
    print(f"Total candidates from all buckets: {len(joined)}")
    
    # Group by province
    province_candidates = defaultdict(list)
    for m, d in joined:
        text = m.get('text', '')
        prov = text[0] if text else None
        if prov:
            province_candidates[prov].append((m, d))
    
    all_provinces = sorted(province_candidates.keys())
    print(f"Found {len(all_provinces)} provinces")
    for prov in all_provinces:
        print(f"  {prov}: {len(province_candidates[prov])} candidates")
    
    # Load spec
    qspec = load_qspec(Path(args.spec_json))
    
    # Generate per-province
    all_manifest_rows = []
    all_detail_rows = []
    all_cards = []
    province_counts = Counter()
    all_stats = []
    
    for prov in all_provinces:
        candidates = province_candidates[prov]
        candidates.sort(key=lambda x: float(x[1].get('score', '9999')))
        
        generated_for_province = 0
        source_idx = 0
        
        print(f"\nProcessing {prov} (target: {args.per_province_target}, candidates: {len(candidates)})")
        
        for src_manifest_row, src_detail_row in candidates:
            if generated_for_province >= args.per_province_target:
                break
            
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
                
                # Use relaxed spec check
                if not relaxed_within_spec(stats, qspec, args.spec_relax_factor):
                    continue
                
                score = spec_score(stats, qspec)
                text = src_manifest_row['text']
                bucket = src_detail_row.get('bucket', 'unknown')
                
                out_name = f'{prov}_{bucket}_s{source_idx:04d}_v{variants_for_this_source:02d}_{text}.ppm'
                rel_path = f'images/train/{bucket}/{prov}/{out_name}'
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
                    'source_idx': source_idx,
                    'variant_idx': variants_for_this_source,
                    'text': text,
                    'province': prov,
                    'bucket': bucket,
                    'src_img_path': src_manifest_row['img_path'],
                    'out_img_path': str(abs_path),
                    'out_img_rel_path': rel_path,
                    'score': f'{score:.6f}',
                    'mean': f"{stats['mean']:.6f}",
                    'left_minus_right': f"{stats['left_minus_right']:.6f}",
                    'border_dark_ratio': f"{stats['border_dark_ratio']:.6f}",
                    'left_edge': f"{stats['left_edge']:.6f}",
                    'mid_edge': f"{stats['mid_edge']:.6f}",
                    'occ_ratio': f"{stats['occ_ratio']:.6f}",
                })
                
                province_counts[prov] += 1
                all_stats.append(stats)
                
                if generated_for_province < 5 and variants_for_this_source < 1:
                    all_cards.append(make_card(src_img, gen, f'{prov} {text}', stats, score))
                
                variants_for_this_source += 1
                generated_for_province += 1
            
            source_idx += 1
        
        print(f"  Generated {generated_for_province} samples for {prov}")
    
    # Write outputs
    append_manifest_path = manifests_dir / 'train_manifest.csv'
    if all_manifest_rows:
        fieldnames = list(all_manifest_rows[0].keys())
        with append_manifest_path.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_manifest_rows)
    
    details_path = details_dir / 'details.tsv'
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
        'experiment_name': 'E7_v2_all_buckets',
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
    print(f"Generated {len(all_manifest_rows)} samples total")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
