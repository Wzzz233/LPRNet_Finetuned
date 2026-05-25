#!/usr/bin/env python3
"""Phase 2: Export quadwarp plate images for Route A'.

Reads manifest + quad coordinates, perspective-warps plate region,
exports as PNG images at specified sizes and crop modes.

Usage:
  python tools/routeA_prime/export_quadwarp_plate_images.py \
    --manifest manifests_rebased/routeA_prime_quadwarp_20260512/train_real_replace_raw_v1.csv \
    --out_dir datasets/routeA_prime_quadwarp_20260512/fullplate_224x72 \
    --mode fullplate --height 72 --width 224 \
    --max_n 1000 --dry_run

  python tools/routeA_prime/export_quadwarp_plate_images.py \
    --manifest manifests_rebased/routeA_prime_quadwarp_20260512/train_real_replace_bal31_v1.csv \
    --out_dir datasets/routeA_prime_quadwarp_20260512/fullplate_224x72_bal31 \
    --mode fullplate --height 72 --width 224

  python tools/routeA_prime/export_quadwarp_plate_images.py \
    --manifest manifests_rebased/routeA_prime_quadwarp_20260512/train_real_replace_bal31_v1.csv \
    --out_dir datasets/routeA_prime_quadwarp_20260512/leftbias_224x72_bal31 \
    --mode leftbias --height 72 --width 224 --left_ratio 0.55
"""
import argparse, csv, json, os, sys, cv2, numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path('/home/wzzz/LPRNet')


def warp_plate(img, quad, target_w, target_h, mode='fullplate', left_ratio=0.55):
    """Perspective-warp the plate region specified by quad.

    quad: (4,2) numpy array of corner points [TL, TR, BR, BL] (CCPD order)
    mode: 'fullplate' or 'leftbias'
    """
    pts = quad.astype('float32')

    if mode == 'leftbias':
        # Keep left portion of the plate: crop to left_ratio of width
        cx = pts[:, 0].mean()
        left_mask = pts[:, 0] < cx
        if left_mask.sum() >= 2:
            left_pts = pts[pts[:, 0] < cx]
            right_bound = left_pts[:, 0].max() + (pts[:, 0].max() - pts[:, 0].min()) * (1 - left_ratio)
        else:
            right_bound = pts[:, 0].min() + (pts[:, 0].max() - pts[:, 0].min()) * left_ratio

        # Clip points to left region
        new_pts = pts.copy()
        new_pts[:, 0] = np.clip(new_pts[:, 0], pts[:, 0].min(), right_bound)
        # Scale destination to full target width (stretch left region)
        src_pts = new_pts
    else:
        src_pts = pts

    # Bounding box of the plate region
    x_min = max(0, int(src_pts[:, 0].min()) - 5)
    y_min = max(0, int(src_pts[:, 1].min()) - 5)
    x_max = min(img.shape[1], int(src_pts[:, 0].max()) + 5)
    y_max = min(img.shape[0], int(src_pts[:, 1].max()) + 5)

    if x_max <= x_min or y_max <= y_min:
        return None

    # Crop region from original image
    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    # Normalize source points to crop coordinates
    src_norm = src_pts - np.array([[x_min, y_min]], dtype='float32')

    # Destination points: full output rectangle
    dst_pts = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ], dtype='float32')

    matrix = cv2.getPerspectiveTransform(src_norm, dst_pts)
    warped = cv2.warpPerspective(crop, matrix, (target_w, target_h))
    return warped


def export_manifest(manifest_path, out_dir, mode, target_w, target_h,
                    left_ratio=0.55, max_n=None, dry_run=False):
    """Export quad-warped plate images from a manifest file."""
    out_dir = Path(out_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(manifest_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    if max_n:
        rows = rows[:max_n]

    exported = 0
    errors = 0
    missing_quad = 0
    missing_img = 0
    warp_fail = 0

    out_rows = []  # For manifest pointing at exported images

    for idx, row in enumerate(rows):
        img_path = row.get('img_path', '').strip()
        text = row.get('text', '').strip()
        has_quad = row.get('has_quad', '0').strip() in ('1', 'True')

        if not img_path or not text:
            errors += 1
            continue

        # Resolve full path
        full_path = img_path if os.path.isabs(img_path) else str(ROOT / img_path)
        if not Path(full_path).exists():
            missing_img += 1
            continue

        # Parse quad
        if not has_quad:
            missing_quad += 1
            if dry_run:
                continue
            # Fallback: use full image center region
            continue

        try:
            quad = np.array([
                [float(row['quad_1x']), float(row['quad_1y'])],
                [float(row['quad_2x']), float(row['quad_2y'])],
                [float(row['quad_3x']), float(row['quad_3y'])],
                [float(row['quad_4x']), float(row['quad_4y'])],
            ])
        except (KeyError, ValueError):
            missing_quad += 1
            continue

        # Early filter: all coords must be non-negative
        if quad.min() < 0:
            missing_quad += 1
            continue

        if dry_run:
            exported += 1
            continue

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            missing_img += 1
            continue

        # Warp
        plate = warp_plate(img, quad, target_w, target_h, mode, left_ratio)
        if plate is None:
            warp_fail += 1
            continue

        # Save
        safe_name = f'{idx:08d}_{text}.png'.replace('/', '_')
        out_path = out_dir / safe_name
        cv2.imwrite(str(out_path), plate)
        exported += 1

        # Create output manifest row
        new_row = {k: row.get(k, '') for k in fieldnames}
        new_row['img_path'] = str(out_path)
        new_row['export_mode'] = mode
        new_row['export_size'] = f'{target_w}x{target_h}'
        out_rows.append(new_row)

        if exported % 2000 == 0:
            print(f'  Exported {exported}...')

    stats = {
        'total_input': len(rows),
        'exported': exported,
        'errors': errors,
        'missing_quad': missing_quad,
        'missing_img': missing_img,
        'warp_fail': warp_fail,
        'out_dir': str(out_dir),
        'mode': mode,
        'size': f'{target_w}x{target_h}',
    }

    print(f'  Stats: {exported} exported, {missing_quad} no quad, {missing_img} img missing, {warp_fail} warp fail')

    # Write output manifest if not dry_run
    if not dry_run and out_rows:
        manifest_fields = fieldnames + ['export_mode', 'export_size']
        out_manifest = out_dir / 'manifest.csv'
        with open(out_manifest, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=manifest_fields)
            w.writeheader()
            w.writerows(out_rows)
        print(f'  Output manifest: {out_manifest}')

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mode', default='fullplate', choices=['fullplate', 'leftbias'])
    ap.add_argument('--width', type=int, default=224)
    ap.add_argument('--height', type=int, default=72)
    ap.add_argument('--left_ratio', type=float, default=0.55)
    ap.add_argument('--max_n', type=int, default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    print(f'Exporting: {args.mode} {args.width}x{args.height}')
    print(f'  Manifest: {args.manifest}')
    print(f'  Output:   {args.out_dir}')

    stats = export_manifest(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        mode=args.mode,
        target_w=args.width,
        target_h=args.height,
        left_ratio=args.left_ratio,
        max_n=args.max_n,
        dry_run=args.dry_run,
    )

    # Save stats
    if args.dry_run:
        print(f'\n  Dry-run: would export {stats["exported"]} images')
    else:
        stats_path = Path(args.out_dir) / 'export_stats.json'
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f'\n  Stats saved: {stats_path}')


if __name__ == '__main__':
    main()
