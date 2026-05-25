#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import generate_green_edgefit_v4_boardlike_equalprov as base

TOOLS_DIR = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/tools')
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_exact_quad_repo_pilot import (  # type: ignore
    DIRECTION_COMBOS,
    ensure_repo_imports,
    detect_canonical_plate_quad,
    augment_small_new_energy_exact,
    apply_extra_perspective_exact,
    add_canvas_border,
    project_char_quads_to_lpr,
    angle_error_deg,
)

TARGET_PROVINCES = ['京', '沪', '苏', '浙', '粤', '陕']
BUCKET_PLAN = [
    ('geometry_clean', 800),
    ('board_mid_occ', 600),
    ('board_low_occ', 400),
    ('board_extreme_tail', 200),
]
GEOMETRY_CFG = {
    'canvas_border': 64,
    'min_occ_ratio': 0.72,
    'max_char_angle_error_deg': 8.0,
    'angle_horizontal': 24.0,
    'angle_vertical': 24.0,
    'angle_up_down': 18.0,
    'angle_left_right': 12.0,
    'perspective_factor': 14.0,
    'extra_perspective_ratio': 0.18,
}
EXACT_RENDER_CFG = {
    # Keep disabled by default. The external exact-quad augmenter stack still
    # assumes the original template raster in some internal paths; pushing a
    # larger raster through it can desynchronize glyphs from the plate body.
    'supersample_scale': 1.0,
    'supersample_interp': cv2.INTER_CUBIC,
    # The exact-quad source should stay structurally clean. Heavy synthetic
    # appearance corruption here destroys first-character strokes before the
    # later curriculum geometry ever gets a chance to act on them.
    'disable_rand_environment': True,
    'disable_smudge': True,
    'disable_noise': True,
    'disable_gauss': True,
    'disable_hsv': True,
    # Source plates for downstream curriculum should stay fronto-parallel and
    # single-instance. The later curriculum stages already inject geometry;
    # baking another strong perspective here compounds aliasing and can leak
    # reflected plate fragments into the source canvas.
    'source_front_clean_mode': True,
}


def setup_augmenter(augmenter, geom_cfg: dict):
    augmenter.angle_horizontal = float(geom_cfg['angle_horizontal'])
    augmenter.angle_vertical = float(geom_cfg['angle_vertical'])
    augmenter.angle_up_down = float(geom_cfg['angle_up_down'])
    augmenter.angle_left_right = float(geom_cfg['angle_left_right'])
    augmenter.factor = float(geom_cfg['perspective_factor'])
    return augmenter


STRONG_COMBOS = [x for x in DIRECTION_COMBOS if x != ('mid', 'mid')]


def supersample_exact_render(render_item, canonical_quad, scale):
    scale = float(scale)
    image = render_item['image']
    if scale <= 1.0:
        char_quads = []
        for box in render_item['char_boxes']:
            char_quads.append(np.asarray([
                [box['x1'], box['y1']],
                [box['x2'], box['y1']],
                [box['x2'], box['y2']],
                [box['x1'], box['y2']],
            ], dtype=np.float32))
        return image, np.asarray(char_quads, dtype=np.float32), np.asarray(canonical_quad, dtype=np.float32)

    hi_w = max(1, int(round(image.shape[1] * scale)))
    hi_h = max(1, int(round(image.shape[0] * scale)))
    hi_img = cv2.resize(
        image,
        (hi_w, hi_h),
        interpolation=int(EXACT_RENDER_CFG['supersample_interp']),
    )

    char_quads = []
    for box in render_item['char_boxes']:
        char_quads.append(np.asarray([
            [box['x1'], box['y1']],
            [box['x2'], box['y1']],
            [box['x2'], box['y2']],
            [box['x1'], box['y2']],
        ], dtype=np.float32) * scale)

    return (
        hi_img,
        np.asarray(char_quads, dtype=np.float32),
        np.asarray(canonical_quad, dtype=np.float32) * scale,
    )


def load_prepare_board(lpr_root: str):
    lpr_root = str(Path(lpr_root).resolve())
    for p in [lpr_root, str(Path(lpr_root) / 'src')]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from load_data import prepare_board_ocr_input_from_quad_bgr888
    return prepare_board_ocr_input_from_quad_bgr888


def load_shared_suffixes(manifest_path: Path, suffix_count: int, seed: int):
    suffixes = []
    seen = set()
    with manifest_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '')
            if row.get('family') != 'green8':
                continue
            if len(text) != 8:
                continue
            if text[1:3] != 'AD':
                continue
            suffix = text[1:]
            if suffix not in seen:
                seen.add(suffix)
                suffixes.append(suffix)
    if len(suffixes) < suffix_count:
        raise RuntimeError(f'not enough unique AD suffixes: have={len(suffixes)} need={suffix_count}')
    rng = random.Random(seed)
    rng.shuffle(suffixes)
    picked = suffixes[:suffix_count]
    return picked


MANIFEST_FIELDS = base.MANIFEST_FIELDS


def append_manifest(base_manifest: Path, append_rows, out_manifest: Path):
    with base_manifest.open('r', encoding='utf-8', newline='') as f:
        base_rows = list(csv.DictReader(f))
    fieldnames = list(base_rows[0].keys())
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(base_rows)
        w.writerows(append_rows)
    return len(base_rows), len(base_rows) + len(append_rows)


def validate_generated_set(manifest_rows):
    bad = []
    for row in manifest_rows[: min(20, len(manifest_rows))]:
        path = Path(row['img_path'])
        img = cv2.imread(str(path))
        if img is None:
            bad.append({'img_path': str(path), 'reason': 'read_fail'})
            continue
        if img.ndim != 3:
            bad.append({'img_path': str(path), 'reason': f'bad_ndim:{img.ndim}'})
    summary = {'checked_rows': min(20, len(manifest_rows)), 'bad_count': len(bad), 'bad_examples': bad[:10]}
    if bad:
        raise RuntimeError(f'generated set validation failed: {summary}')
    return summary


def render_standard_exact_quad(text: str, chars_gen, augmenter, canonical_quad, prepare_board, rng: random.Random):
    render_item = chars_gen.generate_images_with_metadata([text])[0]
    render_img, char_quads, canonical_quad = supersample_exact_render(
        render_item,
        canonical_quad,
        float(EXACT_RENDER_CFG['supersample_scale']),
    )

    hdir, vdir = rng.choice(STRONG_COMBOS)
    realized, quad, char_quads = augment_small_new_energy_exact(
        render_img,
        char_quads,
        canonical_quad,
        augmenter,
        rng,
        hdir,
        vdir,
        disable_rand_environment=bool(EXACT_RENDER_CFG['disable_rand_environment']),
        disable_smudge=bool(EXACT_RENDER_CFG['disable_smudge']),
        disable_noise=bool(EXACT_RENDER_CFG['disable_noise']),
        disable_gauss=bool(EXACT_RENDER_CFG['disable_gauss']),
        disable_hsv=bool(EXACT_RENDER_CFG['disable_hsv']),
        disable_rand_perspective=False,
    )
    if GEOMETRY_CFG['extra_perspective_ratio'] > 0:
        realized, quad, char_quads, _ = apply_extra_perspective_exact(
            realized, quad, char_quads, rng, float(GEOMETRY_CFG['extra_perspective_ratio'])
        )
    realized, quad, char_quads = add_canvas_border(realized, quad, char_quads, int(GEOMETRY_CFG['canvas_border']))

    prepared94, occ_ratio, _, _, _ = prepare_board(
        realized,
        quad,
        94,
        24,
        'letterbox',
        'nn',
        'none',
        'bgr',
        quad_pad_ratio=0.0,
    )
    lpr_char_quads = project_char_quads_to_lpr(char_quads, quad)
    max_char_err = max(angle_error_deg(cq) for cq in lpr_char_quads) if len(lpr_char_quads) else 0.0
    return {
        'realized': realized,
        'quad': np.asarray(quad, dtype=np.float32),
        'occ_ratio': float(occ_ratio),
        'max_char_angle_error_deg': float(max_char_err),
        'prepared94': prepared94,
        'horizontal_sight_direction': hdir,
        'vertical_sight_direction': vdir,
    }


def render_clean_source_exact_quad(text: str, chars_gen, augmenter, canonical_quad, prepare_board):
    render_item = chars_gen.generate_images_with_metadata([text])[0]
    render_img, char_quads, canonical_quad = supersample_exact_render(
        render_item,
        canonical_quad,
        float(EXACT_RENDER_CFG['supersample_scale']),
    )

    # Compose the glyph render directly onto the plate template with no
    # perspective, no reflected borders, and no extra appearance corruption.
    template_img = augmenter.template_image.copy()
    realized = cv2.bitwise_and(render_img, template_img)
    quad = np.asarray(canonical_quad, dtype=np.float32).copy()
    realized, quad, char_quads = add_canvas_border(
        realized,
        quad,
        char_quads,
        int(GEOMETRY_CFG['canvas_border']),
    )

    prepared94, occ_ratio, _, _, _ = prepare_board(
        realized,
        quad,
        94,
        24,
        'letterbox',
        'nn',
        'none',
        'bgr',
        quad_pad_ratio=0.0,
    )
    lpr_char_quads = project_char_quads_to_lpr(char_quads, quad)
    max_char_err = max(angle_error_deg(cq) for cq in lpr_char_quads) if len(lpr_char_quads) else 0.0
    return {
        'realized': realized,
        'quad': np.asarray(quad, dtype=np.float32),
        'occ_ratio': float(occ_ratio),
        'max_char_angle_error_deg': float(max_char_err),
        'prepared94': prepared94,
        'horizontal_sight_direction': 'mid',
        'vertical_sight_direction': 'mid',
    }


def detail_row_from_record(sample_id, suffix, record, render, province):
    return {
        'sample_id': sample_id,
        'province': province,
        'bucket': record['bucket'],
        'suffix': suffix,
        'text': record['text'],
        'out_rel_path': record['rel_path'],
        'out_abs_path': record['abs_path'],
        'occ_ratio': round(float(record['occ_ratio']), 6),
        'warped_aspect': round(float(record['warped_aspect']), 6),
        'left_right_width_ratio': round(float(record['left_right_width_ratio']), 6),
        'max_char_angle_error_deg': round(float(render['max_char_angle_error_deg']), 6),
        'horizontal_sight_direction': render['horizontal_sight_direction'],
        'vertical_sight_direction': render['vertical_sight_direction'],
    }


def main():
    ap = argparse.ArgumentParser(description='Generate E16A non-anhui AD-balance exact-quad append set from E9C.')
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--repo-root', default='/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/repo_license_plate_generator')
    ap.add_argument('--lpr-root', default='/home/wzzz/LPRNet')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--dataset-name', default='green_e16a_nonanhui_ad_balance_12k_std_v1')
    ap.add_argument('--source-name', default='e16a_nonanhui_ad_balance_12k_std_v1')
    ap.add_argument('--suffix-source-manifest', default='/home/wzzz/LPRNet/manifests/unified_manifest_green_e9c_exact_template_allprov_1800.csv')
    ap.add_argument('--suffix-count', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260417)
    ap.add_argument('--smoke-count-per-province', type=int, default=0)
    args = ap.parse_args()

    base_manifest = Path(args.base_manifest)
    out_root = Path(args.out_dir)
    details_dir = out_root / 'details'
    manifests_dir = out_root / 'manifests'
    qa_dir = out_root / 'qa'
    for d in [out_root, details_dir, manifests_dir, qa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    suffixes = load_shared_suffixes(Path(args.suffix_source_manifest), args.suffix_count, args.seed)
    bucket_for_suffix = []
    idx = 0
    for bucket, count in BUCKET_PLAN:
        for _ in range(count):
            bucket_for_suffix.append((suffixes[idx], bucket))
            idx += 1

    if args.smoke_count_per_province > 0:
        bucket_for_suffix = bucket_for_suffix[: int(args.smoke_count_per_province)]

    chars_gen, augmenter = ensure_repo_imports(args.repo_root)
    augmenter = setup_augmenter(augmenter, GEOMETRY_CFG)
    canonical_quad = detect_canonical_plate_quad(augmenter.template_image)
    prepare_board = load_prepare_board(args.lpr_root)

    manifest_rows = []
    accepted_details = []
    cards = []

    for province in TARGET_PROVINCES:
        for item_idx, (suffix, bucket) in enumerate(bucket_for_suffix):
            text = province + suffix
            item_seed_base = args.seed + (TARGET_PROVINCES.index(province) * 10_000_000) + item_idx * 1009
            accepted = None
            rejects = []
            for attempt in range(40):
                attempt_seed = item_seed_base + attempt * 101
                rng = random.Random(attempt_seed)
                np.random.seed(attempt_seed % (2**32 - 1))
                render = render_standard_exact_quad(text, chars_gen, augmenter, canonical_quad, prepare_board, rng)
                if render['occ_ratio'] < float(GEOMETRY_CFG['min_occ_ratio']):
                    rejects.append({'reason': 'occ_ratio', 'value': render['occ_ratio'], 'attempt': attempt})
                    continue
                if render['max_char_angle_error_deg'] > float(GEOMETRY_CFG['max_char_angle_error_deg']):
                    rejects.append({'reason': 'char_angle_error', 'value': render['max_char_angle_error_deg'], 'attempt': attempt})
                    continue
                sample_id = f'e16a-{province}-{bucket}-{item_idx:04d}-{suffix}'
                record = base.build_sample_record(
                    img=render['realized'],
                    split='train',
                    bucket=bucket,
                    province=province,
                    text=text,
                    exact_quad=render['quad'],
                    board_quad=render['quad'],
                    asym_mode='exact',
                    appearance_meta={'blur_strength': 0.0, 'jpeg_quality': 95, 'appearance_mode': 'standard_exact_ad'},
                    out_root=out_root,
                    uid=sample_id,
                    prepare_fn=prepare_board,
                    dataset_name=args.dataset_name,
                    source=args.source_name,
                )
                manifest_rows.append(record['manifest_row'])
                accepted_details.append(detail_row_from_record(sample_id, suffix, record, render, province))
                if len(cards) < 48:
                    preview = cv2.resize(render['prepared94'], (94 * 4, 24 * 4), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(preview, f'{province} {suffix} {bucket}', (4, preview.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    cards.append(preview)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(f'failed to generate province={province} suffix={suffix} bucket={bucket}; rejects={rejects[:5]}')

    manifest_local = manifests_dir / f'train_manifest_{args.dataset_name}.csv'
    with manifest_local.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest_rows)

    with (details_dir / 'accepted.tsv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = list(accepted_details[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(accepted_details)

    base_count, merged_count = append_manifest(base_manifest, manifest_rows, Path(args.out_manifest))
    validation = validate_generated_set(manifest_rows)

    if cards:
        cols = 4
        pad = 8
        cell_h = max(x.shape[0] for x in cards)
        cell_w = max(x.shape[1] for x in cards)
        rows_n = int(np.ceil(len(cards) / cols))
        sheet = np.full((rows_n * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3), 0, dtype=np.uint8)
        for i, img in enumerate(cards):
            r = i // cols
            c = i % cols
            y = pad + r * (cell_h + pad)
            x = pad + c * (cell_w + pad)
            sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.imwrite(str(qa_dir / 'contact_sheet.jpg'), sheet)

    summary = {
        'dataset_name': args.dataset_name,
        'source_name': args.source_name,
        'accepted_count': len(accepted_details),
        'base_manifest_count': base_count,
        'merged_manifest_count': merged_count,
        'target_provinces': TARGET_PROVINCES,
        'suffix_count': len(bucket_for_suffix),
        'bucket_plan': {k: v for k, v in BUCKET_PLAN},
        'province_counts': dict(sorted(Counter(x['province'] for x in accepted_details).items())),
        'bucket_counts': dict(sorted(Counter(x['bucket'] for x in accepted_details).items())),
        'validation': validation,
        'paths': {
            'out_dir': str(out_root),
            'manifest_local': str(manifest_local),
            'manifest_merged': str(args.out_manifest),
            'accepted_tsv': str(details_dir / 'accepted.tsv'),
            'contact_sheet': str(qa_dir / 'contact_sheet.jpg'),
        },
    }
    (details_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
