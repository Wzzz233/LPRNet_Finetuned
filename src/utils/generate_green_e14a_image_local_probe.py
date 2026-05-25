#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

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

TOOLS_DIR = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/test/tools')
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_exact_quad_repo_pilot import (  # type: ignore
    PROVINCES,
    DIRECTION_COMBOS,
    ensure_repo_imports,
    ensure_lpr_imports,
    detect_canonical_plate_quad,
    augment_small_new_energy_exact,
    apply_extra_perspective_exact,
    add_canvas_border,
    project_char_quads_to_lpr,
    angle_error_deg,
)

MANIFEST_FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type',
    'source', 'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad',
    'can_parse_ccpd_geom', 'can_perspective', 'bbox_source', 'quad_source',
    'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
]

PROV_DIR = {p: f'p{i:02d}_u{ord(p):04x}' for i, p in enumerate(PROVINCES)}
STRONG_COMBOS = [x for x in DIRECTION_COMBOS if x != ('mid', 'mid')]


def apportion(total, ratio_map, ordered_keys):
    raw = {k: total * float(ratio_map[k]) for k in ordered_keys}
    counts = {k: int(np.floor(raw[k])) for k in ordered_keys}
    rem = total - sum(counts.values())
    if rem > 0:
        order = sorted(ordered_keys, key=lambda k: (raw[k] - counts[k], ratio_map[k], k), reverse=True)
        for k in order[:rem]:
            counts[k] += 1
    return counts


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


def append_manifest(base_manifest: Path, append_rows, out_manifest: Path):
    base_rows = read_csv_rows(base_manifest)
    if not base_rows:
        raise RuntimeError(f'base manifest empty: {base_manifest}')
    fieldnames = list(base_rows[0].keys())
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(base_rows)
        w.writerows(append_rows)
    return len(base_rows), len(base_rows) + len(append_rows)


def family_to_text(province: str, family_type: str, cfg: dict, rng: random.Random) -> str:
    if family_type == 'exact_anchor':
        return province + 'AA02222'
    if family_type == 'tail_neighbor':
        return province + 'AA' + rng.choice(cfg['tail_neighbor_suffixes'])
    if family_type == 'prefix_neighbor':
        return province + 'AD02222'
    if family_type == 'short_control':
        return province + 'AA0222' + rng.choice(cfg['short_control_last_digits'])
    raise ValueError(f'unknown family_type: {family_type}')


LOCAL_CYCLE = ['aa0_local_blur', 'aa0_local_highlight', 'aa0_local_edge']


def local_mode_schedule(base_mode: str, attempt: int, max_attempts: int) -> str:
    if base_mode == 'none' or max_attempts <= 1:
        return base_mode
    if attempt < max_attempts // 3:
        return base_mode
    if attempt < (2 * max_attempts) // 3:
        cycle = [base_mode] + [m for m in LOCAL_CYCLE if m != base_mode]
        return cycle[(attempt - max_attempts // 3) % len(cycle)]
    rescue_priority = {
        'aa0_local_highlight': ['aa0_local_blur', 'aa0_local_edge', 'aa0_local_highlight'],
        'aa0_local_blur': ['aa0_local_blur', 'aa0_local_edge', 'aa0_local_highlight'],
        'aa0_local_edge': ['aa0_local_edge', 'aa0_local_blur', 'aa0_local_highlight'],
    }
    cycle = rescue_priority.get(base_mode, [base_mode] + [m for m in LOCAL_CYCLE if m != base_mode])
    return cycle[(attempt - (2 * max_attempts) // 3) % len(cycle)]


def build_plan(cfg: dict, rng: random.Random):
    items = []
    family_keys = ['exact_anchor', 'tail_neighbor', 'prefix_neighbor', 'short_control']
    layer_keys = ['local_strong', 'strong_no_local', 'board_only']
    local_idx = 0
    for province, total in cfg['province_quota'].items():
        family_counts = apportion(int(total), cfg['family_ratios'], family_keys)
        for family_type in family_keys:
            count = family_counts[family_type]
            if count <= 0:
                continue
            if family_type == 'exact_anchor':
                layer_counts = apportion(count, cfg['major_layer_ratios'], layer_keys)
                for layer_name in layer_keys:
                    for _ in range(layer_counts[layer_name]):
                        local_mode = 'none'
                        if layer_name == 'local_strong':
                            local_mode = LOCAL_CYCLE[local_idx % len(LOCAL_CYCLE)]
                            local_idx += 1
                        items.append({
                            'province': province,
                            'family_type': family_type,
                            'layer_name': layer_name,
                            'local_mode': local_mode,
                            'text': family_to_text(province, family_type, cfg, rng),
                        })
            else:
                for _ in range(count):
                    items.append({
                        'province': province,
                        'family_type': family_type,
                        'layer_name': 'strong_no_local',
                        'local_mode': 'none',
                        'text': family_to_text(province, family_type, cfg, rng),
                    })
    rng.shuffle(items)
    return items


def setup_augmenter(augmenter, geom_cfg: dict):
    augmenter.angle_horizontal = float(geom_cfg['angle_horizontal'])
    augmenter.angle_vertical = float(geom_cfg['angle_vertical'])
    augmenter.angle_up_down = float(geom_cfg['angle_up_down'])
    augmenter.angle_left_right = float(geom_cfg['angle_left_right'])
    augmenter.factor = float(geom_cfg['perspective_factor'])
    return augmenter


def render_exact_quad(text: str, chars_gen, augmenter, canonical_quad, prepare_board, item: dict, geom_cfg: dict, rng: random.Random):
    render_item = chars_gen.generate_images_with_metadata([text])[0]
    char_quads = []
    for box in render_item['char_boxes']:
        char_quads.append(np.asarray([
            [box['x1'], box['y1']],
            [box['x2'], box['y1']],
            [box['x2'], box['y2']],
            [box['x1'], box['y2']],
        ], dtype=np.float32))
    char_quads = np.asarray(char_quads, dtype=np.float32)

    if item['layer_name'] == 'board_only':
        realized = cv2.bitwise_not(render_item['image'])
        quad = np.asarray(canonical_quad, dtype=np.float32).copy()
        hdir, vdir = ('mid', 'mid')
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
            'prepared94': ensure_94x24(prepared94),
            'quad': np.asarray(quad, dtype=np.float32),
            'lpr_char_quads': np.asarray(lpr_char_quads, dtype=np.float32),
            'occ_ratio': float(occ_ratio),
            'max_char_angle_error_deg': float(max_char_err),
            'horizontal_sight_direction': hdir,
            'vertical_sight_direction': vdir,
        }

    geom_override = item.get('geometry_mode_override', 'default')
    if geom_override == 'relaxed':
        combo_pool = list(DIRECTION_COMBOS)
        hdir, vdir = rng.choice(combo_pool)
        disable_rand_perspective = False
        extra_perspective_ratio = float(geom_cfg['extra_perspective_ratio']) * 0.5
        canvas_border = max(32, int(geom_cfg['canvas_border']) * 3 // 4)
    else:
        hdir, vdir = rng.choice(STRONG_COMBOS)
        disable_rand_perspective = False
        extra_perspective_ratio = float(geom_cfg['extra_perspective_ratio'])
        canvas_border = int(geom_cfg['canvas_border'])

    realized, quad, char_quads = augment_small_new_energy_exact(
        render_item['image'],
        char_quads,
        canonical_quad,
        augmenter,
        rng,
        hdir,
        vdir,
        disable_rand_environment=False,
        disable_smudge=False,
        disable_noise=False,
        disable_gauss=False,
        disable_hsv=False,
        disable_rand_perspective=disable_rand_perspective,
    )
    if extra_perspective_ratio > 0:
        realized, quad, char_quads, _ = apply_extra_perspective_exact(realized, quad, char_quads, rng, extra_perspective_ratio)
    realized, quad, char_quads = add_canvas_border(realized, quad, char_quads, canvas_border)

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
        'prepared94': ensure_94x24(prepared94),
        'quad': np.asarray(quad, dtype=np.float32),
        'lpr_char_quads': np.asarray(lpr_char_quads, dtype=np.float32),
        'occ_ratio': float(occ_ratio),
        'max_char_angle_error_deg': float(max_char_err),
        'horizontal_sight_direction': hdir,
        'vertical_sight_direction': vdir,
    }


def _clip_u8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def slot_bbox_from_char_quads(char_quads: np.ndarray, start_idx=1, end_idx=4, pad=1):
    if char_quads is None or len(char_quads) < end_idx:
        return None
    region = np.asarray(char_quads[start_idx:end_idx], dtype=np.float32).reshape(-1, 2)
    x1 = max(0, int(np.floor(region[:, 0].min())) - pad)
    y1 = max(0, int(np.floor(region[:, 1].min())) - pad)
    x2 = min(93, int(np.ceil(region[:, 0].max())) + pad)
    y2 = min(23, int(np.ceil(region[:, 1].max())) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def apply_local_mode(img: np.ndarray, char_quads: np.ndarray, mode: str, rng_np: np.random.Generator) -> np.ndarray:
    bbox = slot_bbox_from_char_quads(char_quads)
    if bbox is None or mode == 'none':
        return img
    x1, y1, x2, y2 = bbox
    out = ensure_94x24(img).astype(np.float32)
    patch = out[y1:y2 + 1, x1:x2 + 1].copy()
    if patch.size == 0:
        return img
    if mode == 'aa0_local_blur':
        sigma = float(rng_np.uniform(0.5, 1.2))
        k = int(rng_np.choice([3, 5]))
        blur = cv2.GaussianBlur(patch, (k, k), sigma)
        mix = float(rng_np.uniform(0.30, 0.65))
        patch = patch * (1.0 - mix) + blur * mix
    elif mode == 'aa0_local_highlight':
        alpha = float(rng_np.uniform(0.82, 0.92))
        beta = float(rng_np.uniform(10.0, 24.0))
        patch = (patch - 128.0) * alpha + 128.0 + beta
        if rng_np.random() < 0.5:
            patch = cv2.GaussianBlur(patch, (3, 3), float(rng_np.uniform(0.25, 0.7)))
    elif mode == 'aa0_local_edge':
        target_w = max(2, int((x2 - x1 + 1) * float(rng_np.uniform(0.90, 0.97))))
        tmp = cv2.resize(patch, (target_w, patch.shape[0]), interpolation=cv2.INTER_LINEAR)
        tmp = cv2.resize(tmp, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
        mix = float(rng_np.uniform(0.25, 0.55))
        patch = patch * (1.0 - mix) + tmp * mix
        if rng_np.random() < 0.5:
            patch = cv2.GaussianBlur(patch, (3, 3), float(rng_np.uniform(0.25, 0.7)))
    else:
        raise ValueError(f'unknown local mode: {mode}')
    out[y1:y2 + 1, x1:x2 + 1] = patch
    return _clip_u8(out)


def save_sample(candidate: np.ndarray, province: str, out_root: Path, sample_id: str) -> tuple[str, str]:
    pdir = PROV_DIR.get(province, f'p{ord(province):02d}_u{ord(province):04x}')
    rel = f'images/train/{pdir}/{sample_id}.ppm'
    abs_path = out_root / rel
    write_ppm(abs_path, candidate)
    return rel, str(abs_path)


def build_manifest_row(abs_path: str, rel_path: str, text: str, cfg: dict):
    return {
        'img_path': abs_path,
        'img_rel_path': rel_path,
        'dataset_name': cfg['dataset_name'],
        'split': 'train',
        'text': text,
        'plate_len': len(text),
        'family': 'green8',
        'sub_type': 'green_small',
        'source': cfg['source_name'],
        'is_real': 0,
        'need_tilt_aug': 0,
        'preprocess_group': 'board_dump',
        'has_bbox': 0,
        'has_quad': 0,
        'can_parse_ccpd_geom': 0,
        'can_perspective': 0,
        'bbox_source': 'none',
        'quad_source': 'none',
        'ocr_channel_order': 'bgr',
        'ocr_crop_mode': 'board_dump',
        'ocr_resize_mode': 'letterbox',
        'ocr_resize_kernel': 'nn',
        'ocr_preproc': 'none',
        'ocr_min_occ_ratio': 1.0,
        'ocr_quad_pad_ratio': 0.0,
    }


def validate_generated_set(manifest_rows, out_root: Path):
    bad = []
    for row in manifest_rows[: min(20, len(manifest_rows))]:
        path = Path(row['img_path'])
        img = cv2.imread(str(path))
        if img is None:
            bad.append({'img_path': str(path), 'reason': 'read_fail'})
            continue
        if img.shape[:2] != (24, 94):
            bad.append({'img_path': str(path), 'reason': f'bad_shape:{img.shape[:2]}'})
    summary = {'checked_rows': min(20, len(manifest_rows)), 'bad_count': len(bad), 'bad_examples': bad[:10]}
    if bad:
        raise RuntimeError(f'generated set validation failed: {summary}')
    return summary


def main():
    ap = argparse.ArgumentParser(description='Generate E14A image-local accepted-only board_dump append data.')
    ap.add_argument('--plan-json', default='/home/wzzz/LPRNet/tmp/e14a_image_local_probe_plan_v1.json')
    ap.add_argument('--limit-items', type=int, default=0, help='debug only: limit planned items')
    args = ap.parse_args()

    cfg = json.loads(Path(args.plan_json).read_text(encoding='utf-8'))
    seed = int(cfg['seed'])
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    out_root = Path(cfg['out_dir'])
    details_dir = out_root / 'details'
    qa_dir = out_root / 'qa'
    cards_dir = qa_dir / 'cards'
    manifests_dir = out_root / 'manifests'
    for d in [out_root, details_dir, qa_dir, cards_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    chars_gen, augmenter = ensure_repo_imports(cfg['repo_root'])
    augmenter = setup_augmenter(augmenter, cfg['geometry'])
    canonical_quad = detect_canonical_plate_quad(augmenter.template_image)
    prepare_board = ensure_lpr_imports(cfg['lpr_root'])
    qspec = load_qspec(Path(cfg['spec_json']))

    plan_items = build_plan(cfg, rng)
    if args.limit_items > 0:
        plan_items = plan_items[: int(args.limit_items)]

    accepted_details = []
    rejected_details = []
    manifest_rows = []
    cards = []

    max_attempts_per_item = int(cfg['matching']['max_attempts_per_item'])
    max_spec_tries = int(cfg['matching']['max_spec_tries'])
    contact_sheet_limit = int(cfg['matching']['contact_sheet_limit'])
    min_occ_ratio = float(cfg['geometry']['min_occ_ratio'])
    max_char_angle_error_deg = float(cfg['geometry']['max_char_angle_error_deg'])

    for item_idx, item in enumerate(plan_items):
        accepted = False
        item_seed_base = seed + item_idx * 1000003
        for attempt in range(max_attempts_per_item):
            attempt_seed = item_seed_base + attempt * 10007
            attempt_rng = random.Random(attempt_seed)
            attempt_rng_np = np.random.default_rng(attempt_seed)
            random.seed(attempt_seed)
            np.random.seed(attempt_seed % (2**32 - 1))
            render_item = dict(item)
            applied_local_mode = local_mode_schedule(item['local_mode'], attempt, max_attempts_per_item)
            render_item['local_mode_applied'] = applied_local_mode
            if item['layer_name'] == 'local_strong' and attempt >= max_attempts_per_item // 2:
                render_item['geometry_mode_override'] = 'relaxed'
            render = render_exact_quad(
                item['text'],
                chars_gen,
                augmenter,
                canonical_quad,
                prepare_board,
                render_item,
                cfg['geometry'],
                attempt_rng,
            )
            if render['occ_ratio'] < min_occ_ratio:
                rejected_details.append({
                    'item_idx': item_idx,
                    'text': item['text'],
                    'province': item['province'],
                    'family_type': item['family_type'],
                    'layer_name': item['layer_name'],
                    'local_mode': item['local_mode'],
                    'local_mode_applied': applied_local_mode,
                    'reject_stage': 'geometry',
                    'reason': 'occ_ratio',
                    'occ_ratio': render['occ_ratio'],
                    'attempt': attempt,
                })
                continue
            if render['max_char_angle_error_deg'] > max_char_angle_error_deg:
                rejected_details.append({
                    'item_idx': item_idx,
                    'text': item['text'],
                    'province': item['province'],
                    'family_type': item['family_type'],
                    'layer_name': item['layer_name'],
                    'local_mode': item['local_mode'],
                    'local_mode_applied': applied_local_mode,
                    'reject_stage': 'geometry',
                    'reason': 'char_angle_error',
                    'max_char_angle_error_deg': render['max_char_angle_error_deg'],
                    'attempt': attempt,
                })
                continue

            base94 = ensure_94x24(render['prepared94'])
            for spec_try in range(max_spec_tries):
                params = sample_params(attempt_rng_np)
                candidate = transform_dumplike_to_board_native(base94, params)
                candidate = apply_local_mode(candidate, render['lpr_char_quads'], applied_local_mode, attempt_rng_np)
                stats = gray_stats(candidate)
                score = spec_score(stats, qspec)
                if not within_spec(stats, qspec):
                    rejected_details.append({
                        'item_idx': item_idx,
                        'text': item['text'],
                        'province': item['province'],
                        'family_type': item['family_type'],
                        'layer_name': item['layer_name'],
                        'local_mode': item['local_mode'],
                        'local_mode_applied': applied_local_mode,
                        'reject_stage': 'spec',
                        'reason': 'out_of_spec',
                        'spec_try': spec_try,
                        'attempt': attempt,
                        'score': score,
                        **stats,
                    })
                    continue

                sample_id = f'e14a-{item_idx:04d}-{attempt:02d}-{spec_try:02d}-{item["province"]}-{item["family_type"]}-{item["layer_name"]}'
                rel_path, abs_path = save_sample(candidate, item['province'], out_root, sample_id)
                manifest_row = build_manifest_row(abs_path, rel_path, item['text'], cfg)
                manifest_rows.append(manifest_row)
                detail_row = {
                    'sample_id': sample_id,
                    'text': item['text'],
                    'province': item['province'],
                    'family_type': item['family_type'],
                    'layer_name': item['layer_name'],
                    'local_mode': item['local_mode'],
                    'local_mode_applied': applied_local_mode,
                    'horizontal_sight_direction': render['horizontal_sight_direction'],
                    'vertical_sight_direction': render['vertical_sight_direction'],
                    'occ_ratio_raw': render['occ_ratio'],
                    'max_char_angle_error_deg': render['max_char_angle_error_deg'],
                    'out_rel_path': rel_path,
                    'out_abs_path': abs_path,
                    'score': score,
                    **stats,
                }
                accepted_details.append(detail_row)
                if len(cards) < contact_sheet_limit:
                    label = f"{item['province']} {item['text']} {item['family_type']} {item['layer_name']} {item['local_mode']}"
                    cards.append(make_card(base94, candidate, label, stats, score))
                accepted = True
                break
            if accepted:
                break

        if not accepted:
            raise RuntimeError(f'failed to generate accepted sample for item_idx={item_idx} text={item["text"]} family={item["family_type"]} layer={item["layer_name"]}')

    manifest_local = manifests_dir / 'train_manifest_green_e14a_image_local_probe_300_v1.csv'
    with manifest_local.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest_rows)

    with (details_dir / 'accepted.tsv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = list(accepted_details[0].keys()) if accepted_details else ['sample_id']
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(accepted_details)

    with (details_dir / 'rejected.tsv').open('w', encoding='utf-8', newline='') as f:
        if rejected_details:
            fieldnames = sorted({k for row in rejected_details for k in row.keys()})
        else:
            fieldnames = ['item_idx', 'reason']
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(rejected_details)

    write_contact_sheet(cards, qa_dir / 'contact_sheet.png', cols=4)
    for i, card in enumerate(cards[:contact_sheet_limit]):
        cv2.imwrite(str(cards_dir / f'card_{i:03d}.jpg'), card)

    base_count, merged_count = append_manifest(Path(cfg['base_manifest']), manifest_rows, Path(cfg['out_manifest']))
    validation = validate_generated_set(manifest_rows, out_root)

    summary = {
        'plan_name': cfg['name'],
        'accepted_count': len(accepted_details),
        'rejected_count': len(rejected_details),
        'base_manifest_count': base_count,
        'merged_manifest_count': merged_count,
        'province_counts': dict(sorted(Counter(x['province'] for x in accepted_details).items())),
        'family_counts': dict(sorted(Counter(x['family_type'] for x in accepted_details).items())),
        'layer_counts': dict(sorted(Counter(x['layer_name'] for x in accepted_details).items())),
        'local_mode_counts': dict(sorted(Counter(x['local_mode'] for x in accepted_details).items())),
        'validation': validation,
        'spec_metric_mean': {
            'mean': float(np.mean([x['mean'] for x in accepted_details])) if accepted_details else 0.0,
            'left_minus_right': float(np.mean([x['left_minus_right'] for x in accepted_details])) if accepted_details else 0.0,
            'border_dark_ratio': float(np.mean([x['border_dark_ratio'] for x in accepted_details])) if accepted_details else 0.0,
            'left_edge': float(np.mean([x['left_edge'] for x in accepted_details])) if accepted_details else 0.0,
            'mid_edge': float(np.mean([x['mid_edge'] for x in accepted_details])) if accepted_details else 0.0,
            'occ_ratio': float(np.mean([x['occ_ratio'] for x in accepted_details])) if accepted_details else 0.0,
        },
        'paths': {
            'out_dir': str(out_root),
            'manifest_local': str(manifest_local),
            'manifest_merged': str(cfg['out_manifest']),
            'accepted_tsv': str(details_dir / 'accepted.tsv'),
            'rejected_tsv': str(details_dir / 'rejected.tsv'),
            'contact_sheet': str(qa_dir / 'contact_sheet.png'),
        },
    }
    (details_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
