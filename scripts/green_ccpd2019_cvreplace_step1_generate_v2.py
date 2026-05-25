#!/usr/bin/env python3
"""Step 1 v2: Generate green plate replacement data from CCPD2019 tilt/db/challenge.
v2 fixes:
- L-channel ONLY CV transfer (preserve green A/B from generated plate)
- enforce_green_plate_color() guard
- blue_ratio < 0.10 threshold, skip samples that fail
- Output to v2 directories"""

import csv, json, os, sys, math, random, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'utils'))

from load_data import prepare_board_ocr_input_from_quad_bgr888, parse_ccpd_quad_from_name, order_quad_points
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--train_per_province_per_subset', type=int, default=200)
parser.add_argument('--val_per_province_per_subset', type=int, default=20)
parser.add_argument('--skip_generate', action='store_true')
args = parser.parse_args()

if args.smoke:
    TRAIN_PER_PROV = 2
    VAL_PER_PROV = 1
else:
    TRAIN_PER_PROV = args.train_per_province_per_subset
    VAL_PER_PROV = args.val_per_province_per_subset

DATE_TAG = '20260508'
POSE_JSONL = ROOT / 'datasets' / f'ccpd2019_tilt_db_challenge_posquads_{DATE_TAG}' / 'pose_quads.jsonl'
BLUE_TRAIN_MANIFEST = ROOT / 'manifests_rebased' / f'blue_ccpd2019_tilt_db_challenge_posquad_{DATE_TAG}' / 'train_posquad.csv'
BLUE_TEST_MANIFEST = ROOT / 'manifests_rebased' / f'blue_ccpd2019_tilt_db_challenge_posquad_{DATE_TAG}' / 'test_posquad.csv'
OUT_DIR = ROOT / 'datasets' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v2_{DATE_TAG}'
MANIFEST_DIR = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v2_{DATE_TAG}'
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/green_ccpd2019_cvreplace_v2')

OUT_IMG_DIR_TRAIN = OUT_DIR / 'images' / 'train'
OUT_IMG_DIR_VAL = OUT_DIR / 'images' / 'val'
for d in [OUT_DIR, OUT_IMG_DIR_TRAIN, OUT_IMG_DIR_VAL, MANIFEST_DIR, WIN_QA]:
    d.mkdir(parents=True, exist_ok=True)

random.seed(20260508)
ALL_PROVINCES = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                 '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                 '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']

# ── Init generation pipeline ────────────────────────────────────
print("Init generation pipeline...", flush=True)
_chars_gen = CharsImageGenerator('small_new_energy')
_tg = LicensePlateImageGenerator('small_new_energy')
_tmpl = _tg.generate_template_image(_chars_gen.plate_width, _chars_gen.plate_height)
_aug = ImageAugmentation('small_new_energy', _tmpl)
_aug.env_data_paths = [str(ROOT / p) for p in _aug.env_data_paths]
_smu = str(ROOT / 'images' / 'smu.jpg')
if os.path.exists(_smu):
    _aug.smu = cv2.imread(_smu)

CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0,0],[CANVAS_W-1,0],[CANVAS_W-1,CANVAS_H-1],[0,CANVAS_H-1]])
LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
DIGITS = list('0123456789')
used_texts = set()

def build_plate(text):
    ci = _chars_gen.generate_images([text])[0]
    ai = _aug.augment(ci, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(ai, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

def make_new_text(province):
    while True:
        t = province + random.choice(LETTERS) + random.choice(['D','F']) \
            + ''.join(random.choice(DIGITS) for _ in range(5))
        if t not in used_texts:
            used_texts.add(t)
            return t

# ── CV transfer — L-ONLY ───────────────────────────────────────
def estimate_sharpness(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def estimate_noise(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = cv2.boxFilter(gray, -1, (7,7), normalize=False)
    local_mean = cv2.boxFilter(gray, -1, (7,7), normalize=True)
    local_var = local_std - local_mean * local_mean * 49
    local_var = np.clip(local_var, 0, None)
    noise = float(np.median(np.sqrt(local_var)))
    return max(noise, 0.5)

def transfer_cv_properties_lonly(plate_bgr, orig_patch_bgr):
    """
    Transfer ONLY luminance/capture-degradation from original plate to generated green plate.
    A/B color channels remain from the generated green plate — no blue hue contamination.
    """
    h, w = plate_bgr.shape[:2]
    oh, ow = orig_patch_bgr.shape[:2]
    if (oh, ow) != (h, w):
        orig_resized = cv2.resize(orig_patch_bgr, (w, h), interpolation=cv2.INTER_AREA)
    else:
        orig_resized = orig_patch_bgr.copy()

    # Convert both to LAB
    plate_lab = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    # ── Step 1: L-channel mean/std matching ONLY ──
    L_plate = plate_lab[:,:,0]
    L_orig = orig_lab[:,:,0]
    mn_p, std_p = L_plate.mean(), L_plate.std() + 1e-6
    mn_o, std_o = L_orig.mean(), L_orig.std() + 1e-6
    ratio = max(0.3, min(3.0, std_o / std_p))
    L_adjusted = np.clip((L_plate - mn_p) * ratio + mn_o, 0, 255)
    plate_lab[:,:,0] = L_adjusted

    # Preserve A/B from generated green plate — do NOT touch them
    # plate_lab[:,:,1] and plate_lab[:,:,2] remain unchanged

    result = cv2.cvtColor(plate_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── Step 2: Low-frequency illumination matching (V channel, not H) ──
    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    orig_hsv = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Build illumination maps from V channel
    ill_orig = cv2.resize(cv2.resize(orig_hsv[:,:,2], (8, 4), interpolation=cv2.INTER_AREA),
                          (w, h), interpolation=cv2.INTER_LINEAR)
    ill_result = cv2.resize(cv2.resize(result_hsv[:,:,2], (8, 4), interpolation=cv2.INTER_AREA),
                            (w, h), interpolation=cv2.INTER_LINEAR)

    ill_diff = ill_orig - ill_result
    # Apply illumination only to V channel — not H or S
    result_hsv[:,:,2] = np.clip(result_hsv[:,:,2] + ill_diff * 0.5, 0, 255)
    result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── Step 3: Sharpness matching ──
    sharp_plate = estimate_sharpness(result)
    sharp_orig = estimate_sharpness(orig_resized)
    if sharp_plate > sharp_orig * 1.5:
        ksize = 3
        while ksize <= 15 and estimate_sharpness(result) > sharp_orig * 1.3:
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)
            ksize += 2

    # ── Step 4: Noise transfer ──
    noise_orig = estimate_noise(orig_resized)
    noise_plate = estimate_noise(result)
    if noise_orig > noise_plate * 1.2:
        noise_amt = min(noise_orig - noise_plate, 30)
        noise_map = np.random.randn(h, w, 3).astype(np.float32) * noise_amt
        result = np.clip(result.astype(np.float32) + noise_map, 0, 255).astype(np.uint8)

    return result


# ── Color protection ────────────────────────────────────────────
def enforce_green_plate_color(bgr):
    """
    Check and enforce that the plate is green-dominant, not blue.
    Returns (corrected_bgr, green_ratio, blue_ratio).
    If blue_ratio >= 0.10, fall back to L-only version.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0].astype(np.float32), hsv[:,:,1].astype(np.float32), hsv[:,:,2].astype(np.float32)

    # Green: H 45-100 (OpenCV: 0-180, green ~60-100)
    green_mask = (s > 30) & (v > 40) & ((h >= 45) & (h <= 100))
    # Blue: H 100-140 (OpenCV: blue ~100-130)
    blue_mask = (s > 30) & (v > 40) & ((h >= 100) & (h <= 140))

    total_pixels = bgr.shape[0] * bgr.shape[1]
    green_cnt = int(green_mask.sum())
    blue_cnt = int(blue_mask.sum())
    green_ratio = green_cnt / max(total_pixels, 1)
    blue_ratio = blue_cnt / max(total_pixels, 1)

    return bgr, green_ratio, blue_ratio


# ── Load pose data ──────────────────────────────────────────────
print("Loading pose quad results...", flush=True)
all_pose = [json.loads(l) for l in open(POSE_JSONL)]
print(f"  {len(all_pose)} pose entries", flush=True)

# Load blue manifests for split info
print("Loading blue manifests for split info...", flush=True)
train_fnames = set()
with open(BLUE_TRAIN_MANIFEST) as f:
    for row in csv.DictReader(f):
        train_fnames.add(Path(row['img_path']).name)

test_fnames = set()
with open(BLUE_TEST_MANIFEST) as f:
    for row in csv.DictReader(f):
        test_fnames.add(Path(row['img_path']).name)

for entry in all_pose:
    fname = Path(entry['img_path']).name
    entry['_split'] = 'test' if fname in test_fnames else 'train'

by_subset = defaultdict(lambda: {'train': [], 'val': []})
for entry in all_pose:
    subset = entry['subset']
    by_subset[subset]['val' if entry['_split'] == 'test' else 'train'].append(entry)

# ── Generate ────────────────────────────────────────────────────
print(f"\nGenerating replacements (v2, L-only transfer)...", flush=True)
print(f"  Smoke mode: {args.smoke}", flush=True)
print(f"  Train/prov/subset: {TRAIN_PER_PROV}, Val/prov/subset: {VAL_PER_PROV}", flush=True)

MANIFEST_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'preprocess_group', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source', 'bbox_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio',
]

train_rows = []
val_rows = []
color_failures = []
t0 = time.time()
total_generated = 0
total_errors = 0
total_color_skipped = 0

for subset in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge']:
    for split_name in ['train', 'val']:
        pool = by_subset[subset][split_name]
        per_prov = TRAIN_PER_PROV if split_name == 'train' else VAL_PER_PROV
        if len(pool) == 0 or per_prov == 0:
            continue

        for prov in ALL_PROVINCES:
            selected = random.choices(pool, k=per_prov)
            for src_rec in selected:
                img_path = src_rec['img_path']
                gt_quad_raw = np.array(src_rec['gt_quad'])  # [BR,BL,TL,TR]
                gt_quad = np.array([gt_quad_raw[2], gt_quad_raw[3],
                                    gt_quad_raw[0], gt_quad_raw[1]], dtype=np.float32)
                pose_quad = np.array(src_rec['pose_quad'])  # [TL,TR,BR,BL]

                new_text = make_new_text(prov)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Cannot read {img_path}")
                    h, w = img.shape[:2]

                    # Forward warp → original plate region
                    M_fwd = cv2.getPerspectiveTransform(gt_quad.astype(np.float32), SRC_RECT)
                    plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H),
                                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                    # Capture original CV stats
                    sharp_orig = estimate_sharpness(plate_orig)
                    noise_orig = estimate_noise(plate_orig)

                    # Generate clean green plate
                    raw_plate = build_plate(new_text)

                    # L-only CV transfer
                    processed_plate = transfer_cv_properties_lonly(raw_plate, plate_orig)

                    # Color protection check
                    _, green_ratio, blue_ratio = enforce_green_plate_color(processed_plate)
                    if blue_ratio >= 0.10:
                        total_color_skipped += 1
                        if total_color_skipped <= 5:
                            print(f"  COLOR_SKIP [{subset}/{prov}]: blue_ratio={blue_ratio:.3f} >= 0.10", flush=True)
                        continue  # Skip this sample, don't write to manifest

                    # Paste back
                    M_inv = cv2.getPerspectiveTransform(SRC_RECT, gt_quad.astype(np.float32))
                    warped = cv2.warpPerspective(processed_plate, M_inv, (w, h),
                                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [gt_quad.round().astype(np.int32)], 255)
                    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
                    result = (img.astype(np.float32) * (1 - mask[:,:,None]) +
                              warped.astype(np.float32) * mask[:,:,None]).clip(0, 255).astype(np.uint8)

                    # Save
                    stem = Path(img_path).stem
                    fname = f"{stem}_green_{new_text}.jpg"
                    out_path = (OUT_IMG_DIR_TRAIN if split_name == 'train' else OUT_IMG_DIR_VAL) / fname
                    if not args.skip_generate:
                        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    rel_path = str(out_path.relative_to(ROOT))
                    row = {
                        'img_path': rel_path, 'text': new_text, 'family': 'green8',
                        'source': f'green_ccpd2019_{subset}_cvreplace_v2', 'split': split_name,
                        'preprocess_group': 'ccpd_board', 'has_quad': '1',
                        'can_parse_ccpd_geom': '0', 'can_perspective': '1',
                        'quad_source': 'pose_yolov8n_ccpd2019', 'bbox_source': 'pose_yolov8n_ccpd2019',
                        'quad_1x': f'{pose_quad[0][0]:.1f}', 'quad_1y': f'{pose_quad[0][1]:.1f}',
                        'quad_2x': f'{pose_quad[1][0]:.1f}', 'quad_2y': f'{pose_quad[1][1]:.1f}',
                        'quad_3x': f'{pose_quad[2][0]:.1f}', 'quad_3y': f'{pose_quad[2][1]:.1f}',
                        'quad_4x': f'{pose_quad[3][0]:.1f}', 'quad_4y': f'{pose_quad[3][1]:.1f}',
                        'ocr_crop_mode': 'obb_warp', 'ocr_resize_mode': 'letterbox',
                        'ocr_resize_kernel': 'nn', 'ocr_preproc': 'none',
                        'ocr_channel_order': 'bgr', 'ocr_quad_pad_ratio': '0.0',
                    }
                    if split_name == 'train':
                        train_rows.append(row)
                    else:
                        val_rows.append(row)
                    total_generated += 1

                except Exception as e:
                    total_errors += 1
                    if total_errors <= 10:
                        print(f"  ERROR [{subset}/{prov}]: {img_path}: {e}", flush=True)

print(f"\nTotal generated: {total_generated}, errors: {total_errors}, color_skipped: {total_color_skipped}", flush=True)
print(f"Train: {len(train_rows)}, Val: {len(val_rows)}", flush=True)

# ── Write manifests ────────────────────────────────────────────
for split_name, rows, csv_name in [
    ('train', train_rows, 'train_cvreplace_v2.csv'),
    ('val', val_rows, 'val_cvreplace_v2.csv'),
]:
    out_path = MANIFEST_DIR / csv_name
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Manifest: {out_path} ({len(rows)} rows)", flush=True)

# ── Reports ─────────────────────────────────────────────────────
all_rows = train_rows + val_rows
prov_count = Counter(r['text'][0] for r in all_rows if r['text'])
source_count = Counter(r['source'] for r in all_rows)

print(f"\nProvince distribution:")
for p in ALL_PROVINCES:
    print(f"  {p}: {prov_count.get(p, 0)}", flush=True)

meta = {
    'mode': 'smoke' if args.smoke else 'full',
    'v2_fix': 'L-only CV transfer + green color protection',
    'n_train': len(train_rows), 'n_val': len(val_rows),
    'total_generated': total_generated, 'total_errors': total_errors,
    'color_skipped': total_color_skipped,
    'province_distribution': dict(prov_count),
    'source_distribution': dict(source_count),
}
json.dump(meta, open(OUT_DIR / 'generation_meta_v2.json', 'w'), ensure_ascii=False, indent=2)
print(f"\nMeta: {OUT_DIR / 'generation_meta_v2.json'}", flush=True)
print("Done.", flush=True)
