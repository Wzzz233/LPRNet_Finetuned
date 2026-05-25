#!/usr/bin/env python3
"""Step 1: Generate green plate replacement data from CCPD2019 tilt/db/challenge.
- Uses GT quad for text placement (warp source plate → generate green → paste back)
- Uses Pose quad in manifest for training input
- CV characteristics transferred from original plate region
- Province-balanced text generation (31 provinces)"""

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

# ── CLI ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true', help='Smoke mode: 2 train + 1 val per province per subset')
parser.add_argument('--train_per_province_per_subset', type=int, default=200, help='Full mode samples per province per subset for train')
parser.add_argument('--val_per_province_per_subset', type=int, default=20, help='Full mode samples per province per subset for val')
parser.add_argument('--skip_generate', action='store_true', help='Skip generation, only produce manifest from existing images')
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
OUT_DIR = ROOT / 'datasets' / f'green_ccpd2019_tilt_db_challenge_cvreplace_{DATE_TAG}'
MANIFEST_DIR = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_{DATE_TAG}'
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/green_ccpd2019_cvreplace')

OUT_IMG_DIR_TRAIN = OUT_DIR / 'images' / 'train'
OUT_IMG_DIR_VAL = OUT_DIR / 'images' / 'val'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMG_DIR_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_IMG_DIR_VAL.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
WIN_QA.mkdir(parents=True, exist_ok=True)

random.seed(20260508)

# ── Province setup ──────────────────────────────────────────────
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
ALNUM = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')

used_texts = set()

def build_plate(text):
    ci = _chars_gen.generate_images([text])[0]
    ai = _aug.augment(ci, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(ai, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

def make_new_text(province):
    while True:
        # Green plate: province + letter + D/F + 5 alnum
        t = province + random.choice(LETTERS) + random.choice(['D','F']) \
            + ''.join(random.choice(DIGITS) for _ in range(5))
        if t not in used_texts:
            used_texts.add(t)
            return t

# ── CV transfer functions ───────────────────────────────────────
def extract_lab_stats(patch_bgr):
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    stats = {}
    for i, ch in enumerate(['L', 'A', 'B']):
        stats[f'{ch}_mean'] = float(lab[:,:,i].mean())
        stats[f'{ch}_std'] = float(lab[:,:,i].std())
    return stats, lab

def extract_illumination_map(lab_l_channel, dst_size=(CANVAS_W, CANVAS_H)):
    """Extract low-frequency illumination from L channel."""
    # Resize to very small then back up to get low-frequency approximation
    small = cv2.resize(lab_l_channel, (8, 4), interpolation=cv2.INTER_AREA)
    ill_map = cv2.resize(small, dst_size, interpolation=cv2.INTER_LINEAR)
    return ill_map

def estimate_sharpness(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def estimate_noise(bgr):
    """Estimate Gaussian noise std from high-frequency components."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Use median of local std in 7x7 patches
    local_std = cv2.boxFilter(gray, -1, (7,7), normalize=False)
    local_mean = cv2.boxFilter(gray, -1, (7,7), normalize=True)
    local_var = local_std - local_mean * local_mean * 49
    local_var = np.clip(local_var, 0, None)
    local_std2 = np.sqrt(local_var)
    # Take median of local std as noise estimate
    noise = float(np.median(local_std2))
    return max(noise, 0.5)

def transfer_cv_properties(plate_bgr, orig_patch_bgr):
    """Transfer CV properties from original plate patch to generated plate."""
    h, w = plate_bgr.shape[:2]
    oh, ow = orig_patch_bgr.shape[:2]
    
    # 1. Resize orig patch to match plate size
    if (oh, ow) != (h, w):
        orig_resized = cv2.resize(orig_patch_bgr, (w, h), interpolation=cv2.INTER_AREA)
    else:
        orig_resized = orig_patch_bgr.copy()
    
    # 2. LAB mean/std matching per channel
    plate_lab = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    for i in range(3):
        p_ch = plate_lab[:,:,i]
        o_ch = orig_lab[:,:,i]
        p_mean, p_std = p_ch.mean(), p_ch.std() + 1e-6
        o_mean, o_std = o_ch.mean(), o_ch.std() + 1e-6
        ratio = max(0.3, min(3.0, o_std / p_std))
        plate_lab[:,:,i] = np.clip((p_ch - p_mean) * ratio + o_mean, 0, 255)
    
    result = cv2.cvtColor(plate_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # 3. Low-frequency illumination matching
    orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    plate_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    ill_orig = cv2.resize(cv2.resize(orig_gray, (8, 4), interpolation=cv2.INTER_AREA), 
                          (w, h), interpolation=cv2.INTER_LINEAR)
    ill_plate = cv2.resize(cv2.resize(plate_gray, (8, 4), interpolation=cv2.INTER_AREA),
                           (w, h), interpolation=cv2.INTER_LINEAR)
    
    ill_diff = ill_orig - ill_plate
    # Apply illumination difference gently (0.5 strength)
    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    result_hsv[:,:,2] = np.clip(result_hsv[:,:,2] + ill_diff * 0.5, 0, 255)
    result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 4. Sharpness matching
    sharp_plate = estimate_sharpness(result)
    sharp_orig = estimate_sharpness(orig_resized)
    
    if sharp_plate > sharp_orig * 1.5:
        # Need to blur the plate to match orig sharpness
        ksize = 3
        while ksize <= 15 and estimate_sharpness(result) > sharp_orig * 1.3:
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)
            ksize += 2
    
    # 5. Noise estimation and transfer
    noise_orig = estimate_noise(orig_resized)
    noise_plate = estimate_noise(result)
    
    if noise_orig > noise_plate * 1.2:
        # Add Gaussian noise to match
        noise_amt = min(noise_orig - noise_plate, 30)
        noise_map = np.random.randn(h, w, 3).astype(np.float32) * noise_amt
        result = np.clip(result.astype(np.float32) + noise_map, 0, 255).astype(np.uint8)
    
    return result


# ── Load pose data ──────────────────────────────────────────────
print("Loading pose quad results...", flush=True)
all_pose = [json.loads(l) for l in open(POSE_JSONL)]
print(f"  {len(all_pose)} pose entries", flush=True)

# Load blue manifests to determine split
print("Loading blue train/test manifests for split info...", flush=True)
train_fnames = set()
with open(BLUE_TRAIN_MANIFEST) as f:
    for row in csv.DictReader(f):
        fname = Path(row['img_path']).name
        train_fnames.add(fname)

test_fnames = set()
with open(BLUE_TEST_MANIFEST) as f:
    for row in csv.DictReader(f):
        fname = Path(row['img_path']).name
        test_fnames.add(fname)

print(f"  Blue train: {len(train_fnames)} unique, test: {len(test_fnames)} unique", flush=True)

# Assign split to each pose entry
for entry in all_pose:
    fname = Path(entry['img_path']).name
    if fname in test_fnames:
        entry['_split'] = 'test'
    elif fname in train_fnames:
        entry['_split'] = 'train'
    else:
        entry['_split'] = 'train'  # fallback

# Group by subset and split
by_subset = defaultdict(lambda: {'train': [], 'val': []})
for entry in all_pose:
    subset = entry['subset']
    if entry['_split'] == 'test':
        by_subset[subset]['val'].append(entry)
    else:
        by_subset[subset]['train'].append(entry)

print(f"\nPer-subset counts:")
for subset in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge']:
    n_train = len(by_subset[subset]['train'])
    n_val = len(by_subset[subset]['val'])
    print(f"  {subset}: train={n_train} val={n_val}", flush=True)


# ── Generate replacements ──────────────────────────────────────
print(f"\nGenerating replacements...", flush=True)
print(f"  Smoke mode: {args.smoke}", flush=True)
print(f"  Train per province per subset: {TRAIN_PER_PROV}", flush=True)
print(f"  Val per province per subset: {VAL_PER_PROV}", flush=True)

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

# For each subset, for each province, sample source images and generate
t0 = time.time()
total_generated = 0
total_errors = 0

for subset in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge']:
    for split_name in ['train', 'val']:
        pool = by_subset[subset][split_name]
        per_prov = TRAIN_PER_PROV if split_name == 'train' else VAL_PER_PROV
        
        if len(pool) == 0:
            print(f"  SKIP {subset}/{split_name}: empty pool", flush=True)
            continue
        
        for prov in ALL_PROVINCES:
            needed = per_prov
            if needed == 0:
                continue
            
            # Sample needed source images (with replacement if needed)
            selected = random.choices(pool, k=needed)
            
            for src_idx, src_rec in enumerate(selected):
                img_path = src_rec['img_path']
                text = src_rec['text']
                gt_quad_raw = np.array(src_rec['gt_quad'])  # [BR,BL,TL,TR]
                # Reorder to [TL, TR, BR, BL] for warp
                gt_quad = np.array([gt_quad_raw[2], gt_quad_raw[3], 
                                    gt_quad_raw[0], gt_quad_raw[1]], dtype=np.float32)
                pose_quad = np.array(src_rec['pose_quad'])  # already [TL, TR, BR, BL]
                confidence = src_rec['confidence']
                
                # New green plate text
                new_text = make_new_text(prov)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Cannot read {img_path}")
                    h, w = img.shape[:2]
                    
                    # Forward warp (GT quad) → get original plate region
                    M_fwd = cv2.getPerspectiveTransform(gt_quad.astype(np.float32), SRC_RECT)
                    plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H),
                                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    
                    # Extract CV characteristics from original plate region
                    lab_stats, lab_img = extract_lab_stats(plate_orig)
                    illumination = extract_illumination_map(lab_img[:,:,0])
                    sharpness = estimate_sharpness(plate_orig)
                    noise = estimate_noise(plate_orig)
                    
                    # Generate clean green plate
                    raw_plate = build_plate(new_text)
                    
                    # Transfer CV properties from original to generated
                    processed_plate = transfer_cv_properties(raw_plate, plate_orig)
                    
                    # Verify processed plate still looks reasonable
                    final_plate = processed_plate
                    
                    # Inverse warp back (GT quad) → paste onto original image
                    M_inv = cv2.getPerspectiveTransform(SRC_RECT, gt_quad.astype(np.float32))
                    warped = cv2.warpPerspective(final_plate, M_inv, (w, h),
                                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    
                    # Soft mask blend
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [gt_quad.round().astype(np.int32)], 255)
                    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
                    result = (img.astype(np.float32) * (1 - mask[:,:,None]) +
                              warped.astype(np.float32) * mask[:,:,None]).clip(0, 255).astype(np.uint8)
                    
                    # Save
                    stem = Path(img_path).stem
                    fname = f"{stem}_green_{new_text}.jpg"
                    if split_name == 'train':
                        out_path = OUT_IMG_DIR_TRAIN / fname
                    else:
                        out_path = OUT_IMG_DIR_VAL / fname
                    
                    if not args.skip_generate:
                        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    rel_path = str(out_path.relative_to(ROOT))
                    
                    # Manifest row — use POSE quad for training
                    row = {
                        'img_path': rel_path,
                        'text': new_text,
                        'family': 'green8',
                        'source': f'green_ccpd2019_{subset}_cvreplace_v1',
                        'split': split_name,
                        'preprocess_group': 'ccpd_board',
                        'has_quad': '1',
                        'can_parse_ccpd_geom': '0',
                        'can_perspective': '1',
                        'quad_source': 'pose_yolov8n_ccpd2019',
                        'bbox_source': 'pose_yolov8n_ccpd2019',
                        'quad_1x': f'{pose_quad[0][0]:.1f}',
                        'quad_1y': f'{pose_quad[0][1]:.1f}',
                        'quad_2x': f'{pose_quad[1][0]:.1f}',
                        'quad_2y': f'{pose_quad[1][1]:.1f}',
                        'quad_3x': f'{pose_quad[2][0]:.1f}',
                        'quad_3y': f'{pose_quad[2][1]:.1f}',
                        'quad_4x': f'{pose_quad[3][0]:.1f}',
                        'quad_4y': f'{pose_quad[3][1]:.1f}',
                        'ocr_crop_mode': 'obb_warp',
                        'ocr_resize_mode': 'letterbox',
                        'ocr_resize_kernel': 'nn',
                        'ocr_preproc': 'none',
                        'ocr_channel_order': 'bgr',
                        'ocr_quad_pad_ratio': '0.0',
                    }
                    
                    if split_name == 'train':
                        train_rows.append(row)
                    else:
                        val_rows.append(row)
                    
                    total_generated += 1
                    
                except Exception as e:
                    total_errors += 1
                    if total_errors <= 10:
                        print(f"  ERROR [{subset}/{split_name}/{prov}]: {img_path}: {e}", flush=True)
                
                if (total_generated + total_errors) % 500 == 0 and total_generated > 0:
                    elapsed = time.time() - t0
                    rate = total_generated / elapsed
                    print(f"  Generated {total_generated} ({total_errors} err) "
                          f"in {elapsed/60:.1f}min ({rate:.1f}/s)", flush=True)

print(f"\nTotal generated: {total_generated}, errors: {total_errors}", flush=True)
print(f"Train: {len(train_rows)}, Val: {len(val_rows)}", flush=True)

# ── Write manifests ────────────────────────────────────────────
for split_name, rows, csv_name in [
    ('train', train_rows, 'train_cvreplace.csv'),
    ('val', val_rows, 'val_cvreplace.csv'),
]:
    out_path = MANIFEST_DIR / csv_name
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Manifest: {out_path} ({len(rows)} rows)", flush=True)

# ── Province distribution report ────────────────────────────────
all_rows = train_rows + val_rows
prov_count = Counter(r['text'][0] for r in all_rows if r['text'])
source_count = Counter(r['source'] for r in all_rows)

print(f"\nProvince distribution:")
for p in ALL_PROVINCES:
    print(f"  {p}: {prov_count.get(p, 0)}", flush=True)
print(f"  Total: {sum(prov_count.values())}", flush=True)
print(f"\nSource distribution: {dict(source_count)}", flush=True)

# ── Save generation metadata ────────────────────────────────────
meta = {
    'mode': 'smoke' if args.smoke else 'full',
    'n_train': len(train_rows),
    'n_val': len(val_rows),
    'province_distribution': dict(prov_count),
    'source_distribution': dict(source_count),
    'samples_per_province_per_subset_train': TRAIN_PER_PROV,
    'samples_per_province_per_subset_val': VAL_PER_PROV,
    'subsets': ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge'],
    'plate_type': 'small_new_energy',
    'cv_transfer': 'LAB_mean_std + illumination + sharpness + noise',
    'quad_source': 'pose_yolov8n_ccpd2019',
    'paste_quad': 'GT_quad',
    'total_generated': total_generated,
    'total_errors': total_errors,
}
json.dump(meta, open(OUT_DIR / 'generation_meta.json', 'w'), ensure_ascii=False, indent=2)
print(f"\nGeneration meta: {OUT_DIR / 'generation_meta.json'}", flush=True)
print("Done.", flush=True)
