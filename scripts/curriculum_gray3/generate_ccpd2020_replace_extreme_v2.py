#!/usr/bin/env python3
"""Generate additional 2700 CCPD2020 replacement samples using the SAME photo pool as B2-C,
each with a different province text for multi-province coverage."""
import csv, json, os, sys, math, random, time
from pathlib import Path
from collections import Counter

import numpy as np
import cv2

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'datasets/ccpd2020_replace_extreme_v2_additional'
IMG_DIR = OUT_DIR / 'images'
QA_DIR = OUT_DIR / 'qa'
MANIFEST_DIR = ROOT / 'manifests/ccpd2020_replace_extreme_v2_additional'
OUT_DIR.mkdir(parents=True)
IMG_DIR.mkdir(parents=True)
QA_DIR.mkdir(parents=True)
MANIFEST_DIR.mkdir(parents=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from load_data import parse_ccpd_quad_from_name, order_quad_points, \
    prepare_board_ocr_input_from_quad_bgr888

sys.path.insert(0, str(ROOT / 'src' / 'utils'))
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

# ── Generation pipeline ──────────────────────────────────────────
print("Init...")
_chars_gen = CharsImageGenerator('small_new_energy')
_tg = LicensePlateImageGenerator('small_new_energy')
_tmpl = _tg.generate_template_image(_chars_gen.plate_width, _chars_gen.plate_height)
_aug = ImageAugmentation('small_new_energy', _tmpl)
_aug.env_data_paths = [str(ROOT / p) for p in _aug.env_data_paths]
_smu = str(ROOT / 'images' / 'smu.jpg')
if os.path.exists(_smu):
    _aug.smu = cv2.imread(_smu)

def build_plate(text):
    ci = _chars_gen.generate_images([text])[0]
    ai = _aug.augment(ci, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(ai, (246, 72), interpolation=cv2.INTER_AREA)

def match_brightness(new_bgr, orig_patch_bgr):
    og = cv2.cvtColor(orig_patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ng = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = og.shape
    y1, x1 = h//10, w//10
    roi_h, roi_w = h*8//10, w*8//10
    m_o = og[y1:y1+roi_h, x1:x1+roi_w].mean()
    s_o = og[y1:y1+roi_h, x1:x1+roi_w].std() + 1e-6
    m_n = ng[y1:y1+roi_h, x1:x1+roi_w].mean()
    s_n = ng[y1:y1+roi_h, x1:x1+roi_w].std() + 1e-6
    r = max(0.5, min(2.0, s_o/s_n))
    return cv2.cvtColor(np.clip((ng - m_n) * r + m_o, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# ── Text ─────────────────────────────────────────────────────────
ALL_PROV = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
            '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
            '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
NON_AH = sorted(p for p in ALL_PROV if p != '皖')
LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')

def make_text(province, used):
    while True:
        t = province + random.choice(LETTERS) + random.choice(['D','F']) \
            + random.choice(ALNUM) + ''.join(random.choice(DIGITS) for _ in range(4))
        if t not in used:
            used.add(t)
            return t

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

# Load the B2-C photo pool
print("Loading photo pool...")
b2c_meta = ROOT / 'datasets/ccpd2020_replace_extreme_v1/generation_meta.json'
# Reconstruct from the metadata or scan the B2-C manifest
b2c_train_manifest = ROOT / 'manifests/ccpd2020_replace_extreme_v1/train_B2C_ccpd2020_replace_extreme.csv'

pool = []
used_texts = set()
with open(b2c_train_manifest, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        pool.append(row['img_path'])
        used_texts.add(row['text'])

# Also get the val manifest
b2c_val_manifest = ROOT / 'manifests/ccpd2020_replace_extreme_v1/val_B2C_ccpd2020_replace_extreme.csv'
with open(b2c_val_manifest, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        pool.append(row['img_path'])

print(f"Photo pool: {len(pool)} images")

# Province quotas for ADDITIONAL 2700 samples
# Already used: 皖=25(train)+5(val)=30, others=~89 each
# Additional: 皖=25(train)+5(val)=30 more, others=~89 each
TOTAL_ADD = 2700
TRAIN_N = 2400
VAL_N = 300

AH_Q = 25  # additional Anhui
OTHER_Q = (TOTAL_ADD - AH_Q) // 30  # ~ 89

# Assign provinces
prov_count = Counter()
random.seed(47)  # deterministic
random.shuffle(pool)

# Get the same splits: first 2700 for train, last 300 for val (matching B2-C)
train_pool = pool[:2700]
val_pool = pool[-300:]

assignments = []
for src_pool, split_name, n in [(train_pool, 'train', TRAIN_N), (val_pool, 'val', VAL_N)]:
    for i, p in enumerate(src_pool):
        if len(assignments) >= TOTAL_ADD:
            break
        # Province selection
        if prov_count.get('皖', 0) < AH_Q:
            prov = '皖'
        else:
            avail = [x for x in NON_AH if prov_count.get(x, 0) < OTHER_Q]
            if not avail:
                avail = NON_AH
            prov = random.choice(avail)
        prov_count[prov] += 1
        text = make_text(prov, used_texts)
        assignments.append({'path': p, 'text': text, 'province': prov, 'split': split_name})

print(f"\nAdditional province distribution:")
for p in ALL_PROV:
    print(f"  {p}: {prov_count.get(p, 0)}")

# ── Generate ──────────────────────────────────────────────────────
SRC_RECT = np.float32([[0,0],[245,0],[245,71],[0,71]])

rows_train, rows_val = [], []
t0 = time.time()
n_total = len(assignments)

for idx, item in enumerate(assignments):
    img = cv2.imread(item['path'])
    if img is None:
        continue
    h, w = img.shape[:2]
    
    stem = Path(item['path']).stem
    # Extract orig_stem (remove existing _replaced_ suffix if present)
    if '_replaced_' in stem:
        orig_stem = stem.rsplit('_replaced_', 1)[0]
    else:
        orig_stem = stem
    
    quad = order_quad_points(parse_ccpd_quad_from_name(item['path']))
    
    # Forward warp for brightness ref
    M_fwd = cv2.getPerspectiveTransform(quad, SRC_RECT)
    plate_orig = cv2.warpPerspective(img, M_fwd, (246, 72), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Generate + match
    raw = build_plate(item['text'])
    matched = match_brightness(raw, plate_orig)
    
    # Inverse warp
    M_inv = cv2.getPerspectiveTransform(SRC_RECT, quad)
    warped = cv2.warpPerspective(matched, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    
    result = img.astype(np.float32)
    for c in range(3):
        result[:,:,c] = warped[:,:,c] * mask + result[:,:,c] * (1.0 - mask)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Save
    fname = f"{orig_stem}_replaced_{item['text']}.jpg"
    split_dir = IMG_DIR / item['split']
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / fname
    cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    row = {
        'img_path': str(out_path),
        'text': item['text'],
        'family': 'green8',
        'source': 'green_ccpd2020_replace_extreme_v2',
        'split': item['split'],
        'has_quad': '1',
        'can_parse_ccpd_geom': '1',
        'can_perspective': '1',
        'preprocess_group': 'ccpd_board',
        'ocr_crop_mode': 'obb_warp',
        'ocr_resize_mode': 'letterbox',
        'ocr_resize_kernel': 'nn',
        'ocr_preproc': 'gray3',
        'ocr_channel_order': 'bgr',
        'ocr_quad_pad_ratio': '0.0',
    }
    
    if item['split'] == 'train':
        rows_train.append(row)
    else:
        rows_val.append(row)
    
    if (idx + 1) % 300 == 0 or idx + 1 == n_total:
        elapsed = time.time() - t0
        print(f"  [{idx+1}/{n_total}] {elapsed:.0f}s")

print(f"\nGenerated: {len(rows_train)} train + {len(rows_val)} val")

# Write manifests
FIELDS = ['img_path', 'text', 'family', 'source', 'split',
          'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
          'preprocess_group', 'ocr_crop_mode', 'ocr_resize_mode',
          'ocr_resize_kernel', 'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio']

for split_name, rows in [('train', rows_train), ('val', rows_val)]:
    p = MANIFEST_DIR / f'{split_name}_B2D_additional.csv'
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  {p}: {len(rows)} rows")

# Summary
summary = {
    'n_train': len(rows_train), 'n_val': len(rows_val),
    'province_dist': dict(sorted(Counter(r['text'][0] for r in rows_train + rows_val).items())),
    'base': 'B2-C photo pool (same 3000 CCPD2020 test images, second replacement)',
}
with open(OUT_DIR / 'generation_meta.json', 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Province table
print(f"\nFinal province dist:")
for p in ALL_PROV:
    c = prov_count.get(p, 0)
    print(f"  {p}: {c}")

print(f"\nDONE")
