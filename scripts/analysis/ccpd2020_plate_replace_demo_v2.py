#!/usr/bin/env python3
"""
Demo v2: CCPD2020 plate replacement using the established generation pipeline.
- Uses CharsImageGenerator + ImageAugmentation for clean plate rendering
- Correct forward/inverse warp
- Color matching to original plate
"""

import csv, json, os, sys, math, random, shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/utils'))

# ── Geometry ─────────────────────────────────────────────────────

def angle_score_from_quad(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i - 1) % 4]
        b = p[i]
        c = p[(i + 1) % 4]
        v1 = a - b
        v2 = c - b
        dot = np.dot(v1, v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            angles.append(0)
        else:
            cos_angle = dot / (n1 * n2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angles.append(abs(math.degrees(math.acos(cos_angle)) - 90))
    return float(np.mean(angles))

CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0, 0], [CANVAS_W-1, 0], [CANVAS_W-1, CANVAS_H-1], [0, CANVAS_H-1]])

# ── Initialize generation pipeline ───────────────────────────────

print("Initializing plate generation pipeline...")
# Import established generators
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

chars_gen = CharsImageGenerator('small_new_energy')
template_gen = LicensePlateImageGenerator('small_new_energy')
template = template_gen.generate_template_image(chars_gen.plate_width, chars_gen.plate_height)
augmenter = ImageAugmentation('small_new_energy', template)
augmenter.env_data_paths = [str(ROOT / p) for p in augmenter.env_data_paths]
smu_path = str(ROOT / 'images' / 'smu.jpg')
if os.path.exists(smu_path):
    augmenter.smu = cv2.imread(smu_path)

def build_base_plate(text):
    """Generate a clean green plate at standard resolution 246x72."""
    char_img = chars_gen.generate_images([text])[0]
    img = augmenter.augment(char_img, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(img, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

# ── Replacement pipeline ─────────────────────────────────────────

ALL_PROVINCES = [
    '京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
    '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
    '桂','琼','川','贵','云','藏','陕','甘','青','宁','新',
]
LETTERS_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')

def generate_text(province=None, used_texts=None):
    if province is None:
        province = random.choice(ALL_PROVINCES)
    while True:
        third = random.choice(['D', 'F'])
        text = (province + random.choice(LETTERS_NO_IO) + third
                + random.choice(ALNUM_NO_IO) + ''.join(random.choice(DIGITS) for _ in range(4)))
        if used_texts is None or text not in used_texts:
            if used_texts is not None:
                used_texts.add(text)
            return text

def color_stats(patch):
    """Per-channel mean + std. Returns lists for BGR."""
    return {
        'mean': [float(patch[:,:,c].mean()) for c in range(3)],
        'std': [float(patch[:,:,c].std()) + 1e-6 for c in range(3)],
    }

def match_color(src_bgr, target_stats, strength=0.6):
    """Match src image color stats to target, blended by strength."""
    src = src_bgr.astype(np.float32)
    out = np.zeros_like(src)
    for c in range(3):
        s_mean = src[:,:,c].mean()
        s_std = src[:,:,c].std() + 1e-6
        t_mean = target_stats['mean'][c]
        t_std = target_stats['std'][c]
        matched = (src[:,:,c] - s_mean) / s_std * t_std + t_mean
        out[:,:,c] = matched * strength + src[:,:,c] * (1.0 - strength)
    return np.clip(out, 0, 255).astype(np.uint8)

def replace_plate(photo_path_str, new_text, out_dir, idx):
    photo_path = Path(photo_path_str)
    img = cv2.imread(str(photo_path))
    if img is None:
        return None, f"cannot read {photo_path}"
    
    from load_data import parse_ccpd_quad_from_name, order_quad_points
    raw_quad = parse_ccpd_quad_from_name(str(photo_path))
    if raw_quad is None:
        return None, f"no quad in {photo_path.name}"
    
    # Reorder to [TL, TR, BR, BL] clockwise (required for perspective transform)
    quad = order_quad_points(raw_quad)
    
    h, w = img.shape[:2]
    
    # 2. Forward warp: extract original plate region to 246×72
    M_fwd = cv2.getPerspectiveTransform(quad, SRC_RECT)
    plate_patch = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H),
                                       flags=cv2.INTER_LINEAR)
    
    # 3. Generate clean new plate using established pipeline (no color matching)
    new_plate = build_base_plate(new_text)  # 246×72 BGR
    
    # 4. Inverse warp: place new plate back into photo (same quad)
    M_inv = cv2.getPerspectiveTransform(SRC_RECT, quad)
    warped_plate = cv2.warpPerspective(new_plate, M_inv, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
    
    # 5. Create feathered mask: white square in SRC_RECT → warp to photo quad
    mask_plate = np.ones((CANVAS_H, CANVAS_W), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask_plate, M_inv, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))
    
    # 6. Feather mask edges slightly
    ksize = 5
    warped_mask = cv2.GaussianBlur(warped_mask, (ksize, ksize), 0)
    warped_mask = warped_mask.astype(np.float32) / 255.0
    warped_mask = np.clip(warped_mask, 0, 1)
    
    # 7. Blend only in mask region
    result = img.astype(np.float32)
    for c in range(3):
        result[:,:,c] = result[:,:,c] * (1.0 - warped_mask) + warped_plate[:,:,c] * warped_mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 8. Save
    stem = photo_path.stem
    out_name = f"{stem}_replaced_{new_text}.jpg"
    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 9. Generate 94×24 board view for comparison
    scale = 24.0 / CANVAS_H
    new_w = int(CANVAS_W * scale)
    orig_board = cv2.resize(plate_patch, (new_w, 24), interpolation=cv2.INTER_NEAREST)
    new_board = cv2.resize(new_plate, (new_w, 24), interpolation=cv2.INTER_NEAREST)
    
    def to_94x24(board_24w):
        pad = np.zeros((24, 94, 3), dtype=np.uint8)
        x_off = (94 - board_24w.shape[1]) // 2
        pad[:, x_off:x_off+board_24w.shape[1]] = board_24w
        return pad
    
    return {
        'orig_path': str(photo_path),
        'new_path': str(out_path),
        'new_text': new_text,
        'angle_score': angle_score_from_quad(quad),
        'orig_94x24': to_94x24(orig_board),
        'new_94x24': to_94x24(new_board),
    }, None

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

OUT_DIR = ROOT / 'tmp/ccpd2020_plate_replace_demo_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading CCPD2020 test samples...")
from load_data import parse_ccpd_quad_from_name

test_labels = ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv'
samples = []
with open(test_labels, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        quad = parse_ccpd_quad_from_name(row['img_path'])
        if quad is None:
            continue
        angle = angle_score_from_quad(quad)
        if angle > 15:
            samples.append({'img_path': row['img_path'], 'text': row['text'], 'angle': angle})

print(f"  {len(samples)} samples with angle > 15")

# Stratified selection
bins = {'15-20': [], '20-30': [], '30+': []}
for s in samples:
    if s['angle'] <= 20: bins['15-20'].append(s)
    elif s['angle'] <= 30: bins['20-30'].append(s)
    else: bins['30+'].append(s)

selected = []
for bin_name in ['15-20', '20-30', '30+']:
    bin_samples = bins[bin_name]
    n_take = min(3, len(bin_samples))
    selected.extend(random.sample(bin_samples, n_take))

print(f"Selected {len(selected)} samples:")
used_texts = set()
results = []

for idx, s in enumerate(selected):
    print(f"\n[{idx+1}/{len(selected)}] angle={s['angle']:.1f}  orig={s['text']}")
    
    # Generate new text with different province
    orig_prov = s['text'][0]
    other_provs = [p for p in ALL_PROVINCES if p != orig_prov]
    # Mix non-Anhui provinces when possible
    non_ah = [p for p in other_provs if p != '皖']
    prov_pool = non_ah if non_ah else other_provs
    new_text = generate_text(province=random.choice(prov_pool), used_texts=used_texts)
    print(f"  new={new_text}")
    
    result, err = replace_plate(s['img_path'], new_text, OUT_DIR, idx+1)
    if err:
        print(f"  ERROR: {err}")
        continue
    
    results.append(result)
    print(f"  saved: {Path(result['new_path']).name}")

# ── Contact sheet ────────────────────────────────────────────────

print("\nGenerating contact sheet...")

def make_contact_sheet(results, save_path):
    cols = 4  # orig photo, orig 94×24, new photo, new 94×24
    cell_w, cell_h = 350, 150
    margin = 50
    rows = len(results)
    
    canvas = Image.new('RGB', (cols * cell_w + margin, rows * cell_h + 80), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    
    # Try CJK font
    font = ImageFont.load_default()
    for fp in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 14)
                break
            except:
                pass
    
    try:
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 24)
    except:
        title_font = ImageFont.load_default()
    
    draw.text((10, 10), "CCPD2020 Plate Replacement v2 — Established Generation Pipeline", fill=(200,200,200), font=title_font)
    
    headers = ['Original Photo', 'Original 94×24', 'Replaced Photo', 'Replaced 94×24']
    for ci, h in enumerate(headers):
        x = margin + ci * cell_w + 30
        draw.text((x, 50), h, fill=(180, 200, 255), font=font)
    
    for ri, r in enumerate(results):
        y0 = ri * cell_h + 80
        
        # Load original photo
        orig_img = Image.open(r['orig_path']).convert('RGB')
        orig_img.thumbnail((cell_w - 20, cell_h - 30))
        
        # Orig 94×24
        orig_94 = Image.fromarray(r['orig_94x24'])
        orig_94 = orig_94.resize((cell_w - 20, cell_h - 30), Image.NEAREST)
        
        # New photo
        new_img = Image.open(r['new_path']).convert('RGB')
        new_img.thumbnail((cell_w - 20, cell_h - 30))
        
        # New 94×24
        new_94 = Image.fromarray(r['new_94x24'])
        new_94 = new_94.resize((cell_w - 20, cell_h - 30), Image.NEAREST)
        
        # Paste
        canvas.paste(orig_img, (margin + 5, y0 + 5))
        canvas.paste(orig_94, (margin + cell_w + 5, y0 + 5))
        canvas.paste(new_img, (margin + cell_w * 2 + 5, y0 + 5))
        canvas.paste(new_94, (margin + cell_w * 3 + 5, y0 + 5))
        
        label = f"angle={r['angle_score']:.1f}  new={r['new_text']}"
        draw.text((margin + 5, y0 + cell_h - 18), label, fill=(200, 220, 255), font=font)
    
    canvas.save(save_path, quality=95)
    return save_path

cs_path = OUT_DIR / 'contact_sheet_v2.jpg'
make_contact_sheet(results, str(cs_path))
print(f"Contact sheet: {cs_path}")

# Copy to Windows
win_qa = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2020_plate_replace_demo_v2')
win_qa.mkdir(parents=True, exist_ok=True)
shutil.copy2(str(cs_path), str(win_qa / cs_path.name))
for r in results:
    shutil.copy2(r['new_path'], str(win_qa / Path(r['new_path']).name))
print(f"Copied to Windows: {win_qa}")

# Metadata
meta = [{
    'orig_path': r['orig_path'],
    'new_path': r['new_path'],
    'new_text': r['new_text'],
    'angle_score': r['angle_score'],
} for r in results]
with open(OUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\nDONE")
