#!/usr/bin/env python3
"""
Demo: Replace text on CCPD2020 real green plates with new generated text.
Uses the original photo's quad geometry + real background, only swaps the text.

Pipeline:
1. Pick CCPD2020 test images with high angle_score
2. Extract original plate patch via quad warp
3. Measure color/lighting stats
4. Generate new plate text using project's CharsImageGenerator
5. Color-match new plate to original patch stats
6. Inverse-warp new plate into original image
7. Blending + save with CCPD filename
8. Run through ccpd_board → 94×24 for comparison
"""

import csv, json, os, sys, math, random
from pathlib import Path
from collections import Counter
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/utils'))

from load_data import parse_ccpd_quad_from_name

# Manual ccpd_board pipeline: warp quad → letterbox → 94×24
STD_RECT = np.float32([[0, 0], [245, 0], [245, 71], [0, 71]])
IN_W, IN_H = 94, 24

def warp_to_board(img, quad):
    """Warp from original quad to 246×72, then letterbox to 94×24 NN."""
    M = cv2.getPerspectiveTransform(quad, STD_RECT)
    warped = cv2.warpPerspective(img, M, (246, 72), flags=cv2.INTER_LINEAR)
    # NN letterbox to 94×24
    scale = IN_H / 72.0
    new_w = int(246 * scale)
    resized = cv2.resize(warped, (new_w, IN_H), interpolation=cv2.INTER_NEAREST)
    board = np.zeros((IN_H, IN_W, 3), dtype=np.uint8)
    x_off = (IN_W - new_w) // 2
    board[:, x_off:x_off+new_w] = resized
    return board

OUT = ROOT / 'tmp/ccpd2020_plate_replace_demo'
OUT.mkdir(parents=True, exist_ok=True)

# ── Plate rendering config ───────────────────────────────────────

ALL_PROVINCES = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
]
LETTERS_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM_NO_IO = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')
CANVAS_W, CANVAS_H = 246, 72  # standard plate dimensions

FONT_PATH = str(ROOT / 'font' / 'platech.ttf')
FONT_EN_PATH = str(ROOT / 'font' / 'platechar.ttf')

# ── Geometry metrics ─────────────────────────────────────────────

def angle_score_from_quad(quad):
    p = quad.reshape(4, 2)
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

def parse_e6_quad(image_name):
    """E1mod/E6Aaxis-{bbox}-{quad}... format: quad = parts[2]"""
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) < 3:
        return None
    points_text = parts[2]
    points = []
    try:
        for item in points_text.split('_'):
            if '&' not in item:
                return None
            xs, ys = item.split('&', 1)
            points.append((float(xs), float(ys)))
    except ValueError:
        return None
    if len(points) != 4:
        return None
    return np.asarray(points, dtype=np.float32)

def detect_quad(img_path):
    q = parse_ccpd_quad_from_name(img_path)
    if q is not None:
        return q
    q = parse_e6_quad(img_path)
    if q is not None:
        return q
    return None

# ── Color matching ───────────────────────────────────────────────

def color_stats(patch):
    """Compute mean and std per channel for a BGR patch."""
    return {
        'mean': [float(patch[:,:,c].mean()) for c in range(3)],
        'std': [float(patch[:,:,c].std()) for c in range(3)],
        'mean_gray': float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()),
    }

def match_color(src, target_stats, strength=0.7):
    """
    Match src BGR image to target_stats per channel.
    strength∈[0,1]: 0=no match, 1=full match.
    """
    result = src.astype(np.float32)
    for c in range(3):
        src_mean = result[:,:,c].mean()
        src_std = result[:,:,c].std() + 1e-6
        t_mean = target_stats['mean'][c]
        t_std = target_stats['std'][c] + 1e-6
        # Adjust mean and std
        result[:,:,c] = (result[:,:,c] - src_mean) / src_std * t_std + t_mean
        # Blend original and matched
        result[:,:,c] = result[:,:,c] * strength + src[:,:,c].astype(np.float32) * (1 - strength)
    return np.clip(result, 0, 255).astype(np.uint8)

# ── Generate a new green plate image ─────────────────────────────

def generate_green_plate_text(province=None, used_texts=None):
    """Generate a random valid green plate text."""
    if province is None:
        province = random.choice(ALL_PROVINCES)
    while True:
        third_char = random.choice(['D', 'F'])
        text = (province + random.choice(LETTERS_NO_IO) + third_char 
                + random.choice(ALNUM_NO_IO) + ''.join(random.choice(DIGITS) for _ in range(4)))
        if used_texts is None or text not in used_texts:
            if used_texts is not None:
                used_texts.add(text)
            return text

def render_font_text(text, plate_w=960, plate_h=280):
    """
    Render a Chinese license plate text onto a standardized plate image.
    Uses project fonts. Returns BGR image of size (plate_h, plate_w).
    """
    img = Image.new('RGB', (plate_w, plate_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font_ch = ImageFont.truetype(FONT_PATH, 180)
    except:
        font_ch = ImageFont.load_default()
    try:
        font_en = ImageFont.truetype(FONT_EN_PATH, 180)  
    except:
        font_en = font_ch
    
    # Layout: province + letter + dot + letter + 4 digits
    # For small_new_energy: plate_width=960, char layout
    x = 32  # left_offset
    y = 10  # height_offset
    
    for i, ch in enumerate(text):
        font = font_ch if ord(ch) > 127 else font_en
        bbox = draw.textbbox((x, y), ch, font=font)
        ch_w = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]
        
        if ord(ch) > 127:
            # Chinese character: 180 wide
            draw.text((x, y + (180 - ch_h) // 2), ch, fill=(0, 0, 0), font=font)
            x += 90  # first_char_width
        else:
            draw.text((x, y + (180 - ch_h) // 2), ch, fill=(0, 0, 0), font=font)
            x += 86  # char_width
        
        if i == 1:
            # Draw dot separator
            x += 62  # point_size spacing
        elif i >= 2:
            x += 18  # char_interval
    
    return np.array(img, dtype=np.uint8)[..., ::-1]  # RGB→BGR

def generate_green_plate(text, plate_w=960, plate_h=280, bg_bgr=(60, 140, 60)):
    """
    Generate a full green plate image with white text on green background.
    Returns (plate_h, plate_w, 3) BGR image.
    """
    # Green background
    img = np.full((plate_h, plate_w, 3), bg_bgr, dtype=np.uint8)
    
    # Render font text on white bg
    from PIL import Image, ImageDraw, ImageFont
    text_img = Image.new('RGB', (plate_w, plate_h), (255, 255, 255))
    draw = ImageDraw.Draw(text_img)
    
    try:
        font_ch = ImageFont.truetype(FONT_PATH, 160)
    except:
        font_ch = ImageFont.load_default()
    try:
        font_en = ImageFont.truetype(FONT_EN_PATH, 200)
    except:
        font_en = font_ch
    
    # Layout for green 8-char plate
    x_positions = []
    x = 30
    for i, ch in enumerate(text):
        x_positions.append(x)
        font = font_ch if ord(ch) > 127 else font_en
        bbox = draw.textbbox((0, 0), ch, font=font)
        ch_w = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]
        
        # Center the character vertically
        ty = (plate_h - ch_h) // 2 - 5
        draw.text((x, ty), ch, fill=(0, 0, 0), font=font)
        
        if i == 0:
            x += 130  # province char wider
        elif i == 1:
            x += 100  # letter
        elif i == 2:
            x += 60   # D/F
        elif i == 3:
            x += 85   # alnum
        else:
            x += 82   # digits
        x += 12  # spacing
    
    # Make white text by using the text as mask
    text_np = np.array(text_img, dtype=np.uint8)[..., ::-1]  # to BGR
    gray = cv2.cvtColor(text_np, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Place white text on green background
    result = img.copy()
    result[mask > 0] = (255, 255, 255)
    
    # Add green border like real plates
    cv2.rectangle(result, (4, 4), (plate_w - 5, plate_h - 5), (50, 130, 50), 2)
    
    return result

# ── Main replacement pipeline ────────────────────────────────────

def replace_plate_on_photo(orig_img_path, new_text, out_dir, idx):
    """
    Full pipeline:
    1. Load original photo
    2. Parse quad, extract plate patch
    3. Measure original plate stats
    4. Generate new plate text image
    5. Color-match new plate
    6. Inverse-warp into photo
    7. Blend with mask
    8. Validate with ccpd_board pipeline
    """
    quad = detect_quad(orig_img_path)
    if quad is None:
        print(f"  [{idx}] SKIP: no quad in {orig_img_path.name}")
        return None
    
    orig = cv2.imread(str(orig_img_path))
    if orig is None:
        return None
    
    h, w = orig.shape[:2]
    
    # Step 1: Extract original plate patch
    SRC_RECT = np.float32([[0, 0], [CANVAS_W-1, 0], [CANVAS_W-1, CANVAS_H-1], [0, CANVAS_H-1]])
    M_forward = cv2.getPerspectiveTransform(quad, SRC_RECT)
    plate_patch = cv2.warpPerspective(orig, M_forward, (CANVAS_W, CANVAS_H),
                                       flags=cv2.INTER_LINEAR)
    
    # Step 2: Measure original plate color stats (from the center area of patch)
    cy, cx = CANVAS_H // 2, CANVAS_W // 2
    center_patch = plate_patch[cy-20:cy+20, cx-60:cx+60]
    orig_stats = color_stats(center_patch)
    
    # Step 3: Determine green background color from original patch
    bg_bgr = tuple(int(plate_patch[:,:,c].mean()) for c in range(3))
    bg_bgr = (max(30, bg_bgr[0]), max(30, bg_bgr[1]), max(30, bg_bgr[2]))
    
    # Step 4: Generate new plate
    new_plate_img = generate_green_plate(new_text, bg_bgr=bg_bgr)
    # Resize to match canvas
    new_plate_resized = cv2.resize(new_plate_img, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    
    # Step 5: Color-match the new plate to original stats
    matched_plate = match_color(new_plate_resized, orig_stats, strength=0.7)
    
    # Step 6: Inverse warp into original photo
    M_inv = cv2.getPerspectiveTransform(SRC_RECT, quad)
    warped_plate = cv2.warpPerspective(matched_plate, M_inv, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_TRANSPARENT)
    
    # Step 7: Create mask for blending
    mask_plate = np.ones((CANVAS_H, CANVAS_W), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask_plate, M_inv, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_TRANSPARENT)
    
    # Feather the mask edges
    quad_np = np.array(quad, dtype=np.float32).reshape(4, 2)
    ksize = max(3, int(min(quad_np[:,0].max() - quad_np[:,0].min(), 
                           quad_np[:,1].max() - quad_np[:,1].min()) * 0.02))
    if ksize % 2 == 0:
        ksize += 1
    warped_mask = cv2.GaussianBlur(warped_mask, (ksize, ksize), 0)
    warped_mask = warped_mask.astype(np.float32) / 255.0
    
    # Blend
    result = orig.astype(np.float32)
    for c in range(3):
        result[:,:,c] = result[:,:,c] * (1 - warped_mask) + warped_plate[:,:,c] * warped_mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Step 8: Save
    orig_stem = Path(orig_img_path).stem
    new_name = f"{orig_stem}__replaced_{new_text}.jpg"
    out_path = out_dir / new_name
    cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Step 9: 94×24 board input (manual ccpd_board pipeline)
    ocr_board = warp_to_board(result, quad)
    
    return {
        'orig_path': str(orig_img_path),
        'new_path': str(out_path),
        'new_text': new_text,
        'quad': quad.tolist(),
        'angle_score': angle_score_from_quad(quad),
        'ocr_board_shape': ocr_board.shape,
        'orig_stats': orig_stats,
    }

# ═══════════════════════════════════════════════════════════════
# MAIN DEMO
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("CCPD2020 PLATE REPLACEMENT DEMO")
print("=" * 60)

# Load CCPD2020 test labels
test_labels = ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv'
samples = []
with open(test_labels, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        quad = detect_quad(row['img_path'])
        if quad is None:
            continue
        angle = angle_score_from_quad(quad)
        # Filter: only moderate-hard samples (angle > 15)
        if angle > 15:
            samples.append({
                'img_path': row['img_path'],
                'text': row['text'],
                'angle': angle,
            })
    print(f"\nLoaded {len(samples)} CCPD2020 test samples with angle > 15")

# Pick a diverse set: different angles, mix of provinces
# Group by angle bins
bins = {'low_15_20': [], 'mid_20_30': [], 'high_30+': []}
for s in samples:
    if s['angle'] <= 20:
        bins['low_15_20'].append(s)
    elif s['angle'] <= 30:
        bins['mid_20_30'].append(s)
    else:
        bins['high_30+'].append(s)

print(f"\nAngle bins:")
for k, v in bins.items():
    print(f"  {k}: {len(v)} samples")

# Pick 2 from each bin
selected = []
for bin_name, bin_samples in bins.items():
    if len(bin_samples) >= 2:
        selected.extend(random.sample(bin_samples, 2))
    elif bin_samples:
        selected.extend(bin_samples)

# If not enough, fill remaining with the highest angle
if len(selected) < 6:
    remaining = [s for s in sorted(samples, key=lambda x: -x['angle']) if s not in selected]
    selected.extend(remaining[:6-len(selected)])

print(f"\nSelected {len(selected)} samples for demo:")
for s in selected:
    print(f"  {Path(s['img_path']).name}  text={s['text']}  angle={s['angle']:.1f}")

# Create output dirs
demo_dir = OUT / 'demo_output'
demo_dir.mkdir(parents=True, exist_ok=True)

used_texts = set()
results = []

for idx, s in enumerate(selected):
    print(f"\n{'─'*50}")
    print(f"[{idx+1}/{len(selected)}] {Path(s['img_path']).name}")
    print(f"  Original text: {s['text']}  angle={s['angle']:.1f}")
    
    # Generate new text (different province, avoid original)
    orig_province = s['text'][0] if s['text'] else '皖'
    other_provinces = [p for p in ALL_PROVINCES if p != orig_province]
    new_province = random.choice(other_provinces)
    new_text = generate_green_plate_text(province=new_province, used_texts=used_texts)
    print(f"  New text: {new_text}")
    
    # Run replacement
    r = replace_plate_on_photo(
        s['img_path'], new_text, demo_dir, idx+1
    )
    if r:
        results.append(r)
        print(f"  Saved: {Path(r['new_path']).name}")
        if r['ocr_board_shape']:
            print(f"  OCR board shape: {r['ocr_board_shape']}")

# ── Generate comparison contact sheet ────────────────────────────

print("\n" + "=" * 60)
print("GENERATING COMPARISON CONTACT SHEET")
print("=" * 60)

def generate_contact_sheet(selected, results, save_path):
    """Generate a side-by-side comparison: original → replaced → ccpd_board result."""
    n = len(selected)
    cols = 4  # orig_img, orig_board, replaced_img, replaced_board
    cell_w, cell_h = 320, 140
    
    rows = n
    canvas = Image.new('RGB', (cols * cell_w, rows * cell_h + 50), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 13)
    except:
        font = ImageFont.load_default()
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 22)
    except:
        title_font = ImageFont.load_default()
    
    draw.text((10, 10), "CCPD2020 Plate Replacement Demo — Original vs Replaced", fill=(200,200,200), font=title_font)
    
    # Column headers
    headers = ['Original Photo', 'Original → 94×24', 'Replaced Photo', 'Replaced → 94×24']
    for ci, h in enumerate(headers):
        draw.text((ci * cell_w + 40, 45), h, fill=(180,200,255), font=font)
    
    for ri, (s, r) in enumerate(zip(selected, results)):
        y0 = ri * cell_h + 70
        
        try:
            # Load original processed through ccpd_board
            orig_img = cv2.imread(s['img_path'])
            orig_quad = detect_quad(s['img_path'])
            SRC_RECT = np.float32([[0, 0], [245, 0], [245, 71], [0, 71]])
            M = cv2.getPerspectiveTransform(orig_quad, SRC_RECT)
            orig_warped = cv2.warpPerspective(orig_img, M, (246, 72), flags=cv2.INTER_LINEAR)
            # 94×24 via letterbox
            scale = 24.0 / 72.0
            new_w = int(246 * scale)
            board = cv2.resize(orig_warped, (new_w, 24), interpolation=cv2.INTER_NEAREST)
            board_pad = np.zeros((24, 94, 3), dtype=np.uint8)
            x_off = (94 - new_w) // 2
            board_pad[:, x_off:x_off+new_w] = board
        except:
            board_pad = np.zeros((24, 94, 3), dtype=np.uint8)
        
        try:
            # Load replaced processed through ccpd_board
            new_img = cv2.imread(r['new_path'])
            M_new = cv2.getPerspectiveTransform(np.array(r['quad'], dtype=np.float32), SRC_RECT)
            new_warped = cv2.warpPerspective(new_img, M_new, (246, 72), flags=cv2.INTER_LINEAR)
            new_board = cv2.resize(new_warped, (new_w, 24), interpolation=cv2.INTER_NEAREST)
            new_board_pad = np.zeros((24, 94, 3), dtype=np.uint8)
            new_board_pad[:, x_off:x_off+new_w] = new_board
        except:
            new_board_pad = np.zeros((24, 94, 3), dtype=np.uint8)
        
        # Column 0: Original photo (thumbnail)
        orig_thumb = cv2.resize(orig_img, (cell_w - 20, cell_h - 20)) if 'orig_img' in dir() else np.zeros((100, 100, 3), dtype=np.uint8)
        orig_pil = Image.fromarray(cv2.cvtColor(orig_thumb, cv2.COLOR_BGR2RGB))
        canvas.paste(orig_pil, (5, y0 + 5))
        
        # Column 1: Original 94×24
        board_pil = Image.fromarray(cv2.cvtColor(board_pad, cv2.COLOR_BGR2RGB))
        board_pil = board_pil.resize((cell_w - 20, cell_h - 20), Image.NEAREST)
        canvas.paste(board_pil, (cell_w + 5, y0 + 5))
        
        # Column 2: Replaced photo
        new_thumb = cv2.resize(new_img, (cell_w - 20, cell_h - 20))
        new_pil = Image.fromarray(cv2.cvtColor(new_thumb, cv2.COLOR_BGR2RGB))
        canvas.paste(new_pil, (cell_w * 2 + 5, y0 + 5))
        
        # Column 3: Replaced 94×24
        new_board_pil = Image.fromarray(cv2.cvtColor(new_board_pad, cv2.COLOR_BGR2RGB))
        new_board_pil = new_board_pil.resize((cell_w - 20, cell_h - 20), Image.NEAREST)
        canvas.paste(new_board_pil, (cell_w * 3 + 5, y0 + 5))
        
        # Label: angle + text
        label = f"angle={s['angle']:.1f}  {s['text']} → {r['new_text']}"
        draw.text((5, y0 + cell_h - 18), label, fill=(200, 220, 255), font=font)
    
    canvas.save(save_path, quality=95)
    return save_path

# Only generate if we have results
if results:
    cs_path = OUT / 'ccpd2020_plate_replace_demo_contact_sheet.jpg'
    generate_contact_sheet(selected, results, str(cs_path))
    print(f"\nContact sheet saved: {cs_path}")
    
    # Copy to Windows QA
    win_qa = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2020_plate_replace_demo')
    win_qa.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(cs_path), str(win_qa / cs_path.name))
    for r in results:
        shutil.copy2(r['new_path'], str(win_qa / Path(r['new_path']).name))
    print(f"Copied to Windows: {win_qa}")
else:
    print("\nNo results to generate contact sheet.")

# Save metadata
meta_path = OUT / 'demo_metadata.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump([{
        'orig_path': r['orig_path'],
        'new_path': r['new_path'],
        'new_text': r['new_text'],
        'angle_score': r['angle_score'],
        'ocr_board_shape': r['ocr_board_shape'],
    } for r in results], f, ensure_ascii=False, indent=2)
print(f"\nMetadata saved: {meta_path}")
print("\nDONE")
