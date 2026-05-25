#!/usr/bin/env python3
"""
Demo v3: Clean CCPD2020 plate replacement — tight quad mask, no blending artifacts.
"""

import csv, json, os, sys, math, random, shutil
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

# ── Geometry ─────────────────────────────────────────────────────

from load_data import parse_ccpd_quad_from_name, order_quad_points, prepare_board_ocr_input_from_quad_bgr888

def angle_score(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i-1)%4]; b = p[i]; c = p[(i+1)%4]
        v1, v2 = a - b, c - b
        d = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles.append(abs(math.degrees(math.acos(max(-1, min(1, d)))) - 90))
    return float(np.mean(angles))

def board_94x24_from_photo(photo_bgr, quad):
    """Use exact training pipeline: warp_quad_to_rect → letterbox 94×24 NN."""
    prepared, occ, warped, ordered_q, matrix = prepare_board_ocr_input_from_quad_bgr888(
        photo_bgr, quad, 94, 24,
        resize_mode='letterbox',
        resize_kernel='nn',
        preproc_mode='none',
        channel_order='bgr',
        quad_pad_ratio=0.0,
    )
    return prepared  # (24, 94, 3) BGR

# ── Generation pipeline ──────────────────────────────────────────

print("Init generation pipeline...")
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

chars_gen = CharsImageGenerator('small_new_energy')
tg = LicensePlateImageGenerator('small_new_energy')
tmpl = tg.generate_template_image(chars_gen.plate_width, chars_gen.plate_height)
aug = ImageAugmentation('small_new_energy', tmpl)
aug.env_data_paths = [str(ROOT / p) for p in aug.env_data_paths]
smu = str(ROOT / 'images' / 'smu.jpg')
if os.path.exists(smu):
    aug.smu = cv2.imread(smu)

def build_plate(text):
    ci = chars_gen.generate_images([text])[0]
    ai = aug.augment(ci, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(ai, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

# ── Text gen ─────────────────────────────────────────────────────

ALL_PROV = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
            '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
            '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')

def new_text(province, used):
    while True:
        t = province + random.choice(LETTERS) + random.choice(['D','F']) \
            + random.choice(ALNUM) + ''.join(random.choice(DIGITS) for _ in range(4))
        if t not in used:
            used.add(t)
            return t

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

OUT = ROOT / 'tmp/ccpd2020_plate_replace_v3'
if OUT.exists():
    shutil.rmtree(str(OUT))
OUT.mkdir(parents=True)

# Load CCPD2020 test
lab = ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv'
candidates = []
with open(lab, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        q = parse_ccpd_quad_from_name(row['img_path'])
        if q is None: continue
        a = angle_score(q)
        if a > 15:
            candidates.append({'path': row['img_path'], 'text': row['text'], 'angle': a, 'raw_quad': q})

print(f"{len(candidates)} candidates (angle>15)")

# Select: 3 easy (15-20), 3 mid (20-30), 3 high (>30)
def pick(src, n):
    return random.sample(src, min(n, len(src)))
pool = {'easy': [], 'mid': [], 'high': []}
for s in candidates:
    if s['angle'] <= 20: pool['easy'].append(s)
    elif s['angle'] <= 30: pool['mid'].append(s)
    else: pool['high'].append(s)

selected = pick(pool['easy'], 3) + pick(pool['mid'], 3) + pick(pool['high'], 3)
used = set()
results = []

print("\n─── Processing ───")
for idx, s in enumerate(selected):
    prov = s['text'][0]
    other = [p for p in ALL_PROV if p != prov]
    non_ah = [p for p in other if p != '皖']
    tgt = new_text(random.choice(non_ah if non_ah else other), used)
    print(f"[{idx+1}/9] angle={s['angle']:.1f}  {s['text']} → {tgt}")
    
    img = cv2.imread(str(s['path']))
    if img is None: continue
    h, w = img.shape[:2]
    
    # Quad → canonical TL-TR-BR-BL
    quad = order_quad_points(s['raw_quad'])
    
    # 1. Extract original plate
    M_fwd = cv2.getPerspectiveTransform(quad, SRC_RECT)
    plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
    
    # 2. Generate new plate
    plate_new = build_plate(tgt)
    
    # 3. Inverse warp new plate onto photo — tight, no feathering
    M_inv = cv2.getPerspectiveTransform(SRC_RECT, quad)
    warped = cv2.warpPerspective(plate_new, M_inv, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    
    # 4. Tight mask: fill just the quad region
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    # Minimal edge blur (2px) to hide seam
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask_f = mask.astype(np.float32) / 255.0
    
    # 5. Replace — only inside quad
    out = img.astype(np.float32)
    for c in range(3):
        out[:,:,c] = warped[:,:,c] * mask_f + out[:,:,c] * (1.0 - mask_f)
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # 6. Save
    name = f"{Path(s['path']).stem}_replaced_{tgt}.jpg"
    dst = OUT / name
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    results.append({
        'orig_path': s['path'],
        'new_path': str(dst),
        'orig_text': s['text'],
        'new_text': tgt,
        'angle': s['angle'],
        'plate_orig': plate_orig,
        'plate_new': plate_new,
    })

# ── Contact sheet ────────────────────────────────────────────────

print("\n─── Contact sheet ───")

def make_sheet(results, path):
    cols, cell_w, cell_h = 4, 360, 150
    rows = len(results)
    H = rows * cell_h + 80
    W = cols * cell_w + 40
    canvas = Image.new('RGB', (W, H), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    
    font = ImageFont.load_default()
    for fp in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if os.path.exists(fp):
            try: font = ImageFont.truetype(fp, 13); break
            except: pass
    try: tf = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 22)
    except: tf = ImageFont.load_default()
    
    draw.text((10, 8), "CCPD2020 Plate Replacement v3 — Tight Quad Only", fill=(220,220,220), font=tf)
    for ci, hdr in enumerate(['Original Photo', 'Orig 94×24', 'New Plate 246×72', 'New 94×24']):
        draw.text((20 + ci * cell_w, 48), hdr, fill=(180,200,255), font=font)
    
    for ri, r in enumerate(results):
        y0 = ri * cell_h + 80
        
        # Orig photo thumbnail
        ip = Image.open(r['orig_path']).convert('RGB')
        ip.thumbnail((cell_w - 20, cell_h - 30))
        canvas.paste(ip, (15, y0 + 5))
        
        # Orig 94×24
        ob = board_94x24(r['plate_orig'])
        ob_im = Image.fromarray(cv2.cvtColor(ob, cv2.COLOR_BGR2RGB))
        ob_im = ob_im.resize((cell_w - 20, cell_h - 30), Image.NEAREST)
        canvas.paste(ob_im, (cell_w + 15, y0 + 5))
        
        # New plate 246×72 (scaled up for visibility)
        np_im = Image.fromarray(cv2.cvtColor(r['plate_new'], cv2.COLOR_BGR2RGB))
        np_im = np_im.resize((cell_w - 20, cell_h - 30), Image.LANCZOS)
        canvas.paste(np_im, (cell_w * 2 + 15, y0 + 5))
        
        # New 94×24
        nb = board_94x24(r['plate_new'])
        nb_im = Image.fromarray(cv2.cvtColor(nb, cv2.COLOR_BGR2RGB))
        nb_im = nb_im.resize((cell_w - 20, cell_h - 30), Image.NEAREST)
        canvas.paste(nb_im, (cell_w * 3 + 15, y0 + 5))
        
        lbl = f"angle={r['angle']:.1f}  {r['orig_text']} → {r['new_text']}"
        draw.text((15, y0 + cell_h - 18), lbl, fill=(200,220,255), font=font)
    
    canvas.save(path, quality=95)
    return path

cs = OUT / 'contact_sheet_v3.jpg'
make_sheet(results, str(cs))
print(f"Contact sheet: {cs}")

# Copy to Windows
win = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2020_plate_replace_v3')
if win.exists():
    shutil.rmtree(str(win))
win.mkdir(parents=True)
shutil.copy2(str(cs), str(win / cs.name))
for r in results:
    shutil.copy2(r['new_path'], str(win / Path(r['new_path']).name))
print(f"Windows QA: {win}")

# Metadata
json.dump([{
    'orig': r['orig_path'], 'new': r['new_path'],
    'orig_text': r['orig_text'], 'new_text': r['new_text'],
    'angle': r['angle'],
} for r in results], open(OUT / 'meta.json', 'w'), ensure_ascii=False, indent=2)

print("\nDONE")
