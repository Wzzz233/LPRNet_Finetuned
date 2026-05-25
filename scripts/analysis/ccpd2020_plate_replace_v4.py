#!/usr/bin/env python3
"""
Demo v4: CCPD2020 plate replacement — 94×24 via exact training pipeline.
Uses prepare_board_ocr_input_from_quad_bgr888 for both orig and replaced.
"""

import csv, json, os, sys, math, random, shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from load_data import parse_ccpd_quad_from_name, order_quad_points, \
    prepare_board_ocr_input_from_quad_bgr888

def angle_score(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i-1)%4]; b = p[i]; c = p[(i+1)%4]
        v1, v2 = a - b, c - b
        d = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angles.append(abs(math.degrees(math.acos(max(-1,min(1,d)))) - 90))
    return float(np.mean(angles))

def board_94x24(img_bgr, quad, preproc='gray3'):
    """Exact training pipeline: warp_quad_to_rect → letterbox 94×24 NN → preproc."""
    prepared, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img_bgr, quad, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode=preproc, channel_order='bgr',
        quad_pad_ratio=0.0,
    )
    return prepared  # (24, 94, 3) BGR


def match_brightness(new_plate_bgr, orig_patch_bgr):
    """Simple brightness/contrast matching: match gray mean+std of new plate to original.
    new_plate_bgr: 246×72 BGR (generated)
    orig_patch_bgr: 246×72 BGR (extracted from photo via warp)
    Returns: 246×72 BGR with matched intensity, still in BGR (3 identical gray channels).
    """
    # Convert both to gray
    orig_gray = cv2.cvtColor(orig_patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    new_gray = cv2.cvtColor(new_plate_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Use center 80% to avoid black border artifacts
    h, w = orig_gray.shape
    cy, cx = h // 2, w // 2
    roi_h, roi_w = int(h * 0.8), int(w * 0.8)
    y1, x1 = cy - roi_h // 2, cx - roi_w // 2
    orig_roi = orig_gray[y1:y1+roi_h, x1:x1+roi_w]
    new_roi = new_gray[y1:y1+roi_h, x1:x1+roi_w]
    
    m_orig = orig_roi.mean()
    s_orig = orig_roi.std() + 1e-6
    m_new = new_roi.mean()
    s_new = new_roi.std() + 1e-6
    
    # Clamp std ratio to avoid extreme adjustments
    ratio = max(0.5, min(2.0, s_orig / s_new))
    
    matched = (new_gray - m_new) * ratio + m_orig
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    
    # Replicate to 3-channel BGR
    return cv2.cvtColor(matched, cv2.COLOR_GRAY2BGR)

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
    return cv2.resize(ai, (246, 72), interpolation=cv2.INTER_AREA)

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

OUT = ROOT / 'tmp/ccpd2020_plate_replace_v4'
if OUT.exists():
    shutil.rmtree(str(OUT))
OUT.mkdir(parents=True)

# Load CCPD2020 test
candidates = []
with open(ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        q = parse_ccpd_quad_from_name(row['img_path'])
        if q is None: continue
        a = angle_score(q)
        if a > 15:
            candidates.append({'path': row['img_path'], 'text': row['text'], 'angle': a, 'raw_quad': q})

print(f"{len(candidates)} candidates (angle>15)")

pool = {'easy': [], 'mid': [], 'high': []}
for s in candidates:
    if s['angle'] <= 20: pool['easy'].append(s)
    elif s['angle'] <= 30: pool['mid'].append(s)
    else: pool['high'].append(s)

sel = (random.sample(pool['easy'], min(3, len(pool['easy']))) +
       random.sample(pool['mid'], min(3, len(pool['mid']))) +
       random.sample(pool['high'], min(3, len(pool['high']))))
used = set()
results = []

print("\n─── Processing ───")
for idx, s in enumerate(sel):
    prov = s['text'][0]
    other = [p for p in ALL_PROV if p != prov]
    non_ah = [p for p in other if p != '皖']
    tgt = new_text(random.choice(non_ah if non_ah else other), used)
    print(f"[{idx+1}/{len(sel)}] angle={s['angle']:.1f}  {s['text']} → {tgt}")

    img = cv2.imread(str(s['path']))
    if img is None: continue
    h, w = img.shape[:2]

    quad = order_quad_points(s['raw_quad'])

    # === Forward warp: extract original plate patch for brightness reference ===
    rect = np.float32([[0,0],[245,0],[245,71],[0,71]])
    M_fwd = cv2.getPerspectiveTransform(quad, rect)
    plate_orig_patch = cv2.warpPerspective(img, M_fwd, (246, 72),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REPLICATE)

    # === Orig 94×24 via training pipeline (color + gray3) ===
    orig_94_color = board_94x24(img, quad, preproc='none')
    orig_94_gray = board_94x24(img, quad, preproc='gray3')

    # === Generate new plate ===
    plate_new_raw = build_plate(tgt)

    # === Brightness match new plate to original patch ===
    plate_new = match_brightness(plate_new_raw, plate_orig_patch)

    # === Inverse warp new plate onto photo ===
    M_inv = cv2.getPerspectiveTransform(rect, quad)
    warped_new = cv2.warpPerspective(plate_new, M_inv, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)

    # Tight mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0

    result = img.astype(np.float32)
    for c in range(3):
        result[:,:,c] = warped_new[:,:,c] * mask + result[:,:,c] * (1.0 - mask)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # === New 94×24 via same training pipeline (color + gray3) ===
    new_94_color = board_94x24(result, quad, preproc='none')
    new_94_gray = board_94x24(result, quad, preproc='gray3')

    # Save replaced photo
    name = f"{Path(s['path']).stem}_replaced_{tgt}.jpg"
    cv2.imwrite(str(OUT / name), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    results.append({
        'orig_path': s['path'],
        'new_path': str(OUT / name),
        'orig_text': s['text'],
        'new_text': tgt,
        'angle': s['angle'],
        'orig_94_color': orig_94_color,
        'orig_94_gray': orig_94_gray,
        'new_94_color': new_94_color,
        'new_94_gray': new_94_gray,
    })

# ── Contact sheet ────────────────────────────────────────────────

print("\n─── Contact sheet ───")

def make_sheet(results, path):
    cols, cell_w, cell_h = 6, 310, 140
    rows = len(results)
    H = rows * cell_h + 80
    W = cols * cell_w + 40
    canvas = Image.new('RGB', (W, H), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.load_default()
    for fp in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if os.path.exists(fp):
            try: font = ImageFont.truetype(fp, 11); break
            except: pass
    try: tf = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 20)
    except: tf = ImageFont.load_default()

    draw.text((10, 8), "CCPD2020 Plate Replacement v4 — Color + Gray3 94×24 (train pipeline)", fill=(220,220,220), font=tf)
    for ci, hdr in enumerate(['Original Photo',
          'Orig 94×24 (color)', 'Orig 94×24 (gray3)',
          'Replaced Photo',
          'New 94×24 (color)', 'New 94×24 (gray3)']):
        draw.text((15 + ci * cell_w, 48), hdr, fill=(180,200,255), font=font)

    for ri, r in enumerate(results):
        y0 = ri * cell_h + 80
        items = [
            (Image.open(r['orig_path']).convert('RGB'), 'photo'),
            (Image.fromarray(cv2.cvtColor(r['orig_94_color'], cv2.COLOR_BGR2RGB)), 'board'),
            (Image.fromarray(cv2.cvtColor(r['orig_94_gray'], cv2.COLOR_BGR2RGB)), 'board'),
            (Image.open(r['new_path']).convert('RGB'), 'photo'),
            (Image.fromarray(cv2.cvtColor(r['new_94_color'], cv2.COLOR_BGR2RGB)), 'board'),
            (Image.fromarray(cv2.cvtColor(r['new_94_gray'], cv2.COLOR_BGR2RGB)), 'board'),
        ]
        for ci, (im, typ) in enumerate(items):
            if typ == 'photo':
                im.thumbnail((cell_w - 20, cell_h - 30))
            else:
                im = im.resize((cell_w - 20, cell_h - 30), Image.NEAREST)
            canvas.paste(im, (15 + ci * cell_w, y0 + 5))

        lbl = f"angle={r['angle']:.1f}  {r['orig_text']} → {r['new_text']}"
        draw.text((15, y0 + cell_h - 18), lbl, fill=(200,220,255), font=font)

    canvas.save(path, quality=95)
    return path

cs = OUT / 'contact_sheet_v4.jpg'
make_sheet(results, str(cs))
print(f"Contact sheet: {cs}")

# Copy to Windows
win = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2020_plate_replace_v4')
if win.exists():
    shutil.rmtree(str(win))
win.mkdir(parents=True)
shutil.copy2(str(cs), str(win / cs.name))
for r in results:
    shutil.copy2(r['new_path'], str(win / Path(r['new_path']).name))
print(f"Windows QA: {win}")

json.dump([{
    'orig': r['orig_path'], 'new': r['new_path'],
    'orig_text': r['orig_text'], 'new_text': r['new_text'],
    'angle': r['angle'],
} for r in results], open(OUT / 'meta.json', 'w'), ensure_ascii=False, indent=2)
print("\nDONE")
