#!/usr/bin/env python3
"""
CCPD2019 hard categories + GT quad + Gaussian noise → green extreme training data QA.
No OBB detection needed. Noise simulates board-level quad imprecision.
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
sys.path.insert(0, str(ROOT / 'src' / 'utils'))

from load_data import parse_ccpd_quad_from_name, order_quad_points, expand_quad, \
    prepare_board_ocr_input_from_quad_bgr888
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

QA_N = 60
NOISE_SIGMA = 4.0
OCC_MIN, OCC_MAX = 0.50, 0.85

OUT_ROOT = ROOT / 'datasets/ccpd2019_green_noise_qa'
QA_DIR = OUT_ROOT / 'qa'
for d in [QA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALL_PROVINCES = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                 '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                 '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']

CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0,0],[CANVAS_W-1,0],[CANVAS_W-1,CANVAS_H-1],[0,CANVAS_H-1]])

chars_gen = CharsImageGenerator('small_new_energy')
tg = LicensePlateImageGenerator('small_new_energy')
tmpl = tg.generate_template_image(chars_gen.plate_width, chars_gen.plate_height)
aug = ImageAugmentation('small_new_energy', tmpl)
aug.env_data_paths = [str(ROOT / p) for p in aug.env_data_paths]
_smu = str(ROOT / 'images' / 'smu.jpg')
if os.path.exists(_smu):
    aug.smu = cv2.imread(_smu)

def build_plate(text):
    ci = chars_gen.generate_images([text])[0]
    ai = aug.augment(ci, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(ai, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

def match_brightness(new_bgr, orig_patch_bgr):
    new_lab = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = orig_lab.shape[:2]
    cy, cx = h//2, w//2
    roi_h, roi_w = int(h*0.8), int(w*0.8)
    y1, x1 = cy-roi_h//2, cx-roi_w//2
    L_new = new_lab[y1:y1+roi_h, x1:x1+roi_w, 0]
    L_orig = orig_lab[y1:y1+roi_h, x1:x1+roi_w, 0]
    m_n, s_n = L_new.mean(), L_new.std() + 1e-6
    m_o, s_o = L_orig.mean(), L_orig.std() + 1e-6
    ratio = max(0.5, min(2.0, s_o / s_n))
    new_lab[:,:,0] = np.clip((new_lab[:,:,0] - m_n) * ratio + m_o, 0, 255)
    return cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def angle_score(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i-1)%4]; b = p[i]; c = p[(i+1)%4]
        v1, v2 = a - b, c - b
        d = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angles.append(abs(math.degrees(math.acos(max(-1,min(1,d)))) - 90))
    return float(np.mean(angles))

def add_noise_to_quad(quad, sigma):
    noise = np.random.randn(4, 2).astype(np.float32) * sigma
    return (quad.copy().reshape(4, 2) + noise).reshape(-1, 2)

def board_94x24(img_bgr, quad, preproc='gray3'):
    prepared, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img_bgr, quad, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode=preproc, channel_order='bgr', quad_pad_ratio=0.0)
    return prepared, occ

# ═══════════════════════════════════════════════════════════════════
#  STEP 1: Load CCPD2019 hard categories
# ═══════════════════════════════════════════════════════════════════
print("STEP 1: Loading CCPD2019 hard-category images...")
HARD_DIRS = ['ccpd_tilt', 'ccpd_db', 'ccpd_fn', 'ccpd_challenge']
candidates = []
for dname in HARD_DIRS:
    d = ROOT / 'datasets/CCPD2019' / dname
    for fn in os.listdir(d):
        if not fn.endswith('.jpg'):
            continue
        p = str(d / fn)
        q = parse_ccpd_quad_from_name(p)
        if q is None:
            continue
        a = angle_score(q)
        candidates.append({'path': p, 'gt_quad': q, 'angle': a, 'category': dname})
print(f"  Total: {len(candidates)}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 2: Score + noise → occ for top-angle subset
# ═══════════════════════════════════════════════════════════════════
print(f"\nSTEP 2: Adding noise σ={NOISE_SIGMA}px, measuring occ...")
candidates.sort(key=lambda x: -x['angle'])
top = candidates[:400]

scored = []
for s in top:
    img = cv2.imread(s['path'])
    if img is None:
        continue
    gt = order_quad_points(s['gt_quad'])
    noisy = add_noise_to_quad(gt, NOISE_SIGMA)
    _, occ = board_94x24(img, noisy, 'gray3')
    scored.append({**s, 'noisy_quad': noisy, 'occ': float(occ)})

in_range = [s for s in scored if OCC_MIN <= s['occ'] <= OCC_MAX]
print(f"  Scored: {len(scored)}, in occ target [{OCC_MIN},{OCC_MAX}]: {len(in_range)}")
print(f"  Occ range all: {min(s['occ'] for s in scored):.3f} – {max(s['occ'] for s in scored):.3f}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 3: Generate QA replacements
# ═══════════════════════════════════════════════════════════════════
print(f"\nSTEP 3: Generating {QA_N} replacements...")

random.shuffle(in_range)
selected = in_range[:QA_N]

LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')
used_texts = set()

def make_text(province):
    while True:
        t = province + random.choice(LETTERS) + random.choice(['D','F']) \
            + random.choice(ALNUM) + ''.join(random.choice(DIGITS) for _ in range(4))
        if t not in used_texts:
            used_texts.add(t)
            return t

items = []
for i, s in enumerate(selected):
    prov = random.choice(ALL_PROVINCES)
    text = make_text(prov)
    img = cv2.imread(s['path'])
    h, w = img.shape[:2]
    gt_quad = order_quad_points(s['gt_quad'])
    noisy_quad = s['noisy_quad']

    # Compositing: expanded GT quad for clean mask coverage
    exp_quad = expand_quad(gt_quad, pad_ratio=0.08)
    # Brightness matching: use GT quad (not expanded) to avoid diluting with background
    M_fwd = cv2.getPerspectiveTransform(gt_quad, SRC_RECT)
    plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    raw = build_plate(text)
    matched = match_brightness(raw, plate_orig)
    M_inv = cv2.getPerspectiveTransform(SRC_RECT, exp_quad)
    warped = cv2.warpPerspective(matched, M_inv, (w, h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [exp_quad.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0
    result = img.astype(np.float32)
    for c in range(3):
        result[:,:,c] = warped[:,:,c] * mask + result[:,:,c] * (1.0 - mask)
    replaced = np.clip(result, 0, 255).astype(np.uint8)

    # 94x24 gray3: GT quad (ideal reference)
    gt_94, _ = board_94x24(replaced, order_quad_points(s['gt_quad']), 'gray3')
    # 94x24 gray3: noisy quad (simulates board)
    new_94, new_occ = board_94x24(replaced, noisy_quad, 'gray3')

    fname = f"qa_{i:03d}_occ{new_occ:.2f}_{text}.jpg"
    out_path = QA_DIR / fname
    cv2.imwrite(str(out_path), replaced, [cv2.IMWRITE_JPEG_QUALITY, 95])

    items.append({
        'path': str(out_path), 'text': text, 'occ': float(new_occ), 'angle': s['angle'],
        'category': s['category'],
        'orig_img': img, 'replaced_img': replaced,
        'gt_94': gt_94, 'new_94': new_94,
    })

print(f"  Generated: {len(items)}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 4: Occ distribution + category breakdown
# ═══════════════════════════════════════════════════════════════════
print(f"\nOcc distribution:")
bins = [0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    cnt = sum(1 for it in items if lo <= it['occ'] < hi)
    bar = '█' * cnt
    print(f"  {lo:.2f}-{hi:.2f}: {cnt:2d} {bar}")

cat_cnt = Counter(it['category'] for it in items)
print(f"\nCategory breakdown:")
for cat, cnt in sorted(cat_cnt.items()):
    print(f"  {cat}: {cnt}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 5: QA contact sheet
# ═══════════════════════════════════════════════════════════════════
print("STEP 5: Building QA contact sheet...")

cols, cell_w, cell_h = 4, 280, 140
W = cols * cell_w + 40
H = len(items) * cell_h + 80
canvas = Image.new('RGB', (W, H), (25, 25, 25))
draw = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 10)
except: font = ImageFont.load_default()
try: tf = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 16)
except: tf = ImageFont.load_default()

draw.text((10, 8), f"CCPD2019 Hard + GT quad + σ={NOISE_SIGMA}px noise — Green Extreme QA", fill=(220,220,220), font=tf)
headers = ['Original Photo', 'GT Quad 94x24 gray3', 'Replaced Photo', 'Noisy Quad 94x24 gray3']
for ci, h in enumerate(headers):
    draw.text((15 + ci * cell_w, 50), h, fill=(180,200,255), font=font)

for ri, it in enumerate(items):
    y0 = ri * cell_h + 80
    orig_pil = Image.fromarray(cv2.cvtColor(it['orig_img'], cv2.COLOR_BGR2RGB))
    orig_pil.thumbnail((cell_w - 20, cell_h - 25))
    canvas.paste(orig_pil, (15, y0 + 5))

    # GT quad 94x24 gray3 (ideal, no noise)
    gt94 = Image.fromarray(cv2.cvtColor(it['gt_94'], cv2.COLOR_BGR2RGB))
    gt94 = gt94.resize((cell_w - 20, cell_h - 25), Image.NEAREST)
    canvas.paste(gt94, (15 + cell_w, y0 + 5))

    # Replaced photo
    rep_pil = Image.open(it['path']).convert('RGB')
    rep_pil.thumbnail((cell_w - 20, cell_h - 25))
    canvas.paste(rep_pil, (15 + 2 * cell_w, y0 + 5))

    # Noisy quad 94x24 gray3 (simulated board)
    n94 = Image.fromarray(cv2.cvtColor(it['new_94'], cv2.COLOR_BGR2RGB))
    n94 = n94.resize((cell_w - 20, cell_h - 25), Image.NEAREST)
    canvas.paste(n94, (15 + 3 * cell_w, y0 + 5))

    lbl = f"occ={it['occ']:.2f} angle={it['angle']:.0f} {it['category']}"
    draw.text((15, y0 + cell_h - 16), lbl, fill=(200,220,255), font=font)

cs_path = QA_DIR / 'qa_contact_sheet.jpg'
canvas.save(str(cs_path), quality=95)

# ── Copy to Windows ──────────────────────────────────────────────
win_dir = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2019_green_noise_qa')
win_dir.mkdir(parents=True, exist_ok=True)
# Clean any old content
for f in win_dir.iterdir():
    if f.is_file():
        f.unlink()
shutil.copy2(str(cs_path), str(win_dir / cs_path.name))
for it in items[:20]:
    shutil.copy2(it['path'], win_dir / Path(it['path']).name)
print(f"  Windows QA: {win_dir}")
print(f"\nDone.")
