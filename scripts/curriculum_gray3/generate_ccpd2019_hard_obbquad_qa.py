#!/usr/bin/env python3
"""QA batch v2: pre-filter hard-category images by angle, run OBB, sort by occ, pick lowest."""
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

# ── Consts ───────────────────────────────────────────────────────
QA_N = 60
PRE_FILTER = 400  # pre-filter top-N by angle, then run OBB on these
OUT_ROOT = ROOT / 'datasets/ccpd2019_hard_obbquad_qa'
QA_DIR = OUT_ROOT / 'qa'
for d in [QA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALL_PROVINCES = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                 '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                 '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']

CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0,0],[CANVAS_W-1,0],[CANVAS_W-1,CANVAS_H-1],[0,CANVAS_H-1]])

# ── Init generation pipeline ─────────────────────────────────────
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

def board_94x24(img_bgr, quad, preproc='gray3'):
    prepared, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img_bgr, quad, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode=preproc, channel_order='bgr', quad_pad_ratio=0.0)
    return prepared, occ

# ── STEP 1: Collect + rank by angle ──────────────────────────────
print("STEP 1: Collecting hard-category images, scoring by angle...")
HARD_DIRS = ['ccpd_tilt', 'ccpd_fn', 'ccpd_db', 'ccpd_challenge', 'ccpd_blur']
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

candidates.sort(key=lambda x: -x['angle'])
top = candidates[:PRE_FILTER]
print(f"  Collected: {len(candidates)}, pre-filtered top {len(top)} by angle")
print(f"  Angle range: {top[0]['angle']:.1f} – {top[-1]['angle']:.1f}")

# ── STEP 2: Run OBB detection → compute OBB occ ──────────────────
print(f"STEP 2: Running OBB detection on {len(top)}...")
import ultralytics
model = ultralytics.YOLO(str(ROOT / 'datasets/downloaded_green_clone_success/best.pt'))

scored = []
for i, s in enumerate(top):
    img = cv2.imread(s['path'])
    if img is None:
        continue
    out = model(img, verbose=False)[0]
    obb = out.obb
    if obb is not None and len(obb) > 0:
        best = int(obb.conf.argmax())
        q = obb.xyxyxyxy[best].cpu().numpy().astype(np.float32)
        q = order_quad_points(q)
        _, occ = board_94x24(img, q, 'gray3')
        scored.append({**s, 'obb_quad': q, 'obb_conf': float(obb.conf[best]), 'occ': float(occ)})
    if (i+1) % 100 == 0:
        print(f"  [{i+1}/{len(top)}]")

# Filter to target occ range and sample 60
OCC_MIN, OCC_MAX = 0.50, 0.85
in_range = [s for s in scored if OCC_MIN <= s['occ'] <= OCC_MAX]
random.shuffle(in_range)
selected = in_range[:QA_N]
if len(selected) < QA_N:
    print(f"  WARNING: only {len(in_range)} in occ range [{OCC_MIN},{OCC_MAX}], got {len(selected)}")
print(f"  OBB-detected: {len(scored)}, in occ range [{OCC_MIN},{OCC_MAX}]: {len(in_range)}, selected: {len(selected)}")
print(f"  Occ range (selected): {selected[0]['occ']:.3f} – {selected[-1]['occ']:.3f}")

# ── STEP 3: Generate replacements ────────────────────────────────
print(f"STEP 3: Generating {len(selected)} replacements...")

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
    obb_quad = s['obb_quad']

    # Expand GT quad slightly to fully cover original plate area
    exp_quad = expand_quad(gt_quad, pad_ratio=0.08)

    # Compositing: expanded GT quad
    M_fwd = cv2.getPerspectiveTransform(exp_quad, SRC_RECT)
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

    # 94x24 gray3: OBB quad
    orig_94, orig_occ = board_94x24(img, obb_quad, 'gray3')
    new_94, new_occ = board_94x24(replaced, obb_quad, 'gray3')

    fname = f"qa_{i:03d}_occ{new_occ:.2f}_{text}.jpg"
    out_path = QA_DIR / fname
    cv2.imwrite(str(out_path), replaced, [cv2.IMWRITE_JPEG_QUALITY, 95])

    items.append({
        'path': str(out_path), 'text': text, 'category': s['category'],
        'obb_conf': s['obb_conf'], 'occ': float(new_occ), 'angle': s['angle'],
        'orig_img': img, 'replaced_img': replaced,
        'orig_94': orig_94, 'new_94': new_94,
    })

print(f"  Generated: {len(items)}")

# ── STEP 4: Occ distribution ─────────────────────────────────────
print(f"\nOcc distribution:")
bins = [0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    cnt = sum(1 for it in items if lo <= it['occ'] < hi)
    bar = '█' * cnt
    print(f"  {lo:.1f}-{hi:.1f}: {cnt:2d} {bar}")

# ── STEP 5: QA contact sheet ────────────────────────────────────
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

draw.text((10, 8), "CCPD2019 Hard → Lowest OBB-Occ — QA Preview", fill=(220,220,220), font=tf)
headers = ['Original Photo', 'Orig 94x24 gray3', 'Replaced Photo', 'New 94x24 gray3']
for ci, h in enumerate(headers):
    draw.text((15 + ci * cell_w, 50), h, fill=(180,200,255), font=font)

for ri, it in enumerate(items):
    y0 = ri * cell_h + 80
    orig_pil = Image.fromarray(cv2.cvtColor(it['orig_img'], cv2.COLOR_BGR2RGB))
    orig_pil.thumbnail((cell_w - 20, cell_h - 25))
    canvas.paste(orig_pil, (15, y0 + 5))

    o94 = Image.fromarray(cv2.cvtColor(it['orig_94'], cv2.COLOR_BGR2RGB))
    o94 = o94.resize((cell_w - 20, cell_h - 25), Image.NEAREST)
    canvas.paste(o94, (15 + cell_w, y0 + 5))

    rep_pil = Image.open(it['path']).convert('RGB')
    rep_pil.thumbnail((cell_w - 20, cell_h - 25))
    canvas.paste(rep_pil, (15 + 2 * cell_w, y0 + 5))

    n94 = Image.fromarray(cv2.cvtColor(it['new_94'], cv2.COLOR_BGR2RGB))
    n94 = n94.resize((cell_w - 20, cell_h - 25), Image.NEAREST)
    canvas.paste(n94, (15 + 3 * cell_w, y0 + 5))

    lbl = f"occ={it['occ']:.2f} angle={it['angle']:.0f} {it['text']}"
    draw.text((15, y0 + cell_h - 16), lbl, fill=(200,220,255), font=font)

cs_path = QA_DIR / 'qa_contact_sheet.jpg'
canvas.save(str(cs_path), quality=95)

# ── Copy to Windows ──────────────────────────────────────────────
win_dir = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2019_hard_obbquad_qa')
win_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(str(cs_path), str(win_dir / cs_path.name))
for it in items[:20]:
    shutil.copy2(it['path'], win_dir / Path(it['path']).name)
print(f"  Windows QA: {win_dir}")
print(f"\nDone.")
