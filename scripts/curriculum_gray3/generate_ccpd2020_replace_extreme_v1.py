#!/usr/bin/env python3
"""
B2-C 数据生成：CCPD2020 替换 extreme 3000 张
- 按 angle 从高到低取 3000 张
- 省分布：皖 30, 其他 30 省每省 99
- 使用已验证的 v4 pipeline
"""

import csv, json, os, sys, math, random, shutil, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from load_data import parse_ccpd_quad_from_name, order_quad_points, \
    prepare_board_ocr_input_from_quad_bgr888

# ── Consts ───────────────────────────────────────────────────────

OUT_ROOT = ROOT / 'datasets/ccpd2020_replace_extreme_v1'
IMG_DIR = OUT_ROOT / 'images'
QA_DIR = OUT_ROOT / 'qa'
MANIFEST_DIR = ROOT / 'manifests/ccpd2020_replace_extreme_v1'
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

ALL_PROVINCES = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                 '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                 '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
NON_AH = sorted(p for p in ALL_PROVINCES if p != '皖')

LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')

CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0,0],[CANVAS_W-1,0],[CANVAS_W-1,CANVAS_H-1],[0,CANVAS_H-1]])

# ── Metrics ──────────────────────────────────────────────────────

def angle_score(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i-1)%4]; b = p[i]; c = p[(i+1)%4]
        v1, v2 = a - b, c - b
        d = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angles.append(abs(math.degrees(math.acos(max(-1,min(1,d)))) - 90))
    return float(np.mean(angles))

# ── Init generation pipeline ─────────────────────────────────────

print("Init generation pipeline...")
sys.path.insert(0, str(ROOT / 'src' / 'utils'))
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

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
    return cv2.resize(ai, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

# ── Brightness matching ──────────────────────────────────────────

def match_brightness(new_bgr, orig_patch_bgr):
    # LAB: only match L channel, preserve AB (color)
    new_lab = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = orig_lab.shape[:2]
    cy, cx = h//2, w//2
    roi_h, roi_w = int(h*0.8), int(w*0.8)
    y1, x1 = cy-roi_h//2, cx-roi_w//2
    L_new = new_lab[y1:y1+roi_h, x1:x1+roi_w, 0]
    L_orig = orig_lab[y1:y1+roi_h, x1:x1+roi_w, 0]
    m_n = L_new.mean()
    s_n = L_new.std() + 1e-6
    m_o = L_orig.mean()
    s_o = L_orig.std() + 1e-6
    ratio = max(0.5, min(2.0, s_o / s_n))
    new_lab[:,:,0] = np.clip((new_lab[:,:,0] - m_n) * ratio + m_o, 0, 255)
    return cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

# ── Text generation ──────────────────────────────────────────────

def make_text(province, used):
    while True:
        t = province + random.choice(LETTERS) + random.choice(['D','F']) \
            + random.choice(ALNUM) + ''.join(random.choice(DIGITS) for _ in range(4))
        if t not in used:
            used.add(t)
            return t

# ═══════════════════════════════════════════════════════════════════
#  STEP 1: Load + sort CCPD2020 test by angle
# ═══════════════════════════════════════════════════════════════════

print("\n─── STEP 1: Load and score CCPD2020 test ───")
label_path = ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv'
candidates = []
with open(label_path, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        p = row['img_path']
        if not os.path.exists(p):
            continue
        q = parse_ccpd_quad_from_name(p)
        if q is None:
            continue
        a = angle_score(q)
        candidates.append({'path': p, 'text': row['text'], 'angle': a, 'quad': q})

# Sort by angle descending
candidates.sort(key=lambda x: -x['angle'])
print(f"Total CCPD2020 test with valid quad: {len(candidates)}")
print(f"Angle range: {candidates[0]['angle']:.1f} ~ {candidates[-1]['angle']:.1f}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 2: Assign province distribution
# ═══════════════════════════════════════════════════════════════════

print("\n─── STEP 2: Province assignment ───")

TOTAL = 3000
TRAIN_N = 2700
VAL_N = 300
AH_N = 30  # 皖
OTHER_N = (TOTAL - AH_N) // 30  # 99 per non-皖 province

# Build province quotas
prov_quota = {'皖': AH_N}
for p in NON_AH:
    prov_quota[p] = OTHER_N

print(f"Province distribution ({TOTAL} total):")
for p in ALL_PROVINCES:
    print(f"  {p}: {prov_quota[p]:>4d}")

# Assign provinces to the top 3000 candidates (highest angle first)
selected = candidates[:TOTAL]
print(f"\nSelected top {len(selected)} (angle ≥ {selected[-1]['angle']:.1f})")

# Shuffle selected to randomize which photo gets which province
random.shuffle(selected)

# Assign province per sample
prov_count = Counter()
used_texts = set()
assignments = []
for s in selected:
    # Pick province with remaining quota
    if prov_count.get('皖', 0) < AH_N:
        prov = '皖'
    else:
        avail = [p for p in NON_AH if prov_count.get(p, 0) < OTHER_N]
        if not avail:
            avail = NON_AH  # fallback
        prov = random.choice(avail)
    
    prov_count[prov] += 1
    text = make_text(prov, used_texts)
    assignments.append({**s, 'new_text': text, 'province': prov})

print(f"\nActual province distribution:")
for p in ALL_PROVINCES:
    c = prov_count.get(p, 0)
    bar = '█' * (c // 5)
    print(f"  {p}: {c:>4d} {bar}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 3: Generate replacements
# ═══════════════════════════════════════════════════════════════════

print(f"\n─── STEP 3: Generate {TOTAL} replacements ───")

# Split train/val
random.shuffle(assignments)
train_set = assignments[:TRAIN_N]
val_set = assignments[TRAIN_N:TRAIN_N+VAL_N]

train_dir = IMG_DIR / 'train'
val_dir = IMG_DIR / 'val'
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'preprocess_group', 'ocr_crop_mode', 'ocr_resize_mode',
    'ocr_resize_kernel', 'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio',
]

def process_one(item, out_dir, split_name, idx, total):
    """Process a single replacement. Returns manifest row or None on failure."""
    try:
        img = cv2.imread(item['path'])
        if img is None:
            return None
        h, w = img.shape[:2]
        
        quad = order_quad_points(item['quad'])
        
        # Forward warp for brightness reference
        M_fwd = cv2.getPerspectiveTransform(quad, SRC_RECT)
        plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)
        
        # Generate + match
        raw = build_plate(item['new_text'])
        matched = match_brightness(raw, plate_orig)
        
        # Inverse warp
        M_inv = cv2.getPerspectiveTransform(SRC_RECT, quad)
        warped = cv2.warpPerspective(matched, M_inv, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        
        # Tight mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
        
        result = img.astype(np.float32)
        for c in range(3):
            result[:,:,c] = warped[:,:,c] * mask + result[:,:,c] * (1.0 - mask)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Save
        stem = Path(item['path']).stem
        fname = f"{stem}_replaced_{item['new_text']}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Manifest row
        row = {
            'img_path': str(out_path),
            'text': item['new_text'],
            'family': 'green8',
            'source': 'green_ccpd2020_replace_extreme',
            'split': split_name,
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
        return row
    except Exception as e:
        print(f"  [{idx}/{total}] ERROR {item['path']}: {e}")
        return None

# Process train
train_rows = []
t0 = time.time()
for idx, item in enumerate(train_set):
    r = process_one(item, train_dir, 'train', idx+1, TRAIN_N)
    if r:
        train_rows.append(r)
    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (TRAIN_N - idx - 1) / rate if rate > 0 else 0
        print(f"  [{idx+1}/{TRAIN_N}] {rate:.1f} img/s, ETA {eta/60:.0f}min")

# Process val
val_rows = []
for idx, item in enumerate(val_set):
    r = process_one(item, val_dir, 'val', idx+1, VAL_N)
    if r:
        val_rows.append(r)

print(f"\nGenerated: {len(train_rows)} train + {len(val_rows)} val")

# ═══════════════════════════════════════════════════════════════════
#  STEP 4: Write manifests
# ═══════════════════════════════════════════════════════════════════

print("\n─── STEP 4: Write manifests ───")

for split_name, rows, out_name in [
    ('train', train_rows, 'train_B2C_ccpd2020_replace_extreme.csv'),
    ('val', val_rows, 'val_B2C_ccpd2020_replace_extreme.csv'),
]:
    out_path = MANIFEST_DIR / out_name
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  {out_path}: {len(rows)} rows")

# Summary
summary = {
    'n_total': len(train_rows) + len(val_rows),
    'n_train': len(train_rows),
    'n_val': len(val_rows),
    'province_distribution': dict(sorted(Counter(r['text'][0] for r in train_rows + val_rows).items())),
    'angle_range': {'min': selected[-1]['angle'], 'max': selected[0]['angle']},
    'generation_time_s': time.time() - t0,
    'source': 'CCPD2020_test',
    'pipeline': 'v4_ccpd2020_replace',
}
with open(OUT_ROOT / 'generation_meta.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n  Summary: {OUT_ROOT / 'generation_meta.json'}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 5: QA contact sheet
# ═══════════════════════════════════════════════════════════════════

print("\n─── STEP 5: QA contact sheet ───")

def board_94x24(img_bgr, quad, preproc='none'):
    prepared, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        img_bgr, quad, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode=preproc, channel_order='bgr',
        quad_pad_ratio=0.0,
    )
    return prepared

# Sample 12 diverse items for QA
qa_items = []
for prov in random.sample(NON_AH, min(12, len(NON_AH))):
    for r in train_rows:
        if r['text'][0] == prov:
            qa_items.append(r)
            break

if len(qa_items) < 12:
    # Fill with random
    qa_items = random.sample(train_rows, min(12, len(train_rows)))

# Build contact sheet
cols, cell_w, cell_h = 6, 280, 130
rows_n = len(qa_items)
W = cols * cell_w + 40
H = rows_n * cell_h + 80
canvas = Image.new('RGB', (W, H), (25, 25, 25))
draw = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 11)
except: font = ImageFont.load_default()
try: tf = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 18)
except: tf = ImageFont.load_default()

draw.text((10, 8), "CCPD2020 Replace Extreme v1 — QA Sample", fill=(220,220,220), font=tf)
headers = ['Photo', 'Orig 94×24', 'Orig gray3', 'Replaced', 'New 94×24', 'New gray3']
for ci, h in enumerate(headers):
    draw.text((15 + ci * cell_w, 48), h, fill=(180,200,255), font=font)

for ri, r in enumerate(qa_items):
    y0 = ri * cell_h + 80
    path = r['img_path']
    
    # Get original photo and quad
    # We need to find the original CCPD2020 path from the filename
    orig_stem = Path(path).stem.replace('_replaced_*', '')
    # Actually, we can reconstruct. The stem pattern is: {orig_stem}_replaced_{new_text}
    # So orig_stem = everything before '_replaced_'
    sp = Path(path).stem
    m = sp.rsplit('_replaced_', 1)
    orig_stem = m[0] if len(m) > 1 else sp
    
    # Find the original candidate
    orig_item = None
    for c in candidates:
        if Path(c['path']).stem == orig_stem:
            orig_item = c
            break
    
    if orig_item is None:
        continue
    
    orig_img = cv2.imread(orig_item['path'])
    if orig_img is None:
        continue
    q = order_quad_points(orig_item['quad'])
    
    orig_94 = board_94x24(orig_img, q, 'none')
    orig_gray = board_94x24(orig_img, q, 'gray3')
    
    new_img = cv2.imread(path)
    if new_img is None:
        continue
    new_94 = board_94x24(new_img, q, 'none')
    new_gray = board_94x24(new_img, q, 'gray3')
    
    items_display = [
        (Image.open(orig_item['path']).convert('RGB'), 'photo'),
        (Image.fromarray(cv2.cvtColor(orig_94, cv2.COLOR_BGR2RGB)), 'board'),
        (Image.fromarray(cv2.cvtColor(orig_gray, cv2.COLOR_BGR2RGB)), 'board'),
        (Image.open(path).convert('RGB'), 'photo'),
        (Image.fromarray(cv2.cvtColor(new_94, cv2.COLOR_BGR2RGB)), 'board'),
        (Image.fromarray(cv2.cvtColor(new_gray, cv2.COLOR_BGR2RGB)), 'board'),
    ]
    
    for ci, (im, typ) in enumerate(items_display):
        if typ == 'photo':
            im.thumbnail((cell_w - 20, cell_h - 25))
        else:
            im = im.resize((cell_w - 20, cell_h - 25), Image.NEAREST)
        canvas.paste(im, (15 + ci * cell_w, y0 + 5))
    
    # Angle info
    a = angle_score(orig_item['quad'])
    lbl = f"angle={a:.1f}  {r['text']}"
    draw.text((15, y0 + cell_h - 18), lbl, fill=(200,220,255), font=font)

cs_path = QA_DIR / 'qa_contact_sheet.jpg'
canvas.save(str(cs_path), quality=95)
print(f"  QA contact sheet: {cs_path}")

# Copy to Windows
win_qa = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2020_replace_extreme_v1_qa')
win_qa.mkdir(parents=True, exist_ok=True)
shutil.copy2(str(cs_path), str(win_qa / cs_path.name))
print(f"  Windows: {win_qa}")

print(f"\n{'='*60}")
print(f"  DONE")
print(f"  Train: {len(train_rows)} / {TRAIN_N}")
print(f"  Val:   {len(val_rows)} / {VAL_N}")
print(f"  Data:  {OUT_ROOT}")
print(f"  Manifest: {MANIFEST_DIR}")
print(f"  QA: {win_qa}")
print(f"{'='*60}")
