#!/usr/bin/env python3
"""Step2: Province-region degradation training data.
~9000-10000 images: 31 provinces × 200 base + 7 confusion provinces × 400 extra + 1000 anchor style.
Degradation ONLY on left province region. Suffix random."""
import csv, json, os, sys, math, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'src' / 'utils'))
from load_data import order_quad_points
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

random.seed(20260504)
ALL_PROVS = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
             '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
             '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
CONFUSE_PROVS = ['京','苏','皖','粤','青','藏','浙']  # extra weight

# Config
BASE_PER_PROV = 200
EXTRA_PER_CONFUSE = 400  # additional for confusion provinces
ANCHOR_COUNT = 1000
TOTAL = len(ALL_PROVS) * BASE_PER_PROV + len(CONFUSE_PROVS) * EXTRA_PER_CONFUSE + ANCHOR_COUNT
print(f"Target: {len(ALL_PROVS)}×{BASE_PER_PROV} + {len(CONFUSE_PROVS)}×{EXTRA_PER_CONFUSE} + {ANCHOR_COUNT} anchor = {TOTAL}")

POSE_QUADS = ROOT / 'datasets/ccpd2020_pose_quads' / 'pose_quads.jsonl'
OUT_DIR = ROOT / 'datasets' / 'province_degrade_train_v1'
IMG_DIR = OUT_DIR / 'images' / 'train'
MANIFEST_DIR = ROOT / 'manifests' / 'province_degrade_train_v1'
IMG_DIR.mkdir(parents=True, exist_ok=True); MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Init pipeline
_chars_gen = CharsImageGenerator('small_new_energy')
_tg = LicensePlateImageGenerator('small_new_energy')
_tmpl = _tg.generate_template_image(_chars_gen.plate_width, _chars_gen.plate_height)
_aug = ImageAugmentation('small_new_energy', _tmpl)
_aug.env_data_paths = [str(ROOT / p) for p in _aug.env_data_paths]
_smu_path = str(ROOT / 'images' / 'smu.jpg')
if os.path.exists(_smu_path): _aug.smu = cv2.imread(_smu_path)

CANVAS_W, CANVAS_H = 246, 72
SRC_RECT = np.float32([[0,0],[CANVAS_W-1,0],[CANVAS_W-1,CANVAS_H-1],[0,CANVAS_H-1]])
LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
DIGITS = '0123456789'

def make_random_suffix(used):
    # Green plate format: 省(1) + 字母(1) + 字母(1) + 5数字(5) = 8 chars
    while True:
        s = random.choice(LETTERS) + random.choice(LETTERS) + ''.join(random.choices(DIGITS, k=5))
        if s not in used:
            used.add(s); return s

def build_plate(text):
    ci = _chars_gen.generate_images([text])[0]
    ai = _aug.augment(ci, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(ai, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

def match_brightness(new_bgr, orig_patch_bgr):
    new_lab = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = orig_lab.shape[:2]
    cy, cx = h//2, w//2
    roi = orig_lab[cy-h//4:cy+h//4, cx-w//4:cx+w//4, 0]
    L_new = new_lab[cy-h//4:cy+h//4, cx-w//4:cx+w//4, 0]
    ratio = max(0.5, min(2.0, roi.std()/(L_new.std()+1e-6)))
    new_lab[:,:,0] = np.clip((new_lab[:,:,0] - L_new.mean())*ratio + roi.mean(), 0, 255)
    return cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

PROV_RATIO = 0.35

def degrade_province_region(plate, is_anchor=False):
    out = plate.copy()
    h, w = out.shape[:2]
    p_w = int(w * PROV_RATIO)
    
    mask = np.zeros((h, w), dtype=np.float32)
    feather = max(1, int(w * 0.08))
    for x in range(p_w + feather):
        mask[:, x] = max(0.0, 1.0 - x / (p_w + feather)) if x >= p_w else 1.0
    
    base = out.copy()
    degraded = base[:, :p_w + feather, :].copy()
    d_w = degraded.shape[1]
    
    if is_anchor:
        if random.random() < 0.60:
            degraded = cv2.convertScaleAbs(degraded, alpha=random.uniform(0.2, 0.5), beta=0)
        if random.random() < 0.30:
            degraded = np.clip(degraded.astype(np.float32) * (np.random.rand(h,d_w,1)*0.6+0.4) + 40, 0, 255).astype(np.uint8)
        if random.random() < 0.40:
            s = random.randint(3, 7)
            if s < d_w: degraded[:, s:,:] = degraded[:, :-s,:]; degraded[:, :s,:] = 0
    else:
        if random.random() < 0.40:
            degraded = cv2.convertScaleAbs(degraded, alpha=random.uniform(0.3, 0.6), beta=0)
        if random.random() < 0.20:
            degraded = np.clip(degraded.astype(np.float32)*(np.random.rand(h,d_w,1)*0.5+0.5)+np.random.randint(0,60),0,255).astype(np.uint8)
        if random.random() < 0.25:
            g = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY); g = cv2.convertScaleAbs(g, alpha=random.uniform(0.4,0.8), beta=random.uniform(20,60)); degraded = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if random.random() < 0.20:
            s=random.randint(2,5)
            if s<d_w: degraded[:,s:,:]=degraded[:,:-s,:]; degraded[:,:s,:]=random.randint(0,30)
        if random.random() < 0.20:
            s=random.randint(-3,3)
            if s>0 and s<d_w: degraded[:,s:,:]=degraded[:,:-s,:]; degraded[:,:s,:]=np.random.randint(0,40,(h,s,3),dtype=np.uint8)
            elif s<0 and -s<d_w: degraded[:,:s,:]=degraded[:,-s:,:]; degraded[:,s:,:]=np.random.randint(0,40,(h,-s,3),dtype=np.uint8)
        if random.random() < 0.15:
            prov_only = degraded[:,:p_w,:]; k=random.choice([3,5]); prov_only=cv2.GaussianBlur(prov_only,(k,k),0); degraded[:,:p_w,:]=prov_only
        if random.random() < 0.15:
            noise = np.random.randint(-25,25,degraded.shape,dtype=np.int16); degraded=np.clip(degraded.astype(np.int16)+noise,0,255).astype(np.uint8)
    
    mask_3ch = np.stack([mask]*3, axis=2)
    out = (base.astype(np.float32) * (1 - mask_3ch) +
           cv2.resize(np.pad(degraded,((0,0),(0,out.shape[1]-degraded.shape[1]),(0,0)),mode='edge'),(out.shape[1],out.shape[0])).astype(np.float32) * mask_3ch).clip(0,255).astype(np.uint8)
    if random.random() < 0.10 and not is_anchor:
        out = plate.copy()
    return out

# ── Load sources ────────────────────────────────────────────────────
print("Loading pose quads...")
pose_data = [json.loads(l) for l in open(POSE_QUADS)]
test_data = [r for r in pose_data if r['split'] == 'test']
by_prov = {}
for r in test_data:
    by_prov.setdefault(r['text'][0], []).append(r)

# Build source assignment
source_map = []  # (source_record, province)
used_texts = set()

for prov in ALL_PROVS:
    avail = by_prov.get(prov, [])
    fallback = by_prov.get('皖', [])
    n = BASE_PER_PROV + (EXTRA_PER_CONFUSE if prov in CONFUSE_PROVS else 0)
    for i in range(n):
        src = avail[i % len(avail)] if avail else fallback[i % len(fallback)]
        source_map.append((src, prov, 'standard'))

# Anchor samples: use 皖 sources with stronger degradation
for i in range(ANCHOR_COUNT):
    fallback = by_prov.get('皖', test_data)
    src = fallback[i % len(fallback)]
    prov = random.choice(['京', '苏', '粤', '浙'])
    source_map.append((src, prov, 'anchor'))

random.shuffle(source_map)
print(f"Total sources: {len(source_map)}")

# ── Generation ──────────────────────────────────────────────────────
MANIFEST_FIELDS = ['img_path','text','family','source','split','has_quad',
    'can_parse_ccpd_geom','can_perspective','preprocess_group','ocr_crop_mode',
    'ocr_resize_mode','ocr_resize_kernel','ocr_preproc','ocr_channel_order','ocr_quad_pad_ratio',
    'quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']

rows = []
prov_count = Counter()
t0 = time.time()

for idx, (src_rec, prov, mode) in enumerate(source_map):
    img_path = src_rec['img_path']
    gt_quad_raw = np.array(src_rec['gt_quad'])
    gt_quad = np.array([gt_quad_raw[2], gt_quad_raw[3], gt_quad_raw[0], gt_quad_raw[1]], dtype=np.float32)
    pose_quad = np.array(src_rec['pose_quad'])
    
    new_suffix = make_random_suffix(used_texts)
    new_text = prov + new_suffix
    
    try:
        img = cv2.imread(img_path)
        if img is None: raise ValueError("Cannot read")
        h, w = img.shape[:2]
        
        M_fwd = cv2.getPerspectiveTransform(gt_quad.astype(np.float32), SRC_RECT)
        plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR)
        
        raw = build_plate(new_text)
        matched = match_brightness(raw, plate_orig)
        degraded = degrade_province_region(matched, is_anchor=(mode=='anchor'))
        
        M_inv = cv2.getPerspectiveTransform(SRC_RECT, gt_quad.astype(np.float32))
        warped = cv2.warpPerspective(degraded, M_inv, (w, h), flags=cv2.INTER_LINEAR)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [gt_quad.round().astype(np.int32)], 255)
        mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
        result = (img.astype(np.float32) * (1 - mask[:,:,None]) +
                  warped.astype(np.float32) * mask[:,:,None]).clip(0, 255).astype(np.uint8)
        
        fname = f"{Path(img_path).stem}_pdeg_{new_text}.jpg"
        out_path = IMG_DIR / fname
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        rows.append({
            'img_path': str(out_path), 'text': new_text, 'family': 'green8',
            'source': 'province_degrade_train_v1', 'split': 'train',
            'has_quad': '1', 'can_parse_ccpd_geom': '0', 'can_perspective': '1',
            'preprocess_group': 'ccpd_board', 'ocr_crop_mode': 'obb_warp',
            'ocr_resize_mode': 'letterbox', 'ocr_resize_kernel': 'nn',
            'ocr_preproc': 'none', 'ocr_channel_order': 'bgr', 'ocr_quad_pad_ratio': '0.0',
            'quad_1x': f'{pose_quad[0][0]:.2f}', 'quad_1y': f'{pose_quad[0][1]:.2f}',
            'quad_2x': f'{pose_quad[1][0]:.2f}', 'quad_2y': f'{pose_quad[1][1]:.2f}',
            'quad_3x': f'{pose_quad[2][0]:.2f}', 'quad_3y': f'{pose_quad[2][1]:.2f}',
            'quad_4x': f'{pose_quad[3][0]:.2f}', 'quad_4y': f'{pose_quad[3][1]:.2f}',
        })
        prov_count[prov] += 1
        
        if (idx+1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(source_map)}] {elapsed/(idx+1):.1f}s/img, {len(rows)} ok")
    except Exception as e:
        print(f"  Error {img_path}: {e}")
        continue

csv_path = MANIFEST_DIR / 'train_province_degrade_v1.csv'
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
    w.writeheader(); w.writerows(rows)

print(f"\nDone: {len(rows)} images")
print(f"Province distribution (top 15): {dict(sorted(prov_count.items(), key=lambda x:-x[1])[:15])}")
print(f"Manifest: {csv_path}")
PYEOF
