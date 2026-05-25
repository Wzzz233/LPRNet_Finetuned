#!/usr/bin/env python3
"""
Generate extreme replacement data with source-level train/val split.
- Top 25% highest-angle CCPD2020 test images
- 31 provinces × 30 train + 10 val = 930 train + 310 val
- Source images are split FIRST (no overlap between train/val)
- Fixed pipeline: order_quad_points on gt_quad
- Records Pose quad for training
"""
import csv, json, os, sys, math, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT = Path('/home/wzzz/LPRNet')
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'utils'))

from load_data import parse_ccpd_quad_from_name, order_quad_points, prepare_board_ocr_input_from_quad_bgr888
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

random.seed(20260503)

# ── Config ──────────────────────────────────────────────────────────
TOP_PCT = 35  # expanded from 25% for E3: more source diversity
TRAIN_PER_PROV = 30
VAL_PER_PROV = 10
ALL_PROVINCES = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                 '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                 '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']

OUT_DIR = ROOT / 'datasets' / 'ccpd2020_replace_extreme_v4'  # E3: expanded source diversity
IMG_TRAIN = OUT_DIR / 'images' / 'train'
IMG_VAL = OUT_DIR / 'images' / 'val'
MANIFEST_DIR = ROOT / 'manifests' / 'ccpd2020_replace_extreme_v4'
QA_DIR = OUT_DIR / 'qa'
for d in [IMG_TRAIN, IMG_VAL, MANIFEST_DIR, QA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Angle score ─────────────────────────────────────────────────────
def angle_score_from_quad(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i-1)%4]; b = p[i]; c = p[(i+1)%4]
        v1, v2 = a - b, c - b
        d = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angles.append(abs(math.degrees(math.acos(max(-1,min(1,d)))) - 90))
    return float(np.mean(angles))

# ── Load pose data ──────────────────────────────────────────────────
POSE_QUADS = ROOT / 'datasets/ccpd2020_pose_quads' / 'pose_quads.jsonl'
pose_data = [json.loads(l) for l in open(POSE_QUADS)]
for r in pose_data:
    r['angle'] = angle_score_from_quad(np.array(r['gt_quad']))

# Only test split
test_data = [r for r in pose_data if r['split'] == 'test']
test_data.sort(key=lambda x: -x['angle'])  # hardest first

# Select top 25%
n_top = max(1, int(len(test_data) * TOP_PCT / 100))
top_sources = test_data[:n_top]
print(f"Top {TOP_PCT}%: {len(top_sources)} samples, angle range {top_sources[-1]['angle']:.2f}° to {top_sources[0]['angle']:.2f}°")

# ── Source-level train/val split ────────────────────────────────────
unique_sources = list({r['img_path']: r for r in top_sources}.values())
random.shuffle(unique_sources)

n_val = max(1, int(len(unique_sources) * VAL_PER_PROV / (TRAIN_PER_PROV + VAL_PER_PROV)))
val_sources_set = set(r['img_path'] for r in unique_sources[:n_val])
train_sources = [r for r in unique_sources if r['img_path'] not in val_sources_set]
val_sources = [r for r in unique_sources if r['img_path'] in val_sources_set]

print(f"Source split: train={len(train_sources)}, val={len(val_sources)}")

# ── Assign sources per province ─────────────────────────────────────
def assign_sources(source_list, needed_per_prov, label):
    """Assign source images for each province, cycling if needed."""
    assignments = []
    used_texts = set()
    
    for prov in ALL_PROVINCES:
        needed = needed_per_prov
        # Filter sources that have this province (or use all if none)
        prov_sources = [s for s in source_list if s['text'][0] == prov]
        if not prov_sources:
            prov_sources = source_list  # fallback to any source
        
        for i in range(needed):
            src = prov_sources[i % len(prov_sources)]
            # Generate unique text
            while True:
                letters = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
                alnum = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
                digits = list('0123456789')
                t = prov + random.choice(letters) + random.choice(['D','F']) \
                    + random.choice(alnum) + ''.join(random.choice(digits) for _ in range(4))
                if t not in used_texts:
                    used_texts.add(t)
                    break
            assignments.append((src, t))
    
    print(f"  {label}: {len(assignments)} assignments ({len(set(a['img_path'] for a,_ in assignments))} unique sources)")
    return assignments

train_assign = assign_sources(train_sources, TRAIN_PER_PROV, 'train')
val_assign = assign_sources(val_sources, VAL_PER_PROV, 'val')

# ── Init generation pipeline ────────────────────────────────────────
print("\nInit generation pipeline...")
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

def build_plate(text):
    ci = _chars_gen.generate_images([text])[0]
    ai = _aug.augment(ci, horizontal_sight_direction='mid', vertical_sight_direction='mid')
    return cv2.resize(ai, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

def match_brightness(new_bgr, orig_patch_bgr):
    new_lab = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h_p, w_p = orig_lab.shape[:2]
    cy, cx = h_p//2, w_p//2
    roi_h, roi_w = int(h_p*0.8), int(w_p*0.8)
    L_new = new_lab[cy-roi_h//2:cy+roi_h//2, cx-roi_w//2:cx+roi_w//2, 0]
    L_orig = orig_lab[cy-roi_h//2:cy+roi_h//2, cx-roi_w//2:cx+roi_w//2, 0]
    mn, sn = L_new.mean(), L_new.std()+1e-6
    mo, so = L_orig.mean(), L_orig.std()+1e-6
    ratio = max(0.5, min(2.0, so/sn))
    new_lab[:,:,0] = np.clip((new_lab[:,:,0] - mn)*ratio + mo, 0, 255)
    return cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

# ── Generate ────────────────────────────────────────────────────────
def generate_one(src_rec, new_text, split):
    img_path = src_rec['img_path']
    gt_quad = np.array(src_rec['gt_quad'], dtype=np.float32)
    gt_quad = order_quad_points(gt_quad)  # FIXED
    pose_quad = np.array(src_rec['pose_quad'], dtype=np.float32)
    
    img = cv2.imread(img_path)
    if img is None: return None, None
    h, w = img.shape[:2]
    
    # Forward warp (GT quad, fixed order)
    M_fwd = cv2.getPerspectiveTransform(gt_quad, SRC_RECT)
    plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Build + match
    raw = build_plate(new_text)
    matched = match_brightness(raw, plate_orig)
    
    # Inverse warp
    M_inv = cv2.getPerspectiveTransform(SRC_RECT, gt_quad)
    warped = cv2.warpPerspective(matched, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Mask + blend
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [gt_quad.round().astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    result = (img.astype(np.float32) * (1 - mask[:,:,None]) + warped.astype(np.float32) * mask[:,:,None]).clip(0, 255).astype(np.uint8)
    
    # Filename
    stem = Path(img_path).stem
    fname = f"{stem}_extv4_{new_text}.jpg"
    out_dir = IMG_TRAIN if split == 'train' else IMG_VAL
    out_path = str(out_dir / fname)
    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Manifest row (Pose quad for training)
    row = {
        'img_path': out_path, 'text': new_text, 'family': 'green8',
        'source': 'green_ccpd2020_replace_extreme_v4',
        'split': split,
        'has_quad': '1', 'can_parse_ccpd_geom': '1', 'can_perspective': '1',
        'preprocess_group': 'ccpd_board',
        'quad_source': 'pose_v3', 'bbox_source': 'pose_v3',
        'quad_1x': f'{pose_quad[0][0]:.1f}', 'quad_1y': f'{pose_quad[0][1]:.1f}',
        'quad_2x': f'{pose_quad[1][0]:.1f}', 'quad_2y': f'{pose_quad[1][1]:.1f}',
        'quad_3x': f'{pose_quad[2][0]:.1f}', 'quad_3y': f'{pose_quad[2][1]:.1f}',
        'quad_4x': f'{pose_quad[3][0]:.1f}', 'quad_4y': f'{pose_quad[3][1]:.1f}',
        'ocr_quad_pad_ratio': '0.0',
    }
    return row, {'source_img': img_path, 'generated_img': out_path, 'province': new_text[0],
                  'new_text': new_text, 'split': split, 'pose_quad': pose_quad.tolist(),
                  'angle': src_rec['angle']}

print("\nGenerating...")
all_rows = []
source_log = []
t0 = time.time()

for split, assignments, img_dir in [('train', train_assign, IMG_TRAIN), ('val', val_assign, IMG_VAL)]:
    for idx, (src_rec, new_text) in enumerate(assignments):
        row, log_entry = generate_one(src_rec, new_text, split)
        if row is None:
            print(f"  [{idx}] SKIP: {src_rec['img_path']}")
            continue
        all_rows.append(row)
        source_log.append(log_entry)
        
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(assignments) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(assignments)}] {split} {rate:.1f}/s, ETA {eta:.0f}s")

print(f"\nGenerated: {len(all_rows)} total")

# ── Source log ──────────────────────────────────────────────────────
jsonl_path = OUT_DIR / 'source_log.jsonl'
with open(jsonl_path, 'w') as f:
    for s in source_log:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f"Source log: {jsonl_path}")

# ── Manifest CSV ────────────────────────────────────────────────────
MANIFEST_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'preprocess_group', 'quad_source', 'bbox_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_quad_pad_ratio',
]
train_rows = [r for r in all_rows if r['split'] == 'train']
val_rows = [r for r in all_rows if r['split'] == 'val']

for split, rows, name in [('train', train_rows, 'train_extreme_v3.csv'),
                           ('val', val_rows, 'val_extreme_v3.csv')]:
    out_path = MANIFEST_DIR / name
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Manifest: {out_path} ({len(rows)} rows)")

# ── Province distribution ───────────────────────────────────────────
prov_cnt = Counter(r['text'][0] for r in all_rows)
print(f"\nProvince distribution ({len(prov_cnt)} provinces):")
for p in ALL_PROVINCES:
    c = prov_cnt.get(p, 0)
    print(f"  {p}: {c}", end="  " if (ALL_PROVINCES.index(p)+1) % 8 else "\n")
print()

# ── Angle report ────────────────────────────────────────────────────
angles = [s['angle'] for s in source_log]
print(f"\nAngle range: {min(angles):.2f}° - {max(angles):.2f}°")
print(f"Angle mean: {np.mean(angles):.2f}°")
print(f"Angle median: {np.median(angles):.2f}°")
print(f"Angle p25: {np.percentile(angles, 25):.2f}°")
print(f"Angle p75: {np.percentile(angles, 75):.2f}°")

# ── Verify source isolation ─────────────────────────────────────────
train_src_paths = set(s['source_img'] for s in source_log if s['split'] == 'train')
val_src_paths = set(s['source_img'] for s in source_log if s['split'] == 'val')
overlap = train_src_paths & val_src_paths
print(f"\nSource isolation: train={len(train_src_paths)}, val={len(val_src_paths)}, overlap={len(overlap)} {'❌' if overlap else '✅'}")
