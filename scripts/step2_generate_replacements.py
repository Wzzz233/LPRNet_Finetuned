#!/usr/bin/env python3
"""Step 2: Generate province-balanced replacement data.
- Uses GT quad for text placement
- Records Pose quad in manifest
- 31 provinces × 100 samples each = 3100 total
"""

import csv, json, os, sys, math, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'utils'))

from load_data import prepare_board_ocr_input_from_quad_bgr888, order_quad_points
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

# ── Paths ──────────────────────────────────────────────────────────
POSE_QUADS = ROOT / 'datasets/ccpd2020_pose_quads' / 'pose_quads.jsonl'
OUT_DIR = ROOT / 'datasets' / 'ccpd2020_replace_pose_v3'
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT_DIR / 'images' / 'train'
IMG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR = ROOT / 'manifests' / 'ccpd2020_replace_pose_v3'
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR = OUT_DIR / 'qa'
QA_DIR.mkdir(parents=True, exist_ok=True)
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2020_replace_pose_v3')
WIN_QA.mkdir(parents=True, exist_ok=True)

random.seed(20260503)

# ── Province setup ────────────────────────────────────────────────
ALL_PROVINCES = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                 '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                 '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
SAMPLES_PER_PROV = 100
TOTAL_PER_PROV = {p: SAMPLES_PER_PROV for p in ALL_PROVINCES}

# ── Init generation pipeline ──────────────────────────────────────
print("Init generation pipeline...")
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
    h, w = orig_lab.shape[:2]
    cy, cx = h//2, w//2
    roi_h, roi_w = int(h*0.8), int(w*0.8)
    L_new = new_lab[cy-roi_h//2:cy+roi_h//2, cx-roi_w//2:cx+roi_w//2, 0]
    L_orig = orig_lab[cy-roi_h//2:cy+roi_h//2, cx-roi_w//2:cx+roi_w//2, 0]
    mn, sn = L_new.mean(), L_new.std()+1e-6
    mo, so = L_orig.mean(), L_orig.std()+1e-6
    ratio = max(0.5, min(2.0, so/sn))
    new_lab[:,:,0] = np.clip((new_lab[:,:,0] - mn)*ratio + mo, 0, 255)
    return cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')

def make_new_text(province, used):
    while True:
        t = province + random.choice(LETTERS) + random.choice(['D','F']) \
            + random.choice(ALNUM) + ''.join(random.choice(DIGITS) for _ in range(4))
        if t not in used:
            used.add(t)
            return t

def angle_score_from_quad(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i-1)%4]; b = p[i]; c = p[(i+1)%4]
        v1, v2 = a - b, c - b
        d = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angles.append(abs(math.degrees(math.acos(max(-1,min(1,d)))) - 90))
    return float(np.mean(angles))

# ── Load pose quads ───────────────────────────────────────────────
print("Loading pose quad results...")
pose_data = [json.loads(l) for l in open(POSE_QUADS)]

# Add angle score
for r in pose_data:
    r['angle'] = angle_score_from_quad(np.array(r['gt_quad']))

# Separate train and test
test_data = [r for r in pose_data if r['split'] == 'test']
train_data = [r for r in pose_data if r['split'] == 'train']

print(f"Test samples with pose quads: {len(test_data)}")
print(f"Train samples with pose quads: {len(train_data)}")

# Sort test data by angle (hardest first for selection)
test_data.sort(key=lambda x: -x['angle'])

# ── Assign source images per province ─────────────────────────────
print("\nAssigning source images per province...")

# Group test data by province
by_prov = {}
for r in test_data:
    prov = r['text'][0]
    by_prov.setdefault(prov, []).append(r)

# For each province, gather source candidates
# For provinces with <100, reuse images; for provinces with >=100, pick top 100 by angle
source_map = []  # (source_record, province) pairs
used_texts = set()

for prov in ALL_PROVINCES:
    needed = TOTAL_PER_PROV[prov]
    available = by_prov.get(prov, [])
    
    if len(available) == 0:
        # No source images for this province — use 皖 samples as fallback
        # The text will be replaced to the target province anyway
        fallback = by_prov.get('皖', [])
        if not fallback:
            print(f"  {prov}: {needed} — SKIP (no sources at all)")
            continue
        selected = [fallback[i % len(fallback)] for i in range(needed)]
        for s in selected:
            source_map.append((s, prov))
        print(f"  {prov}: {needed} (0 available, using fallback sources)")
    elif len(available) >= needed:
        selected = available[:needed]
        for s in selected:
            source_map.append((s, prov))
        print(f"  {prov}: {needed} (from {len(available)} available, hardest)")
    else:
        print(f"  {prov}: {needed} (only {len(available)} available, cycling)")
        for i in range(needed):
            source_map.append((available[i % len(available)], prov))

# Shuffle for randomness
random.shuffle(source_map)
print(f"\nTotal replacements to generate: {len(source_map)}")

# ── Generate replacements ─────────────────────────────────────────
print("\nGenerating replacement images...")

MANIFEST_FIELDS = [
    'img_path', 'text', 'family', 'source', 'split',
    'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'preprocess_group', 'ocr_crop_mode', 'ocr_resize_mode',
    'ocr_resize_kernel', 'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio',
    'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4',
]

train_rows = []
val_rows = []
source_log = []  # which original test images were used

t0 = time.time()
for idx, (src_rec, prov) in enumerate(source_map):
    img_path = src_rec['img_path']
    gt_quad_raw = np.array(src_rec['gt_quad'])
    # CCPD raw quad is [BR, BL, TL, TR] → reorder to [TL, TR, BR, BL]
    # NOTE: do NOT use order_quad_points() — its diff heuristic swaps BL/TR for CCPD ordering.
    gt_quad = np.array([gt_quad_raw[2], gt_quad_raw[3], gt_quad_raw[0], gt_quad_raw[1]], dtype=np.float32)
    # Pose quad from YOLO is already [TL, TR, BR, BL], use as-is
    pose_quad = np.array(src_rec['pose_quad'])
    
    split = 'train' if idx < int(len(source_map) * 0.9) else 'val'
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read {img_path}")
        h, w = img.shape[:2]
        
        # Forward warp (GT quad) → get plate region
        M_fwd = cv2.getPerspectiveTransform(gt_quad.astype(np.float32), SRC_RECT)
        plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Generate new text
        new_text = make_new_text(prov, used_texts)
        
        # Build + match plate
        raw = build_plate(new_text)
        matched = match_brightness(raw, plate_orig)
        
        # Inverse warp back (GT quad)
        M_inv = cv2.getPerspectiveTransform(SRC_RECT, gt_quad.astype(np.float32))
        warped = cv2.warpPerspective(matched, M_inv, (w, h),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Mask + blend
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [gt_quad.round().astype(np.int32)], 255)
        mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
        result = (img.astype(np.float32) * (1 - mask[:,:,None]) +
                  warped.astype(np.float32) * mask[:,:,None]).clip(0, 255).astype(np.uint8)
        
        # Save
        stem = Path(img_path).stem
        fname = f"{stem}_posev3_{new_text}.jpg"
        if split == 'train':
            out_path = IMG_DIR / fname
        else:
            val_dir = OUT_DIR / 'images' / 'val'
            val_dir.mkdir(parents=True, exist_ok=True)
            out_path = val_dir / fname
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Manifest row — record POSE quad for training warp
        row = {
            'img_path': str(out_path),
            'text': new_text,
            'family': 'green8',
            'source': 'green_ccpd2020_replace_pose_v3',
            'split': split,
            'has_quad': '1',
            'can_parse_ccpd_geom': '0',
            'can_perspective': '1',
            'preprocess_group': 'ccpd_board',
            'ocr_crop_mode': 'obb_warp',
            'ocr_resize_mode': 'letterbox',
            'ocr_resize_kernel': 'nn',
            'ocr_preproc': 'gray3',
            'ocr_channel_order': 'bgr',
            'ocr_quad_pad_ratio': '0.0',
            'x1': f'{pose_quad[0][0]:.2f}', 'y1': f'{pose_quad[0][1]:.2f}',
            'x2': f'{pose_quad[1][0]:.2f}', 'y2': f'{pose_quad[1][1]:.2f}',
            'x3': f'{pose_quad[2][0]:.2f}', 'y3': f'{pose_quad[2][1]:.2f}',
            'x4': f'{pose_quad[3][0]:.2f}', 'y4': f'{pose_quad[3][1]:.2f}',
        }
        if split == 'train':
            train_rows.append(row)
        else:
            val_rows.append(row)
        
        # Source log
        source_log.append({
            'source_img': img_path,
            'generated_img': str(out_path),
            'province': prov,
            'new_text': new_text,
            'split': split,
            'pose_quad': pose_quad.tolist(),
        })
        
    except Exception as e:
        print(f"  [{idx}] ERROR: {img_path}: {e}")
    
    if (idx + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(source_map) - idx - 1) / rate if rate > 0 else 0
        print(f"  [{idx+1}/{len(source_map)}] {rate:.1f}/s, ETA {eta/60:.0f}min")

print(f"\nGenerated: {len(train_rows)} train + {len(val_rows)} val")

# ── Source manifest ────────────────────────────────────────────────
source_csv = OUT_DIR / 'source_manifest.csv'
with open(source_csv, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['source_img', 'generated_img', 'province', 'new_text', 'split'])
    w.writeheader()
    for s in source_log:
        w.writerow({k: s[k] for k in ['source_img', 'generated_img', 'province', 'new_text', 'split']})
print(f"Source manifest: {source_csv} ({len(source_log)} entries)")

# Also save the full source log with pose quads
source_jsonl = OUT_DIR / 'source_log.jsonl'
with open(source_jsonl, 'w') as f:
    for s in source_log:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

# ── Write manifests ────────────────────────────────────────────────
for split, rows, name in [
    ('train', train_rows, 'train_ccpd2020_replace_pose_v3.csv'),
    ('val', val_rows, 'val_ccpd2020_replace_pose_v3.csv'),
]:
    out_path = MANIFEST_DIR / name
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Manifest: {out_path} ({len(rows)} rows)")

# ── Province distribution report ──────────────────────────────────
prov_count = Counter(r['text'][0] for r in train_rows + val_rows)
print(f"\nProvince distribution:")
for p in ALL_PROVINCES:
    print(f"  {p}: {prov_count.get(p, 0)}")
print(f"  Total: {sum(prov_count.values())}")

# ── Quick QA contact sheet for a few samples ──────────────────────
import random as rnd
qa_samples = rnd.sample(source_log, min(6, len(source_log)))
for i, s in enumerate(qa_samples):
    replaced_img = cv2.imread(s['generated_img'])
    if replaced_img is None:
        continue
    h, w = replaced_img.shape[:2]
    
    # Load original for overlay
    orig = cv2.imread(s['source_img'])
    if orig is None:
        continue
    sf = 200 / max(orig.shape[0], orig.shape[1])
    orig_small = cv2.resize(orig, (int(orig.shape[1]*sf), int(orig.shape[0]*sf)))
    
    # Draw pose quad on original
    pq = np.array(s['pose_quad'])
    cv2.polylines(orig_small, [(pq*sf).round().astype(np.int32).reshape(-1,1,2)], True, (255,0,0), 2)
    
    # Replaced small
    replaced_small = cv2.resize(replaced_img, (int(w*sf), int(h*sf)))
    
    # Pose warp gray3 (training input)
    pose_quad = np.array(s['pose_quad'])
    try:
        g3, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
            replaced_img, pose_quad, 94, 24,
            resize_mode='letterbox', resize_kernel='nn',
            preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    except:
        g3 = np.zeros((24, 94, 3), dtype=np.uint8)
    
    # Stack
    vis = np.ones((max(orig_small.shape[0], 50), orig_small.shape[1] + replaced_small.shape[1] + 100, 3), dtype=np.uint8) * 30
    vis[:orig_small.shape[0], :orig_small.shape[1]] = orig_small
    vis[:replaced_small.shape[0], orig_small.shape[1]+5:orig_small.shape[1]+5+replaced_small.shape[1]] = replaced_small
    vis[:g3.shape[0], orig_small.shape[1]+replaced_small.shape[1]+10:orig_small.shape[1]+replaced_small.shape[1]+10+g3.shape[1]] = g3
    
    # Labels
    label_text = f'{s["province"]} {s["new_text"]}'
    cv2.putText(vis, label_text, (5, orig_small.shape[0]+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    
    qa_path = QA_DIR / f'qa_{i:02d}_{s["province"]}.jpg'
    cv2.imwrite(str(qa_path), vis)
    cv2.imwrite(str(WIN_QA / qa_path.name), vis)

print(f"\nQA samples: {QA_DIR}")
print(f"Windows: {WIN_QA}")

# Summary
summary = {
    'n_total': len(train_rows) + len(val_rows),
    'n_train': len(train_rows),
    'n_val': len(val_rows),
    'province_distribution': dict(prov_count),
    'samples_per_province': SAMPLES_PER_PROV,
    'source_domain': 'CCPD2020_test',
    'pipeline': 'pose_v3_gt_text_pose_quad',
}
with open(OUT_DIR / 'generation_meta.json', 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSummary: {OUT_DIR / 'generation_meta.json'}")
print("Done.")
