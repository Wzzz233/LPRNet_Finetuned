#!/usr/bin/env python3
"""
Pose quad replacement QA preview.
For 12 diverse CCPD2020 green samples:
1. Original + GT quad + pose quad overlay
2. Replaced image (text replaced via GT quad)
3. GT warp → 94x24
4. Pose warp → 94x24
5. Gray3 + letterbox 94x24 (LPRNet input via pose quad)
"""

import csv, os, sys, math, random
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from load_data import parse_ccpd_quad_from_name, order_quad_points, \
    prepare_board_ocr_input_from_quad_bgr888
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

POSE_WEIGHT = ROOT / 'experiments/yolov8n-pos/weights/best.pt'
CCPD2020_LABEL = ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv'
OUT_DIR = ROOT / 'datasets' / 'pose_quad_replacement_qa'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pose_quad_replacement_qa')
WIN_QA.mkdir(parents=True, exist_ok=True)

random.seed(20260503)

# ── Pipeline setup ──────────────────────────────────────────────
print("Loading pose model...")
pose_model = YOLO(str(POSE_WEIGHT))

print("Init generation pipeline...")
sys.path.insert(0, str(ROOT / 'src' / 'utils'))
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
    cy, cx = h // 2, w // 2
    roi_h, roi_w = int(h * 0.8), int(w * 0.8)
    L_new = new_lab[cy - roi_h // 2:cy + roi_h // 2, cx - roi_w // 2:cx + roi_w // 2, 0]
    L_orig = orig_lab[cy - roi_h // 2:cy + roi_h // 2, cx - roi_w // 2:cx + roi_w // 2, 0]
    mn, sn = L_new.mean(), L_new.std() + 1e-6
    mo, so = L_orig.mean(), L_orig.std() + 1e-6
    ratio = max(0.5, min(2.0, so / sn))
    new_lab[:,:,0] = np.clip((new_lab[:,:,0] - mn) * ratio + mo, 0, 255)
    return cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

# ── Load CCPD2020 test samples ─────────────────────────────────
print("Loading CCPD2020 test set...")
samples = []
with open(CCPD2020_LABEL, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        p = row['img_path']
        if not os.path.exists(p):
            continue
        q = parse_ccpd_quad_from_name(p)
        if q is None:
            continue
        samples.append({'path': p, 'text': row['text'], 'gt_quad': q})

print(f"Total: {len(samples)}")

# Sample 12 diverse provinces
ALL_PROVINCES = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                 '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                 '桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
prov_counter = Counter(s['text'][0] for s in samples)
print(f"Province distribution: {dict(prov_counter.most_common(10))}")

# Stratified sample: pick top provinces with enough samples
selected = []
target_provs = random.sample([p for p in ALL_PROVINCES if prov_counter.get(p, 0) >= 5],
                               min(12, len([p for p in ALL_PROVINCES if prov_counter.get(p, 0) >= 5])))
for prov in target_provs:
    pool = [s for s in samples if s['text'][0] == prov]
    selected.append(random.choice(pool))

print(f"QA samples: {len(selected)}")

# ── Process each sample ────────────────────────────────────────
qa_rows = []
for idx, s in enumerate(selected):
    print(f"  [{idx+1}/{len(selected)}] {s['text'][:8]}")
    
    img_bgr = cv2.imread(s['path'])
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]
    
    gt_quad_ordered = order_quad_points(s['gt_quad'])
    
    # Run pose model
    r = pose_model(s['path'], imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    pose_quad = None
    if r.keypoints is not None and r.keypoints.xy is not None:
        kps = r.keypoints.xy.cpu().numpy()
        if kps.ndim == 3 and kps.shape[0] > 0 and kps.shape[1] == 4:
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            best = int(np.argmax(confs.mean(axis=1))) if confs is not None and confs.ndim > 1 else 0
            pose_quad = order_quad_points(kps[best])
    
    if pose_quad is None:
        continue
    
    # 1. Forward warp (GT quad) → get plate text region
    M_fwd_gt = cv2.getPerspectiveTransform(gt_quad_ordered, SRC_RECT)
    plate_orig = cv2.warpPerspective(img_bgr, M_fwd_gt, (CANVAS_W, CANVAS_H),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # 2. Generate new text
    new_text = s['text']  # keep original text for QA, just show pipeline
    
    # 3. Match brightness
    raw = build_plate(new_text)
    matched = match_brightness(raw, plate_orig)
    
    # 4. Inverse warp (GT quad) → put text back
    M_inv_gt = cv2.getPerspectiveTransform(SRC_RECT, gt_quad_ordered)
    warped_text = cv2.warpPerspective(matched, M_inv_gt, (w, h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # 5. Mask + blend
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [gt_quad_ordered.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    replaced = img_bgr.astype(np.float32)
    for c in range(3):
        replaced[:,:,c] = warped_text[:,:,c] * mask + replaced[:,:,c] * (1.0 - mask)
    replaced = np.clip(replaced, 0, 255).astype(np.uint8)
    
    # 6. Training inputs:
    # GT warp → 94x24 (how GT quad looks after training pipeline)
    gt_input, _, gt_warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        replaced.copy(), gt_quad_ordered, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    
    # Pose warp → 94x24 (what training actually receives when using pose quad)
    pose_input, _, pose_warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        replaced.copy(), pose_quad, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    
    # Also warp without preprocessing (original warp)
    _, _, pose_warped_raw, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        replaced.copy(), pose_quad, 94, 24,
        resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='none', channel_order='bgr', quad_pad_ratio=0.0)
    
    # ── Build contact sheet cell ─────────────────────────────
    # Layout: 5 images in a row
    # [Quad overlay] [Replaced] [GT warp] [Pose warp] [Pose gray3]
    
    cell_images = []
    
    # Original with quads overlay
    overlay = img_bgr.copy()
    cv2.polylines(overlay, [gt_quad_ordered.astype(np.int32).reshape(-1,1,2)], True, (0,255,0), 2)
    if pose_quad is not None:
        cv2.polylines(overlay, [pose_quad.astype(np.int32).reshape(-1,1,2)], True, (255,0,0), 2)
    # Resize for display
    sf = 180 / max(h, w)
    overlay_small = cv2.resize(overlay, (int(w*sf), int(h*sf)))
    cell_images.append(('Orig+Quads\nG:GT B:Pose', overlay_small))
    
    # Replaced image
    replaced_small = cv2.resize(replaced, (int(w*sf), int(h*sf)))
    cell_images.append(('Replaced', replaced_small))
    
    # GT warp (raw)
    gt_label = f'GT warp {s["text"]}'
    cell_images.append((gt_label, gt_warped))
    
    # Pose warp (raw)
    pose_label = f'Pose warp {new_text}'
    cell_images.append((pose_label, pose_warped_raw))
    
    # Pose gray3 94x24 (LPRNet input)
    cell_images.append(('Pose gray3 94x24', pose_input))
    
    # Create row canvas
    max_h = max(img.shape[0] for _, img in cell_images)
    total_w = sum(img.shape[1] + 5 for _, img in cell_images) + 10
    row_canvas = np.ones((max_h + 35, total_w, 3), dtype=np.uint8) * 40
    
    cx = 5
    for label_text, img in cell_images:
        ih, iw = img.shape[:2]
        y_off = (max_h - ih) // 2 + 15
        row_canvas[y_off:y_off+ih, cx:cx+iw] = img
        # Label
        cv2.putText(row_canvas, label_text.replace('\n', ' '),
                    (cx, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
        cx += iw + 5
    
    qa_rows.append(row_canvas)

# ── Composite all rows ─────────────────────────────────────────
max_w = max(r.shape[1] for r in qa_rows) if qa_rows else 800
total_h = sum(r.shape[0] for r in qa_rows) + 10
final = np.ones((total_h, max_w, 3), dtype=np.uint8) * 25

y = 5
for r in qa_rows:
    final[y:y+r.shape[0], :r.shape[1]] = r
    y += r.shape[0]

out_path = str(OUT_DIR / 'qa_pose_replacement.jpg')
cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\nSaved: {out_path}")

# Copy to Windows
win_path = str(WIN_QA / 'qa_pose_replacement.jpg')
cv2.imwrite(win_path, final, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"Copied: {win_path}")

# Also save each sample individually
for idx, row_img in enumerate(qa_rows):
    sp = str(OUT_DIR / f'sample_{idx:02d}.jpg')
    cv2.imwrite(sp, row_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    wp = str(WIN_QA / f'sample_{idx:02d}.jpg')
    cv2.imwrite(wp, row_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

print(f"Done. {len(qa_rows)} samples.")
