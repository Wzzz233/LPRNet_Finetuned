#!/usr/bin/env python3
"""Generate QA samples for Pose replacement - simplified output."""

import csv, os, sys, math, random
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'utils'))
from load_data import parse_ccpd_quad_from_name, order_quad_points, \
    prepare_board_ocr_input_from_quad_bgr888
from generate_chars_image import CharsImageGenerator
from generate_plate_template import LicensePlateImageGenerator
from augment_image import ImageAugmentation

POSE_WEIGHT = ROOT / 'experiments/yolov8n-pos/weights/best.pt'
CCPD2020_LABEL = ROOT / 'labels/curriculum_gray3/ccpd2020_test.csv'
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pose_replacement_qa')
WIN_QA.mkdir(parents=True, exist_ok=True)
random.seed(20260503)

print("Init...")
pose_model = YOLO(str(POSE_WEIGHT))
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

def angle_score(quad):
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []
    for i in range(4):
        a = p[(i-1)%4]; b = p[i]; c = p[(i+1)%4]
        v1, v2 = a - b, c - b
        d = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angles.append(abs(math.degrees(math.acos(max(-1,min(1,d)))) - 90))
    return float(np.mean(angles))

# Load
samples = []
with open(CCPD2020_LABEL, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        p = row['img_path']
        if not os.path.exists(p):
            continue
        q = parse_ccpd_quad_from_name(p)
        if q is None:
            continue
        a = angle_score(q)
        samples.append({'path': p, 'text': row['text'], 'gt_quad': q, 'angle': a})

samples.sort(key=lambda x: -x['angle'])
extreme = samples[:6]
non_ah = [s for s in samples if s['text'][0] != '皖' and s not in extreme]
random.shuffle(non_ah)
mild = [s for s in samples if s['text'][0] == '皖' and s['angle'] < 20]
random.shuffle(mild)
selected = extreme[:4] + non_ah[:4] + mild[:4]
random.shuffle(selected)
print(f"Selected {len(selected)} samples")

LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
ALNUM = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
DIGITS = list('0123456789')

def make_new_text(province):
    return province + random.choice(LETTERS) + random.choice(['D','F']) \
        + random.choice(ALNUM) + ''.join(random.choice(DIGITS) for _ in range(4))

used_texts = set(s['text'] for s in selected)

for idx, s in enumerate(selected):
    print(f"  [{idx+1}] {s['text'][:8]} a={s['angle']:.1f}")
    
    img = cv2.imread(s['path'])
    if img is None: continue
    h, w = img.shape[:2]
    gt_quad = order_quad_points(s['gt_quad'])
    
    r = pose_model(s['path'], imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    pose_quad = None
    if r.keypoints is not None and r.keypoints.xy is not None:
        kps = r.keypoints.xy.cpu().numpy()
        if kps.ndim == 3 and kps.shape[0] > 0 and kps.shape[1] == 4:
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            best = int(np.argmax(confs.mean(axis=1))) if confs is not None and confs.ndim > 1 else 0
            pose_quad = order_quad_points(kps[best])
    if pose_quad is None: continue
    
    new_text = make_new_text(s['text'][0])
    while new_text in used_texts:
        new_text = make_new_text(s['text'][0])
    used_texts.add(new_text)
    
    # Replacement
    M_fwd = cv2.getPerspectiveTransform(gt_quad, SRC_RECT)
    plate_orig = cv2.warpPerspective(img, M_fwd, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    raw = build_plate(new_text)
    matched = match_brightness(raw, plate_orig)
    M_inv = cv2.getPerspectiveTransform(SRC_RECT, gt_quad)
    warped_text = cv2.warpPerspective(matched, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [gt_quad.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    replaced = (img.astype(np.float32) * (1 - mask[:,:,None]) + warped_text.astype(np.float32) * mask[:,:,None]).clip(0, 255).astype(np.uint8)
    
    # Warps from replaced image
    _, _, gt_warp_raw, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        replaced, gt_quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='none', channel_order='bgr', quad_pad_ratio=0.0)
    gt_g3, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        replaced, gt_quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    _, _, pose_warp_raw, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        replaced, pose_quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='none', channel_order='bgr', quad_pad_ratio=0.0)
    pose_g3, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
        replaced, pose_quad, 94, 24, resize_mode='letterbox', resize_kernel='nn',
        preproc_mode='gray3', channel_order='bgr', quad_pad_ratio=0.0)
    
    # Save individual panels
    sample_dir = WIN_QA / f'sample_{idx:02d}'
    sample_dir.mkdir(exist_ok=True)
    
    # Original + overlay
    sf = 200 / max(h, w)
    overlay = cv2.resize(img, (int(w*sf), int(h*sf)))
    cv2.polylines(overlay, [(gt_quad*sf).round().astype(np.int32).reshape(-1,1,2)], True, (0,255,0), 2)
    cv2.polylines(overlay, [(pose_quad*sf).round().astype(np.int32).reshape(-1,1,2)], True, (255,0,0), 2)
    cv2.imwrite(str(sample_dir / '01_original_with_quads.jpg'), overlay)
    
    # Replaced
    replaced_small = cv2.resize(replaced, (int(w*sf), int(h*sf)))
    cv2.imwrite(str(sample_dir / '02_replaced.jpg'), replaced_small)
    
    # GT warp raw
    cv2.imwrite(str(sample_dir / '03_GT_warp_raw.jpg'), gt_warp_raw)
    # GT gray3
    cv2.imwrite(str(sample_dir / '04_GT_gray3.jpg'), gt_g3)
    # Pose warp raw
    cv2.imwrite(str(sample_dir / '05_Pose_warp_raw.jpg'), pose_warp_raw)
    # Pose gray3 (TRAINING INPUT)
    cv2.imwrite(str(sample_dir / '06_Pose_gray3_TRAINING_INPUT.jpg'), pose_g3)
    
    # Info text
    with open(str(sample_dir / 'info.txt'), 'w') as f:
        f.write(f"Original text: {s['text']}\n")
        f.write(f"Replaced text: {new_text}\n")
        f.write(f"Angle score: {s['angle']:.1f}°\n")
        f.write(f"Image: {os.path.basename(s['path'])}\n")

print(f"\nDone. {len(selected)} samples in {WIN_QA}")
print("Each sample has its own subfolder with individual panel images.")
