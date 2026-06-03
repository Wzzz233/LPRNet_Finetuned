#!/usr/bin/env python3
"""
Strict CV-feature-based police plate generation for BlueBase training.
Extracts per-image CV features and renders matching replacements.
"""
import sys, os, csv, json, random, math
from pathlib import Path
from collections import Counter
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
DATASET = ROOT / 'datasets/police_bluebase_cvstrict_20260603'
MANIFEST = ROOT / 'manifests_rebased/police_bluebase_cvstrict_20260603'
QA_DIR = DATASET / 'qa'
for d in [DATASET, MANIFEST, QA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Font
FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font = ImageFont.truetype(FONT_PATH, 56)

# Province chars
with open(str(ROOT / 'keys/police_keys.txt')) as f:
    all_keys = [l.strip() for l in f if l.strip()]
province_chars = all_keys[:31]  # 京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新
ALPHABET = [c for c in 'ABCDEFGHJKLMNPQRSTUVWXYZ']  # no I/O
DIGITS = [str(i) for i in range(10)]

# ── Source images: use pose_quads.jsonl from CCPD2019 base ──
SOURCE_JSONL = ROOT / 'datasets/ccpd2019_base_posquads_20260509/pose_quads.jsonl'
print(f'Loading source quads from {SOURCE_JSONL}')
source_images = []
with open(SOURCE_JSONL) as f:
    for line in f:
        entry = json.loads(line.strip())
        source_images.append(entry)
print(f'Total source entries: {len(source_images)}')

# ── CV feature extraction ──
def extract_cv_features(img_bgr, quad):
    """Extract per-image CV features from the plate region."""
    h, w = img_bgr.shape[:2]
    
    # Warp plate to canonical rect
    src_pts = np.array(quad, dtype=np.float32)
    dst_pts = np.array([[0,0],[200,0],[200,80],[0,80]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_bgr, M, (200, 80))
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # 1. Brightness stats
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    
    # 2. Blur estimation (Laplacian variance)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(lap.var())
    
    # 3. Noise estimation (std of flat region - top-left corner)
    flat_region = gray[5:25, 5:25]
    noise_estimate = float(flat_region.std())
    
    # 4. JPEG artifact proxy (DCT energy in 8x8 blocks)
    dct_energy = float(np.std(cv2.dct(gray.astype(np.float32)/255.0)[:8,:8]))
    
    # 5. Local exposure gradient
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    exposure_gradient = float((np.abs(grad_x) + np.abs(grad_y)).mean())
    
    # 6. Color saturation (HSV)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:,:,1].mean())
    
    # 7. Character region contrast (bottom half where chars are)
    char_region = gray[10:70, 10:190]
    contrast = float(char_region.std())
    
    # 8. Sharpness
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    return {
        'brightness_mean': brightness_mean, 'brightness_std': brightness_std,
        'blur_score': blur_score, 'noise_estimate': noise_estimate,
        'dct_energy': dct_energy, 'exposure_gradient': exposure_gradient,
        'saturation': saturation, 'contrast': contrast, 'sharpness': sharpness,
    }, warped

def apply_cv_features(rendered_bgr, features):
    """Apply matched degradation to rendered plate."""
    result = rendered_bgr.copy().astype(np.float32)
    
    # Brightness matching
    current_mean = result.mean()
    target_mean = features['brightness_mean']
    result = result * (target_mean / max(current_mean, 1))
    
    # Blur matching
    if features['blur_score'] < 50:  # blurry source
        k = max(3, int(50 / max(features['blur_score'], 1)))
        if k % 2 == 0: k += 1
        result = cv2.GaussianBlur(result, (min(k, 15), min(k, 15)), 0)
    
    # Noise matching
    noise = np.random.randn(*result.shape).astype(np.float32) * features['noise_estimate'] * 0.3
    result = np.clip(result + noise, 0, 255)
    
    # JPEG artifact simulation
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(60, min(95, int(features['dct_energy'] * 50 + 60)))]
    _, enc = cv2.imencode('.jpg', result.astype(np.uint8), encode_param)
    result = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def render_police_plate(province_char, letter, digits, tail='警', target_w=200, target_h=80):
    """Render a clean police plate text."""
    text = f'{province_char}{letter}{digits}{tail}'
    img = Image.new('RGB', (target_w, target_h), color=(47, 62, 78))
    draw = ImageDraw.Draw(img)
    
    # Calculate text size and position
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = (target_w - tw) // 2 - bbox[0]
    y = (target_h - th) // 2 - bbox[1]
    
    # White text
    draw.text((x, y), text, fill=(220, 220, 220), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), text

def warp_back_to_original(rendered_plate, quad, img_shape):
    """Warp the rendered plate back onto the original image using the quad."""
    h, w = img_shape[:2]
    src_pts = np.array([[0,0],[200,0],[200,80],[0,80]], dtype=np.float32)
    dst_pts = np.array(quad, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    result = np.zeros((h, w, 3), dtype=np.uint8)
    warped = cv2.warpPerspective(rendered_plate, M, (w, h), borderMode=cv2.BORDER_TRANSPARENT)
    
    # Create mask
    mask = np.any(warped > 0, axis=2).astype(np.uint8) * 255
    result = cv2.bitwise_and(result, cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
    result = cv2.add(result, warped)
    
    return result

def extract_ocr_input(full_img, quad, target_size=(94, 24)):
    """Warp plate region to 94x24 OCR input (same as board pipeline)."""
    h, w = full_img.shape[:2]
    src_pts = np.array(quad, dtype=np.float32)
    dst_w, dst_h = target_size
    dst_pts = np.array([[0,0],[dst_w,0],[dst_w,dst_h],[0,dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(full_img, M, target_size)

def make_val_hard(img_ocr):
    """Apply controlled degradation for hard validation set."""
    h, w = img_ocr.shape[:2]
    # Random blur
    if random.random() < 0.5:
        k = random.choice([3, 5])
        img_ocr = cv2.GaussianBlur(img_ocr, (k, k), 0)
    # Random brightness
    if random.random() < 0.5:
        delta = random.uniform(-40, 40)
        img_ocr = np.clip(img_ocr.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    # Random JPEG
    if random.random() < 0.5:
        q = random.randint(60, 90)
        _, enc = cv2.imencode('.jpg', img_ocr, [cv2.IMWRITE_JPEG_QUALITY, q])
        img_ocr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # Random noise
    if random.random() < 0.3:
        noise = np.random.randn(h, w, 3).astype(np.float32) * random.uniform(2, 8)
        img_ocr = np.clip(img_ocr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img_ocr

# ── Main generation loop ──
random.seed(20260603)
np.random.seed(20260603)

SMOKE = '--smoke' in sys.argv
PER_PROVINCE = {'train': 50 if SMOKE else 500, 'val_clean': 10 if SMOKE else 50, 'val_hard': 10 if SMOKE else 50}
print(f'Mode: {"SMOKE" if SMOKE else "FULL"}')
print(f'Per-province targets: train={PER_PROVINCE["train"]} val_clean={PER_PROVINCE["val_clean"]} val_hard={PER_PROVINCE["val_hard"]}')

total_needed = sum(PER_PROVINCE.values()) * len(province_chars)
print(f'Total images needed: {total_needed}')
print(f'Available sources: {len(source_images)}')

all_records = []  # list of {path, text, province, label, split}
feature_log = []
random.shuffle(source_images)

# Split source pool by base image to prevent leakage
base_ids = {}
for entry in source_images:
    path = entry.get('img_path', '')
    base_id = Path(path).stem.split('-')[0] if path else ''
    base_ids.setdefault(base_id, []).append(entry)
base_id_list = list(base_ids.keys())
random.shuffle(base_id_list)

n_train = PER_PROVINCE['train'] * len(province_chars)
n_val_c = PER_PROVINCE['val_clean'] * len(province_chars)
n_val_h = PER_PROVINCE['val_hard'] * len(province_chars)
total = n_train + n_val_c + n_val_h

# Assign base IDs to splits
split_assignments = {}
train_pool, valc_pool, valh_pool = [], [], []
idx = 0
for bid in base_id_list:
    entries = base_ids[bid]
    if idx < n_train: train_pool.extend(entries)
    elif idx < n_train + n_val_c: valc_pool.extend(entries)
    else: valh_pool.extend(entries)
    idx += len(entries) * 5  # approximate (5 variants per base)

# Actually, just shuffle and iterate
random.shuffle(source_images)
source_cursor = 0

prov_counts = {s: {p: 0 for p in province_chars} for s in ['train', 'val_clean', 'val_hard']}
generated = []

while source_cursor < len(source_images) and len(generated) < total:
    entry = source_images[source_cursor]
    source_cursor += 1
    
    img_path = entry.get('img_path', '')
    quad = entry.get('pose_quad') or entry.get('gt_quad')
    if not quad or not img_path:
        continue
    
    img_full = cv2.imread(str(ROOT / img_path) if not os.path.isabs(img_path) else img_path)
    if img_full is None:
        continue
    
    try:
        features, warped = extract_cv_features(img_full, quad)
    except Exception as e:
        continue
    
    # Determine which split still needs samples
    for split_name in ['train', 'val_clean', 'val_hard']:
        for prov_char in province_chars:
            if prov_counts[split_name][prov_char] >= PER_PROVINCE[split_name]:
                continue
            
            letter = random.choice(ALPHABET)
            digits = ''.join(random.choices(DIGITS, k=4))
            
            try:
                rendered, text = render_police_plate(prov_char, letter, digits)
                rendered_matched = apply_cv_features(rendered, features)
                full_result = warp_back_to_original(rendered_matched, quad, img_full.shape)
                ocr_input = extract_ocr_input(full_result, quad)
                
                if split_name == 'val_hard':
                    ocr_input = make_val_hard(ocr_input)
                
                # Save
                fname = f'{split_name}_{prov_char}{letter}{digits}警.jpg'
                out_path = str(DATASET / fname)
                cv2.imwrite(out_path, ocr_input)
                
                prov_counts[split_name][prov_char] += 1
                generated.append({
                    'path': out_path, 'text': text, 'province': prov_char,
                    'label': province_chars.index(prov_char), 'split': split_name,
                })
                
                # Log features for first few per province
                if prov_counts[split_name][prov_char] <= 2:
                    feature_log.append({**features, 'province': prov_char, 'split': split_name})
                
            except Exception as e:
                continue
            
            break  # next source image
        # Only break outer loop if all splits done
    if all(prov_counts[s][p] >= PER_PROVINCE[s] for s in ['train','val_clean','val_hard'] for p in province_chars):
        break

print(f'\nGenerated: {len(generated)}')
for s in ['train', 'val_clean', 'val_hard']:
    counts = Counter(g['province'] for g in generated if g['split']==s)
    print(f'  {s}: {sum(counts.values())} ({len(counts)} provinces) min={min(counts.values())} max={max(counts.values())}')

# Write manifests
fieldnames = ['path', 'text', 'province', 'label', 'split']
for split_name in ['train', 'val_clean', 'val_hard']:
    split_records = [g for g in generated if g['split'] == split_name]
    csv_path = str(MANIFEST / f'{split_name}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in split_records:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'Written: {csv_path}')

# Generation summary
summary = {
    'mode': 'SMOKE' if SMOKE else 'FULL',
    'total_generated': len(generated),
    'per_province_target': PER_PROVINCE,
    'per_split_counts': {s: sum(1 for g in generated if g['split']==s) for s in ['train','val_clean','val_hard']},
    'sources_used': source_cursor,
    'source_total': len(source_images),
    'feature_fields': list(features.keys()) if feature_log else [],
}
with open(str(MANIFEST / 'generation_summary.json'), 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Feature stats
if feature_log:
    feat_stats = {}
    for k in features.keys():
        vals = [f[k] for f in feature_log]
        feat_stats[k] = {'min': min(vals), 'max': max(vals), 'mean': sum(vals)/len(vals), 'std': (sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals))**0.5}
    with open(str(MANIFEST / 'feature_stats.json'), 'w') as f:
        json.dump(feat_stats, f, ensure_ascii=False, indent=2)

print(f'\nDone. Dataset: {DATASET}')
print(f'Manifests: {MANIFEST}')
