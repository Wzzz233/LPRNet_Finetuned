#!/usr/bin/env python3
"""Step 2 v2: QA for green CCPD2019 CV-replace v2 (L-only transfer).
Reports green_ratio, blue_ratio, L stats, sharpness, noise per sample.
Generates contact sheet with before/after comparisons."""

import csv, json, random, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

DATE_TAG = '20260508'
GEN_DIR = ROOT / 'datasets' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v2_{DATE_TAG}'
MANIFEST_DIR = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_v2_{DATE_TAG}'
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/green_ccpd2019_cvreplace_v2')
WIN_QA.mkdir(parents=True, exist_ok=True)
random.seed(20260508)

FONT = None
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']:
    if Path(fp).exists():
        FONT = ImageFont.truetype(fp, size=14)
        break

# ── CV measurement functions ────────────────────────────────────
def lab_stats(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return {'L_mean': float(lab[:,:,0].mean()), 'L_std': float(lab[:,:,0].std()),
            'A_mean': float(lab[:,:,1].mean()), 'B_mean': float(lab[:,:,2].mean())}

def compute_sharpness(bgr):
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

def compute_noise(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = cv2.boxFilter(gray, -1, (7,7), normalize=False)
    local_mean = cv2.boxFilter(gray, -1, (7,7), normalize=True)
    local_var = local_std - local_mean * local_mean * 49
    local_var = np.clip(local_var, 0, None)
    return float(np.median(np.sqrt(local_var)))

def color_ratios(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0].astype(np.float32), hsv[:,:,1].astype(np.float32), hsv[:,:,2].astype(np.float32)
    green_mask = (s > 30) & (v > 40) & (h >= 45) & (h <= 100)
    blue_mask = (s > 30) & (v > 40) & (h >= 100) & (h <= 140)
    total = bgr.shape[0] * bgr.shape[1]
    return int(green_mask.sum())/max(total,1), int(blue_mask.sum())/max(total,1)

# ── Load manifests ──────────────────────────────────────────────
print("Loading manifests...", flush=True)
train_rows = list(csv.DictReader(open(MANIFEST_DIR / 'train_cvreplace_v2.csv')))
val_rows = list(csv.DictReader(open(MANIFEST_DIR / 'val_cvreplace_v2.csv')))
all_rows = train_rows + val_rows
print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}", flush=True)

# Required field check
required = ['img_path','text','family','source','split','preprocess_group',
            'has_quad','can_parse_ccpd_geom','can_perspective','quad_source','bbox_source',
            'quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y',
            'ocr_crop_mode','ocr_resize_mode','ocr_resize_kernel','ocr_preproc',
            'ocr_channel_order','ocr_quad_pad_ratio']
field_set = set(all_rows[0].keys()) if all_rows else set()
errors = []
for f in required:
    if f not in field_set:
        errors.append(f"MISSING_FIELD: {f}")

# Sample per subset
by_subset = defaultdict(list)
for r in all_rows:
    src = r.get('source','')
    for sub in ['ccpd_tilt','ccpd_db','ccpd_challenge']:
        if sub in src:
            by_subset[sub].append(r); break

sample_rows = []
for subset, rows in by_subset.items():
    by_prov = defaultdict(list)
    for r in rows:
        by_prov[r.get('text','?')[:1]].append(r)
    provs = random.sample(list(by_prov.keys()), min(len(by_prov), 6))
    for p in provs:
        sample_rows.append(random.choice(by_prov[p]))
random.shuffle(sample_rows)
sample_rows = sample_rows[:18]
print(f"  Sampled {len(sample_rows)} for QA", flush=True)

# ── QA loop ─────────────────────────────────────────────────────
qa_results = {'samples': [], 'auto_checks': {'samples_checked': 0, 'passed': 0, 'failed': 0, 'details': []}}
contact_panels = []

for idx, row in enumerate(sample_rows):
    rel_path = row['img_path']
    abs_path = ROOT / rel_path
    text = row['text']
    source = row.get('source','?')
    
    qa_results['auto_checks']['samples_checked'] += 1
    
    if not abs_path.exists():
        qa_results['auto_checks']['details'].append(f"MISSING: {rel_path}"); qa_results['auto_checks']['failed'] += 1
        continue
    if len(text) != 8:
        qa_results['auto_checks']['details'].append(f"LABEL_LEN: {text} ({len(text)})"); qa_results['auto_checks']['failed'] += 1
        continue
    if row.get('family') != 'green8':
        qa_results['auto_checks']['details'].append(f"FAMILY: {row.get('family')}"); qa_results['auto_checks']['failed'] += 1
        continue
    
    # Read quad
    try:
        quad = np.array([[float(row['quad_1x']),float(row['quad_1y'])],
                         [float(row['quad_2x']),float(row['quad_2y'])],
                         [float(row['quad_3x']),float(row['quad_3y'])],
                         [float(row['quad_4x']),float(row['quad_4y'])]], dtype=np.float32)
    except:
        qa_results['auto_checks']['details'].append(f"BADQUAD: {rel_path}"); qa_results['auto_checks']['failed'] += 1
        continue
    
    img = cv2.imread(str(abs_path))
    if img is None:
        qa_results['auto_checks']['details'].append(f"BADIMG: {rel_path}"); qa_results['auto_checks']['failed'] += 1
        continue
    
    h, w = img.shape[:2]
    xs, ys = quad[:,0], quad[:,1]
    if xs.min() < -5 or ys.min() < -5 or xs.max() > w+5 or ys.max() > h+5:
        qa_results['auto_checks']['details'].append(f"BOUNDS: {rel_path}"); qa_results['auto_checks']['failed'] += 1
    
    # Warp
    try:
        prepared, occ, warped, ordered_quad, matrix = prepare_board_ocr_input_from_quad_bgr888(
            img, quad, in_w=94, in_h=24,
            resize_mode='letterbox', resize_kernel='nn',
            preproc_mode='none', channel_order='bgr', quad_pad_ratio=0.0)
        if prepared is None or prepared.max() < 5:
            qa_results['auto_checks']['details'].append(f"BLANK_WARP: {rel_path}"); qa_results['auto_checks']['failed'] += 1
            continue
    except Exception as e:
        qa_results['auto_checks']['details'].append(f"WARP_ERR: {rel_path}: {e}"); qa_results['auto_checks']['failed'] += 1
        continue
    
    qa_results['auto_checks']['passed'] += 1
    
    # Color analysis on the plate region (warped)
    gr, br = color_ratios(warped)
    ls = lab_stats(warped)
    sharp = compute_sharpness(warped)
    ns = compute_noise(warped)
    
    sample_result = {
        'text': text, 'source': source,
        'green_ratio': round(gr, 3), 'blue_ratio': round(br, 3),
        'color_pass': br < 0.10,
        'L_mean': round(ls['L_mean'], 1), 'L_std': round(ls['L_std'], 1),
        'sharpness': round(sharp, 1),
        'noise': round(ns, 2),
    }
    qa_results['samples'].append(sample_result)
    
    # Contact panel
    sf = min(250 / max(h, w), 1.0)
    small_h, small_w = int(h*sf), int(w*sf)
    img_small = cv2.resize(img, (small_w, small_h))
    quad_small = (quad * sf).astype(np.int32)
    cv2.polylines(img_small, [quad_small.reshape(-1,1,2)], True, (0,255,0), 2)
    for i, pt in enumerate(quad_small):
        cv2.circle(img_small, tuple(pt), 3, (0,0,255), -1)
    
    prep_display = ((prepared-prepared.min())/max(prepared.max()-prepared.min(),1)*255).astype(np.uint8)
    if prep_display.ndim == 2:
        prep_display = cv2.cvtColor(prep_display, cv2.COLOR_GRAY2BGR)
    prep_big = cv2.resize(prep_display, (94*2, 24*2), interpolation=cv2.INTER_NEAREST)
    
    pad_w = max(small_w, 200)
    if small_w < pad_w:
        img_small = cv2.copyMakeBorder(img_small, 0,0,0,pad_w-small_w,cv2.BORDER_CONSTANT,value=(40,40,40))
    if prep_big.shape[1] < pad_w:
        prep_big = cv2.copyMakeBorder(prep_big, 0,0,0,pad_w-prep_big.shape[1],cv2.BORDER_CONSTANT,value=(40,40,40))
    
    panel = np.vstack([img_small, prep_big])
    contact_panels.append({'panel': panel, 'text': text, 'source': source,
                           'gr': gr, 'br': br})

# ── Contact sheet ──────────────────────────────────────────────
if contact_panels:
    max_pw = max(p['panel'].shape[1] for p in contact_panels)
    ph = contact_panels[0]['panel'].shape[0]
    lh = 40; rh = ph + lh
    nc = 3; nr = (len(contact_panels) + nc - 1)//nc
    sw = max_pw * nc + 15 * (nc-1)
    sh = rh * nr + 15 * (nr-1) + 40
    sheet = np.ones((sh, sw, 3), dtype=np.uint8) * 30
    
    for i, cp in enumerate(contact_panels):
        col, ri = i % nc, i // nc
        x = col * (max_pw + 15); y = ri * rh + 40
        panel = cp['panel']; pph, ppw = panel.shape[:2]
        x_off = x + (max_pw - ppw)//2
        if x_off >= 0 and y >= 0:
            sheet[y:y+pph, x_off:x_off+ppw] = panel[:min(pph, sh-y), :min(ppw, sw-x_off)]
        label = f"{cp['text']} G={cp['gr']:.2f} B={cp['br']:.2f} {'PASS' if cp['br']<0.10 else 'FAIL'}"
        if FONT:
            pil = Image.fromarray(sheet); d = ImageDraw.Draw(pil)
            d.text((x+5, y+pph+5), label, fill=(200,200,200), font=FONT); sheet = np.array(pil)
        else:
            cv2.putText(sheet, label, (x+5, y+pph+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    
    out_path = WIN_QA / 'green_ccpd2019_cvreplace_v2_qa_contact.jpg'
    cv2.imwrite(str(out_path), sheet)
    print(f"  Contact: {out_path} ({sheet.shape[1]}x{sheet.shape[0]})", flush=True)
else:
    out_path = None

# ── Aggregate stats ────────────────────────────────────────────
samples = qa_results['samples']
if samples:
    avg_gr = np.mean([s['green_ratio'] for s in samples])
    avg_br = np.mean([s['blue_ratio'] for s in samples])
    color_fail = sum(1 for s in samples if not s['color_pass'])
    print(f"\n  Avg green_ratio: {avg_gr:.3f}, Avg blue_ratio: {avg_br:.3f}", flush=True)
    print(f"  Color failures (blue_ratio>=0.10): {color_fail}/{len(samples)}", flush=True)
else:
    avg_gr = avg_br = 0; color_fail = 0

qa_report = {
    'auto_checks': qa_results['auto_checks'],
    'n_manifest_train': len(train_rows), 'n_manifest_val': len(val_rows),
    'color_analysis': {'samples_checked': len(samples),
                       'avg_green_ratio': round(float(avg_gr), 3),
                       'avg_blue_ratio': round(float(avg_br), 3),
                       'color_failures': color_fail},
    'per_sample': qa_results['samples'],
    'contact_sheet': str(out_path) if out_path else None,
}
json.dump(qa_report, open(GEN_DIR / 'qa_report_v2.json', 'w'), ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"QA REPORT v2")
print(f"{'='*60}")
print(f"  Manifest train: {len(train_rows)}, val: {len(val_rows)}")
print(f"  Checked: {qa_results['auto_checks']['samples_checked']}, "
      f"Passed: {qa_results['auto_checks']['passed']}, "
      f"Failed: {qa_results['auto_checks']['failed']}")
print(f"  Avg green_ratio: {avg_gr:.3f}, Avg blue_ratio: {avg_br:.3f}")
print(f"  Color failures: {color_fail}/{len(samples) if samples else 0}")
print(f"  QA report: {GEN_DIR / 'qa_report_v2.json'}")
print(f"  Contact: {out_path or 'N/A'}")
print(f"Please check QA contact sheet on Windows desktop.", flush=True)
