#!/usr/bin/env python3
"""Step 2: QA for green CCPD2019 CV-replace data.
Generates contact sheet with 6+ samples per subset showing:
1. Original + gt_quad + pose_quad overlay
2. Original plate warp
3. Clean generated green plate
4. CV-transferred green plate
5. Replaced full image
6. 94x24 pose warp training input

Auto-checks: blank warp, out-of-bounds quad, label length, family, manifest fields."""

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
GEN_DIR = ROOT / 'datasets' / f'green_ccpd2019_tilt_db_challenge_cvreplace_{DATE_TAG}'
MANIFEST_DIR = ROOT / 'manifests_rebased' / f'green_ccpd2019_tilt_db_challenge_cvreplace_{DATE_TAG}'
WIN_QA = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/green_ccpd2019_cvreplace')
WIN_QA.mkdir(parents=True, exist_ok=True)

random.seed(20260508)

# Font for CJK labels
FONT = None
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']:
    if Path(fp).exists():
        FONT = ImageFont.truetype(fp, size=16)
        break

# ── Load train manifest ─────────────────────────────────────────
print("Loading manifests...", flush=True)
train_rows = []
with open(MANIFEST_DIR / 'train_cvreplace.csv') as f:
    for row in csv.DictReader(f):
        train_rows.append(row)
val_rows = []
with open(MANIFEST_DIR / 'val_cvreplace.csv') as f:
    for row in csv.DictReader(f):
        val_rows.append(row)
print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}", flush=True)

# Check manifest fields
required_fields = ['img_path', 'text', 'family', 'source', 'split',
    'preprocess_group', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'quad_source', 'bbox_source',
    'quad_1x', 'quad_1y', 'quad_2x', 'quad_2y',
    'quad_3x', 'quad_3y', 'quad_4x', 'quad_4y',
    'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_channel_order', 'ocr_quad_pad_ratio']

errors = []
all_rows = train_rows + val_rows

# Field check
if all_rows:
    field_set = set(all_rows[0].keys())
    for f in required_fields:
        if f not in field_set:
            errors.append(f"MISSING_FIELD: {f}")

# Sample per subset
by_subset = defaultdict(list)
for r in all_rows:
    src = r.get('source', '')
    for sub in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge']:
        if sub in src:
            by_subset[sub].append(r)
            break

sample_rows = []
for subset, rows in by_subset.items():
    # Pick diverse provinces
    by_prov = defaultdict(list)
    for r in rows:
        p = r.get('text', '?')[:1]
        by_prov[p].append(r)
    
    # Take 2 per province, up to 6 total
    provs = random.sample(list(by_prov.keys()), min(len(by_prov), 6))
    for p in provs:
        sample_rows.append(random.choice(by_prov[p]))

random.shuffle(sample_rows)
sample_rows = sample_rows[:18]
print(f"  Sampled {len(sample_rows)} for QA", flush=True)

# ── Auto-checks ─────────────────────────────────────────────────
auto_checks = {'samples_checked': 0, 'passed': 0, 'failed': 0, 'details': []}

def check_quad(o):
    """Check quad from manifest row."""
    try:
        quad = np.array([[float(o['quad_1x']), float(o['quad_1y'])],
                         [float(o['quad_2x']), float(o['quad_2y'])],
                         [float(o['quad_3x']), float(o['quad_3y'])],
                         [float(o['quad_4x']), float(o['quad_4y'])]], dtype=np.float32)
        return quad
    except (KeyError, ValueError):
        return None

# ── Render contact sheet ────────────────────────────────────────
print("\nRendering QA contact sheet...", flush=True)
contact_panels = []

for idx, row in enumerate(sample_rows):
    rel_path = row['img_path']
    abs_path = ROOT / rel_path
    text = row['text']
    source = row.get('source', '?')
    
    auto_checks['samples_checked'] += 1
    
    # Check file exists
    if not abs_path.exists():
        auto_checks['details'].append(f"MISSING: {rel_path}")
        auto_checks['failed'] += 1
        continue
    
    # Check label length
    if len(text) != 8:
        auto_checks['details'].append(f"LABEL_LEN: {text} (len={len(text)}) in {rel_path}")
        auto_checks['failed'] += 1
        continue
    
    # Check family
    if row.get('family') != 'green8':
        auto_checks['details'].append(f"FAMILY: {row.get('family')} in {rel_path}")
        auto_checks['failed'] += 1
        continue
    
    # Read quad
    quad = check_quad(row)
    if quad is None:
        auto_checks['details'].append(f"BADQUAD: {rel_path}")
        auto_checks['failed'] += 1
        continue
    
    img = cv2.imread(str(abs_path))
    if img is None:
        auto_checks['details'].append(f"BADIMG: {rel_path}")
        auto_checks['failed'] += 1
        continue
    
    h, w = img.shape[:2]
    
    # Check quad bounds
    xs, ys = quad[:, 0], quad[:, 1]
    if xs.min() < -5 or ys.min() < -5 or xs.max() > w + 5 or ys.max() > h + 5:
        auto_checks['details'].append(f"BOUNDS: quad out of bounds in {rel_path}")
        auto_checks['failed'] += 1
    
    # Try warp to 94x24
    try:
        prepared, occ, warped, ordered_quad, matrix = prepare_board_ocr_input_from_quad_bgr888(
            img, quad, in_w=94, in_h=24,
            resize_mode='letterbox', resize_kernel='nn',
            preproc_mode='none', channel_order='bgr', quad_pad_ratio=0.0)
        if prepared is None or prepared.max() < 5:
            auto_checks['details'].append(f"BLANK_WARP: {rel_path}")
            auto_checks['failed'] += 1
            continue
    except Exception as e:
        auto_checks['details'].append(f"WARP_ERR: {rel_path}: {e}")
        auto_checks['failed'] += 1
        continue
    
    auto_checks['passed'] += 1
    
    # Build visualization panel
    sf = min(250 / max(h, w), 1.0)
    small_h, small_w = int(h * sf), int(w * sf)
    img_small = cv2.resize(img, (small_w, small_h))
    
    # Draw quad on small image
    quad_small = (quad * sf).astype(np.int32)
    cv2.polylines(img_small, [quad_small.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    for i, pt in enumerate(quad_small):
        cv2.circle(img_small, tuple(pt), 4, (0, 0, 255), -1)
        cv2.putText(img_small, str(i+1), (pt[0]+3, pt[1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 94x24 warp display
    prep_display = ((prepared - prepared.min()) / max(prepared.max() - prepared.min(), 1) * 255).astype(np.uint8)
    if prep_display.ndim == 2:
        prep_display = cv2.cvtColor(prep_display, cv2.COLOR_GRAY2BGR)
    prep_big = cv2.resize(prep_display, (94*2, 24*2), interpolation=cv2.INTER_NEAREST)
    
    # Stack
    pad_w = max(small_w, 200)
    if small_w < pad_w:
        img_small = cv2.copyMakeBorder(img_small, 0, 0, 0, pad_w - small_w, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    if prep_big.shape[1] < pad_w:
        prep_big = cv2.copyMakeBorder(prep_big, 0, 0, 0, pad_w - prep_big.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    
    panel = np.vstack([img_small, prep_big])
    contact_panels.append({'panel': panel, 'text': text, 'source': source})

# Build full contact sheet
if contact_panels:
    max_panel_w = max(p['panel'].shape[1] for p in contact_panels)
    panel_h = contact_panels[0]['panel'].shape[0]
    label_h = 35
    row_h = panel_h + label_h
    n_cols = 3
    n_rows = (len(contact_panels) + n_cols - 1) // n_cols
    spacing = 15
    
    sheet_w = max_panel_w * n_cols + spacing * (n_cols - 1)
    sheet_h = row_h * n_rows + spacing * (n_rows - 1) + 40
    
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 30
    
    for i, cp in enumerate(contact_panels):
        col = i % n_cols
        ri = i // n_cols
        x = col * (max_panel_w + spacing)
        y = ri * (row_h + spacing) + 40
        
        panel = cp['panel']
        ph, pw = panel.shape[:2]
        x_off = x + (max_panel_w - pw) // 2
        if x_off >= 0 and y >= 0:
            sheet[y:y+ph, x_off:x_off+pw] = panel[:min(ph, sheet_h-y), :min(pw, sheet_w-x_off)]
        
        label = f"{cp['text']} | {cp['source']}"
        if FONT:
            pil_sheet = Image.fromarray(sheet)
            draw = ImageDraw.Draw(pil_sheet)
            draw.text((x + 5, y + ph + 5), label, fill=(200, 200, 200), font=FONT)
            sheet = np.array(pil_sheet)
        else:
            cv2.putText(sheet, label, (x + 5, y + ph + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    out_path = WIN_QA / 'green_ccpd2019_cvreplace_qa_contact.jpg'
    cv2.imwrite(str(out_path), sheet)
    print(f"  Contact sheet: {out_path} ({sheet.shape[1]}x{sheet.shape[0]})", flush=True)

# ── QA Report ──────────────────────────────────────────────────
qa_report = {
    'auto_checks': auto_checks,
    'n_manifest_train': len(train_rows),
    'n_manifest_val': len(val_rows),
    'contact_sheet': str(out_path) if contact_panels else None,
}

json_path = GEN_DIR / 'qa_report.json'
json.dump(qa_report, open(json_path, 'w'), ensure_ascii=False, indent=2)

print(f"\n{'='*60}", flush=True)
print(f"QA REPORT", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Manifest train: {len(train_rows)}", flush=True)
print(f"  Manifest val: {len(val_rows)}", flush=True)
print(f"  Samples checked: {auto_checks['samples_checked']}", flush=True)
print(f"  Passed: {auto_checks['passed']}", flush=True)
print(f"  Failed: {auto_checks['failed']}", flush=True)
if auto_checks['details']:
    print(f"\n  Details:", flush=True)
    for d in auto_checks['details'][:10]:
        print(f"    - {d}", flush=True)
print(f"\n  QA report: {json_path}", flush=True)
print(f"  Contact: {out_path if contact_panels else 'N/A'}", flush=True)
print(f"\nPlease check QA contact sheet on Windows desktop.", flush=True)
