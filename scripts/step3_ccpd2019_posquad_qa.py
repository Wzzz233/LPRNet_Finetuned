#!/usr/bin/env python3
"""Step 3: QA visualization for CCPD2019 pose quad manifest.
Checks: blank warp, out-of-bounds quad, bad labels. Generates contact sheet."""

import csv, json, os, sys, random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import prepare_board_ocr_input_from_quad_bgr888

DATE_TAG = '20260508'
MANIFEST_DIR = ROOT / 'manifests_rebased' / f'blue_ccpd2019_tilt_db_challenge_posquad_{DATE_TAG}'
POSE_DIR = ROOT / 'datasets' / f'ccpd2019_tilt_db_challenge_posquads_{DATE_TAG}'
OUT_DIR = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/ccpd2019_posquad_qa')
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(20260508)

# ── Font for CJK labels ────────────────────────────────────────────
FONT_PATHS = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
]
FONT = None
for fp in FONT_PATHS:
    if Path(fp).exists():
        FONT = ImageFont.truetype(fp, size=20)
        break


# ── Load manifest ──────────────────────────────────────────────────
print("Loading manifest...", flush=True)
rows = []
with open(MANIFEST_DIR / 'train_posquad.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
print(f"  {len(rows)} rows", flush=True)

# Stratify by source
by_subset = {}
for r in rows:
    s = r.get('source', 'unknown')
    # extract subset from source (e.g. "ccpd2019_ccpd_tilt")
    subset = s.replace('ccpd2019_', '')
    by_subset.setdefault(subset, []).append(r)

for k, v in by_subset.items():
    print(f"  {k}: {len(v)}", flush=True)

# Sample 6 per subset
sample = []
for subset, subset_rows in by_subset.items():
    sample.extend(random.sample(subset_rows, min(6, len(subset_rows))))

print(f"\nSampled {len(sample)} images for QA", flush=True)

# ── QA checks ──────────────────────────────────────────────────────
errors = []

def check_quad_overlay(quad, w, h):
    xs = np.array([p[0] for p in quad])
    ys = np.array([p[1] for p in quad])
    tol = -5
    if xs.min() < tol or ys.min() < tol or xs.max() > w + 5 or ys.max() > h + 5:
        return f"out_of_bounds: x[{xs.min():.0f},{xs.max():.0f}] y[{ys.min():.0f},{ys.max():.0f}] img={w}x{h}"
    return None

def quad_to_np(quad_str_x, quad_str_y):
    # This function is not needed — we read from the CSV directly as quad_1x..quad_4y
    pass

def draw_quad_on_img(img, quad, color=(0, 255, 0), thickness=2):
    """Draw quad on BGR image. Quad is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"""
    pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    # Draw circles at corners
    for i, (x, y) in enumerate(quad):
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i+1), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# ── Render contact sheet ──────────────────────────────────────────
print("\nRendering QA contact sheet...", flush=True)

contact_rows = []
for idx, row in enumerate(sample):
    img_path = row['img_path']
    abs_path = ROOT / img_path
    text = row['text']
    source = row.get('source', '?')

    if not abs_path.exists():
        errors.append(f"MISSING: {abs_path}")
        continue

    img = cv2.imread(str(abs_path))
    if img is None:
        errors.append(f"BADIMG: {abs_path}")
        continue

    h, w = img.shape[:2]

    # Read quad from manifest
    try:
        quad = np.array([
            [float(row['quad_1x']), float(row['quad_1y'])],
            [float(row['quad_2x']), float(row['quad_2y'])],
            [float(row['quad_3x']), float(row['quad_3y'])],
            [float(row['quad_4x']), float(row['quad_4y'])],
        ], dtype=np.float32)
    except (KeyError, ValueError) as e:
        errors.append(f"BADQUAD: {abs_path} — {e}")
        continue

    # Check bounds
    bounds_err = check_quad_overlay(quad, w, h)
    if bounds_err:
        errors.append(f"{bounds_err} — {img_path}")

    # Draw quad on image
    img_drawn = img.copy()
    draw_quad_on_img(img_drawn, quad)

    # Warp to 94x24 OCR input
    try:
        prepared, occ, warped, ordered_quad, matrix = prepare_board_ocr_input_from_quad_bgr888(
            img, quad,
            in_w=94, in_h=24,
            resize_mode='letterbox', resize_kernel='nn',
            preproc_mode='none', channel_order='bgr',
            quad_pad_ratio=0.0,
        )
        if prepared is None:
            errors.append(f"WARPFAIL: {abs_path}")
            continue
    except Exception as e:
        errors.append(f"WARP_EXCEPTION: {abs_path} — {e}")
        continue

    # Check blank warp
    if prepared.max() < 10:
        errors.append(f"BLANK: {abs_path} — warp appears blank (max={prepared.max():.0f})")

    # Scale prepared for display (range is [-1, 1]?)
    prep_display = ((prepared - prepared.min()) / max(prepared.max() - prepared.min(), 1) * 255).astype(np.uint8)
    if prep_display.ndim == 2:
        prep_display = cv2.cvtColor(prep_display, cv2.COLOR_GRAY2BGR)

    # Resize original for contact sheet (keep aspect, max 300px height)
    scale = min(300 / h, 1.0)
    small_h, small_w = int(h * scale), int(w * scale)
    img_small = cv2.resize(img_drawn, (small_w, small_h))

    # Pad to fixed width for alignment
    pad_w = 400
    if small_w < pad_w:
        img_small = cv2.copyMakeBorder(img_small, 0, 0, 0, pad_w - small_w, cv2.BORDER_CONSTANT, value=(64, 64, 64))

    # Also pad the 94x24 warp to same width
    prep_display_padded = cv2.copyMakeBorder(prep_display, 0, 0, 0, pad_w - prep_display.shape[1], cv2.BORDER_CONSTANT, value=(64, 64, 64))

    # Stack: original+quad overlay | 94x24 warp
    contact_panel = np.vstack([img_small, prep_display_padded])

    contact_rows.append({
        'panel': contact_panel,
        'text': text,
        'source': source,
        'filename': Path(img_path).name,
    })

    if (idx + 1) % 6 == 0:
        print(f"  Processed {idx+1}/{len(sample)}", flush=True)

# ── Build full contact sheet ──────────────────────────────────────
if contact_rows:
    # Determine panel dimensions
    max_panel_w = max(r['panel'].shape[1] for r in contact_rows)
    panel_h = contact_rows[0]['panel'].shape[0]

    # Create contact sheet with labels
    spacing = 30
    col_width = max_panel_w + 10
    n_cols = 3
    n_rows = (len(contact_rows) + n_cols - 1) // n_cols
    sheet_w = col_width * n_cols + spacing * (n_cols - 1)
    # Each panel row needs: panel_h + label_height
    label_h = 40
    row_h = panel_h + label_h
    sheet_h = row_h * n_rows + spacing * (n_rows - 1)

    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 30

    for i, cr in enumerate(contact_rows):
        col = i % n_cols
        row_idx = i // n_cols
        x = col * (col_width + spacing)
        y = row_idx * (row_h + spacing)

        panel = cr['panel']
        panel_h_actual, panel_w_actual = panel.shape[:2]

        # Center panel horizontally in column
        x_off = x + (col_width - panel_w_actual) // 2
        sheet[y:y+panel_h_actual, x_off:x_off+panel_w_actual] = panel

        # Label
        label = f"{cr['text']} | {cr['source']}"
        if FONT:
            # Use PIL for CJK text
            pil_sheet = Image.fromarray(sheet)
            draw = ImageDraw.Draw(pil_sheet)
            draw.text((x + 5, y + panel_h_actual + 5), label, fill=(200, 200, 200), font=FONT)
            sheet = np.array(pil_sheet)
        else:
            cv2.putText(sheet, label, (x + 5, y + panel_h_actual + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Save
    out_path = OUT_DIR / 'ccpd2019_posquad_qa_contact.jpg'
    cv2.imwrite(str(out_path), sheet)
    print(f"\nSaved contact sheet: {out_path} ({sheet.shape[1]}x{sheet.shape[0]})", flush=True)
else:
    out_path = None
    print("\nNo contact rows generated — all samples had errors", flush=True)

# ── Report ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}", flush=True)
print(f"QA REPORT", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"  Samples processed: {len(contact_rows)}", flush=True)
print(f"  Errors: {len(errors)}", flush=True)
if errors:
    print(f"\n  Error details:", flush=True)
    for e in errors[:20]:
        print(f"    - {e}", flush=True)
    if len(errors) > 20:
        print(f"    ... and {len(errors)-20} more", flush=True)
else:
    print(f"  All checks passed!", flush=True)

# Write error report
error_path = OUT_DIR / 'ccpd2019_posquad_qa_errors.json'
json.dump({'errors': errors, 'total_checked': len(contact_rows)}, open(error_path, 'w'), ensure_ascii=False)
print(f"\nError report saved: {error_path}", flush=True)
print(f"\nContact sheet: {out_path}", flush=True)
print(f"Please check the QA contact sheet on Windows desktop.", flush=True)
