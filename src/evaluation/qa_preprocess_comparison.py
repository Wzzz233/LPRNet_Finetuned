#!/usr/bin/env python3
"""Generate QA contact sheets showing preproc method outputs on representative CCPD samples."""

import os, sys, csv
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, '/home/wzzz/LPRNet/src')
from load_data import parse_ccpd_quad_from_name, prepare_board_ocr_input_from_quad_bgr888

# ── Preproc methods (same as ablation) ──
def raw_bgr(bgr):
    return bgr.copy()
def ycrcb_y_clahe(bgr, clip_limit=2.0, tile=(8,8)):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(ycrcb); y_eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(y)
    return cv2.cvtColor(cv2.merge([y_eq,cr,cb]), cv2.COLOR_YCrCb2BGR)
def lab_l_clahe(bgr, clip_limit=2.0, tile=(8,8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab); l_eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq,a,b]), cv2.COLOR_LAB2BGR)
def y_adaptive_gamma(bgr, target=128):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(ycrcb)
    y_f = y.astype(np.float32)
    m = y_f.mean().clip(1e-6)
    g = np.clip(np.log(m/255.0)/np.log(target/255.0+1e-8), 0.3, 3.0)
    y_g = ((y_f/255.0)**(1.0/g)*255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([y_g,cr,cb]), cv2.COLOR_YCrCb2BGR)
def y_highlight_compress(bgr, th=200, slope=0.3):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(ycrcb)
    y_f = y.astype(np.float32); mask=y_f>th; y_f[mask]=th+(y_f[mask]-th)*slope
    return cv2.cvtColor(cv2.merge([np.clip(y_f,0,255).astype(np.uint8),cr,cb]), cv2.COLOR_YCrCb2BGR)
def y_clahe_hc(bgr, clip=1.5, tile=(8,8), th=200, slope=0.3):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(ycrcb)
    y_eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(y)
    y_f = y_eq.astype(np.float32); mask=y_f>th; y_f[mask]=th+(y_f[mask]-th)*slope
    return cv2.cvtColor(cv2.merge([np.clip(y_f,0,255).astype(np.uint8),cr,cb]), cv2.COLOR_YCrCb2BGR)

METHODS = [raw_bgr, ycrcb_y_clahe, lab_l_clahe, y_adaptive_gamma, y_highlight_compress, y_clahe_hc]
METHOD_NAMES = ['raw_bgr', 'ycrcb_y_clahe', 'lab_l_clahe', 'y_adaptive_gamma', 'y_highlight_compress', 'y_clahe+highlight']

# ── Pick representative samples from each subset ──
SUBSET_SEEDS = {
    'ccpd_db': [0, 50, 100, 150, 200, 250, 300, 350],
    'ccpd_challenge': [0, 50, 100, 150, 200, 250, 300, 350],
    'ccpd_tilt': [250, 300, 350, 400, 450, 500],
    'ccpd_base': [0, 50, 100, 150],
    'ccpd_weather': [0, 50, 100, 150],
}

import random

OUT_DIR = Path('eval_reports/front_preprocess_ablation/qa')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_qa_page(subset_dir, subset_alias, indices, output_path):
    jpgs = sorted(Path(subset_dir).glob('*.jpg'))
    selected = [jpgs[i] for i in indices if i < len(jpgs)]
    n = len(selected)
    n_methods = len(METHODS)
    
    # Each row: a sample, each column: a method
    # Display size for each thumbnail: warp output is 94x24, but we'll show at larger scale
    SCALE = 4
    cell_w = 94 * SCALE + 20  # padding
    cell_h = 24 * SCALE + 4
    label_h = 20  # height for method labels
    
    # Left margin for sample labels (subset + index)
    margin_left = 160
    
    canvas_w = margin_left + cell_w * n_methods
    canvas_h = label_h + cell_h * n + 20
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
    
    # Method labels on top
    for mi, mname in enumerate(METHOD_NAMES):
        x = margin_left + mi * cell_w + 10
        cv2.putText(canvas, mname, (x, label_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    
    for si, jpg in enumerate(selected):
        y_off = label_h + si * cell_h + 10
        
        # Sample label on the left
        label_text = f'{subset_alias}[{indices[si]}] {jpg.stem[:30]}'
        cv2.putText(canvas, label_text, (5, y_off + cell_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80,80,80), 1)
        
        img_bgr = cv2.imread(str(jpg))
        if img_bgr is None: continue
        quad = parse_ccpd_quad_from_name(jpg.name)
        if quad is None: continue
        
        # Get warped baseline (no preproc) once
        try:
            prepared, _, _, _, _ = prepare_board_ocr_input_from_quad_bgr888(
                img_bgr, quad, 94, 24, 'letterbox', 'nn', 'none', 'bgr', quad_pad_ratio=0.0)
        except:
            continue
        
        for mi, method_fn in enumerate(METHODS):
            try:
                processed = method_fn(prepared)
            except:
                processed = prepared.copy()
            
            # Scale up for visibility
            display = cv2.resize(processed, (94*SCALE, 24*SCALE), interpolation=cv2.INTER_NEAREST)
            x_off = margin_left + mi * cell_w + 10
            canvas[y_off:y_off+24*SCALE, x_off:x_off+94*SCALE] = display
    
    cv2.imwrite(str(output_path), canvas)
    print(f'Written: {output_path}')
    return len(selected)

for subset_name, indices in SUBSET_SEEDS.items():
    subset_dir = f'/home/wzzz/LPRNet/datasets/CCPD2019/{subset_name}'
    alias = subset_name.replace('ccpd_', '').upper()
    out_path = OUT_DIR / f'qa_{subset_name}.png'
    n = make_qa_page(subset_dir, alias, indices, out_path)

print(f'\nAll QA pages in {OUT_DIR.resolve()}')
