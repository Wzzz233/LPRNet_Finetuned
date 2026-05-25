#!/usr/bin/env python3
"""
Eval E12 on Pose-quad replacement data (no gray3).
Pipeline:
  1. Read replacement image (full CCPD frame with synthetic plate text)
  2. Warp using Pose quad → plate crop
  3. Letterbox resize to 94×24 (NN kernel)
  4. Normalize: (x-127.5)*0.0078125
  5. Feed to E12 green8 head
  6. Compare prediction to ground truth
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'src' / 'training'), str(_SRC_DIR / 'src' / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from src.load_data import CHARS, order_quad_points
from src.training.train_LPRNet import forward_family_logits
from src.evaluation.test_LPRNet import greedy_decode_logits

# ── Paths ──────────────────────────────────────────────────────────
REPLACE_DIR = Path('/home/wzzz/LPRNet/datasets/ccpd2020_replace_pose_v3')
SOURCE_LOG = REPLACE_DIR / 'source_log.jsonl'
IMG_DIR = REPLACE_DIR / 'images' / 'train'
E12_WEIGHTS = '/home/wzzz/LPRNet/experiments/green_e12_e9c_append_boarddump_anticollapse_5prov_1200_stage2/Final_LPRNet_model.pth'

# ── Load model ─────────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'[info] device={device}')
state = torch.load(E12_WEIGHTS, map_location=device)
net, _cfg = build_lprnet_multihead_from_state_dict(
    state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
load_multihead_state_dict_compat(net, state, strict=False)
net.to(device)
net.eval()
print(f'[info] loaded E12 weights')

# ── Load source log ────────────────────────────────────────────────
records = []
with open(SOURCE_LOG) as f:
    for line in f:
        records.append(json.loads(line))
print(f'[info] source_log: {len(records)} records')

# ── Board-side processing (same as prepare_board_ocr_input_bgr888) ─
def warp_and_prepare(img, quad, target_w=94, target_h=24):
    """Pose quad warp → letterbox 94×24 → no gray3."""
    # Order quad points
    ordered = order_quad_points(quad)
    
    # Compute target dimensions
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])
    dst_w = max(1, int(round(max(w_top, w_bot))))
    dst_h = max(1, int(round(max(h_left, h_right))))
    
    # Perspective warp
    src_pts = ordered.astype(np.float32)
    dst_pts = np.float32([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (dst_w, dst_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    
    # Letterbox resize to 94×24 (NN kernel)
    h, w = warped.shape[:2]
    scale = min(target_w / w, target_h / h)
    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))
    resized = cv2.resize(warped, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    
    # Letterbox canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - scaled_w) // 2
    y_off = (target_h - scaled_h) // 2
    canvas[y_off:y_off+scaled_h, x_off:x_off+scaled_w] = resized
    
    return canvas

def normalize(img):
    """(x - 127.5) * 0.0078125 → CHW"""
    x = img.astype(np.float32)
    x -= 127.5
    x *= 0.0078125
    return torch.from_numpy(x).permute(2, 0, 1)

def decode(logits):
    prebs = logits.detach().cpu().numpy()
    ids = greedy_decode_logits(prebs)
    return ''.join(CHARS[int(c)] for c in ids[0])

# ── Evaluation ─────────────────────────────────────────────────────
results = []
exact_ok = first_ok = 0
total = 0
prov_count = {}

# Limit to reasonable batch for quick demo (e.g., first 500)
sample_count = min(500, len(records))
print(f'\n[info] evaluating {sample_count} samples...\n')

for i in range(sample_count):
    rec = records[i]
    gen_path = rec['generated_img']
    gt_text = rec['new_text']
    quad = rec['pose_quad']  # [4, 2] float
    province = rec['province']
    
    gen_path = Path(gen_path)
    if not gen_path.exists():
        continue
    
    img = cv2.imread(str(gen_path))
    if img is None:
        continue
    
    # Warp + prep
    prepared = warp_and_prepare(img, quad)
    
    # Normalize + forward
    x = normalize(prepared).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_dict = forward_family_logits(net, x, return_all=True)
    logits = logits_dict['green8']
    pred = decode(logits)
    
    exact = (pred == gt_text)
    first = (len(pred) > 0 and len(gt_text) > 0 and pred[0] == gt_text[0])
    
    if exact:
        exact_ok += 1
    if first:
        first_ok += 1
    total += 1
    prov_count[province] = prov_count.get(province, 0) + 1
    
    results.append({
        'province': province,
        'gt': gt_text,
        'pred': pred,
        'exact': exact,
        'first_char': first,
    })
    
    if (i+1) % 100 == 0:
        pass  # progress

# ── Report ─────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'E12 on Pose-quad replacement data (no gray3, {total} samples)')
print(f'Exact:  {exact_ok}/{total} = {exact_ok/total*100:.2f}%')
print(f'First:  {first_ok}/{total} = {first_ok/total*100:.2f}%')
print(f'{"="*60}')

# Per-province
print(f'\n{"Province":>8} {"count":>6} {"exact":>8} {"first":>8}')
print('-' * 32)
for p in sorted(prov_count.keys()):
    p_total = prov_count[p]
    p_exact = sum(1 for r in results if r['province'] == p and r['exact'])
    p_first = sum(1 for r in results if r['province'] == p and r['first_char'])
    print(f'{p:>8} {p_total:>6} {p_exact/p_total*100:>7.1f}% {p_first/p_total*100:>7.1f}%')

# Top errors
from collections import Counter
error_patterns = Counter()
for r in results:
    if not r['exact'] and r['gt']:
        error_patterns[r['gt'] + ' → ' + r['pred']] += 1
print(f'\nTop 10 error patterns:')
for pattern, cnt in error_patterns.most_common(10):
    print(f'  {pattern} ({cnt})')
PYEOF
