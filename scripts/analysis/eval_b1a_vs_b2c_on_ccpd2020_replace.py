#!/usr/bin/env python3
"""Evaluate B1A vs B2-C on CCPD2020 replace extreme val set (true perspective)."""
import csv, json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    sys.path.insert(0, str(p))

from load_data import CHARS, parse_ccpd_quad_from_name, \
    prepare_board_ocr_input_from_quad_bgr888
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from train_LPRNet import _select_family_logits_from_dict
from eval_lpr_detailed import decode_logits
import cv2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BLANK = len(CHARS) - 1

MODELS = {
    'B1A_iter2000': ROOT / 'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservativeLPRNet__iteration_2000.pth',
    'B2C_Final': ROOT / 'experiments/curriculum_gray3_stageB_B2C_paradigm3_softfreeze/Final_LPRNet_model.pth',
}
VAL_MANIFEST = ROOT / 'manifests/ccpd2020_replace_extreme_v1/val_B2C_ccpd2020_replace_extreme.csv'
OUT_DIR = ROOT / 'reports/b2c_ccpd2020_replace_extreme_eval_20260430'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def greedy_decode(logits_ct):
    labels = []; prev = None
    for t in range(logits_ct.shape[1]):
        c = int(np.argmax(logits_ct[:, t]))
        if c != BLANK and c != prev:
            labels.append(c)
        prev = c
    return ''.join(CHARS[c] for c in labels)

# Load val samples
samples = []
with open(VAL_MANIFEST, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        quad = parse_ccpd_quad_from_name(row['img_path'])
        if quad is None:
            print(f"  SKIP (no quad): {row['img_path']}")
            continue
        samples.append({'path': row['img_path'], 'gt': row['text'], 'quad': quad})
print(f"Loaded {len(samples)} val samples")

for model_name, model_path in MODELS.items():
    print(f"\n─── {model_name} ───")
    state = torch.load(model_path, map_location=DEVICE)
    net, cfg = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(DEVICE); net.eval()
    
    results = []
    with torch.no_grad():
        for s in samples:
            img = cv2.imread(s['path'])
            if img is None:
                continue
            # Use exact training pipeline: warp → letterbox 94×24 nn → gray3
            prepared, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
                img, s['quad'], 94, 24,
                resize_mode='letterbox', resize_kernel='nn',
                preproc_mode='gray3', channel_order='bgr',
                quad_pad_ratio=0.0,
            )
            # Normalize
            x = prepared.astype('float32')
            x -= 127.5; x *= 0.0078125
            x = np.transpose(x, (2, 0, 1))
            images = torch.from_numpy(x[None, ...]).to(DEVICE)
            
            raw = net(images)
            logits = _select_family_logits_from_dict(raw, sample_families=['green8']).detach().cpu().numpy()[0]
            
            # Beam decode (family-aware, same as eval_stageB)
            fam = ['green8']
            beam_ids = decode_logits(logits[None, ...], 'family_aware_beam', 20, 12, sample_families=fam)[0]
            beam = ''.join(CHARS[int(c)] for c in beam_ids)
            greedy = greedy_decode(logits)
            
            results.append({
                'gt': s['gt'],
                'pred': beam,
                'greedy': greedy,
                'exact': int(beam == s['gt']),
                'first': int(bool(beam) and beam[0] == s['gt'][0]),
                'greedy_exact': int(greedy == s['gt']),
                'greedy_first': int(bool(greedy) and greedy[0] == s['gt'][0]),
            })
    
    n = len(results)
    exact = sum(r['exact'] for r in results)
    first = sum(r['first'] for r in results)
    g_exact = sum(r['greedy_exact'] for r in results)
    g_first = sum(r['greedy_first'] for r in results)
    edit = np.mean([sum(1 for a,b in zip(r['gt'], r['pred']) if a!=b) + abs(len(r['gt'])-len(r['pred'])) for r in results])
    
    print(f"  n={n}")
    print(f"  beam_exact:   {exact}/{n} = {exact/n*100:.2f}%")
    print(f"  beam_first:   {first}/{n} = {first/n*100:.2f}%")
    print(f"  greedy_exact: {g_exact}/{n} = {g_exact/n*100:.2f}%")
    print(f"  greedy_first: {g_first}/{n} = {g_first/n*100:.2f}%")
    print(f"  mean_edit:    {edit:.4f}")
    
    # Save per-model
    with open(OUT_DIR / f'{model_name}.json', 'w') as f:
        json.dump({'model': model_name, 'n': n, 'beam_exact': exact/n,
                   'beam_first': first/n, 'greedy_exact': g_exact/n,
                   'greedy_first': g_first/n, 'mean_edit': float(edit),
                   'results': results}, f, ensure_ascii=False, indent=2)
