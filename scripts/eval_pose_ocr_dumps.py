#!/usr/bin/env python3
"""Evaluate B2-C vs B2-D on board pose OCR dumps (ocrin + gray3)."""

import csv, json, sys, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    sys.path.insert(0, str(p))

from load_data import CHARS
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from train_LPRNet import _select_family_logits_from_dict

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BLANK = len(CHARS) - 1
IMG_SIZE = (94, 24)  # W, H

# ── Models ─────────────────────────────────────────────────────────
MODELS = {
    'B2C': ROOT / 'experiments/curriculum_gray3_stageB_B2C_paradigm3_softfreeze/best_LPRNet_model.pth',
    'B2D': ROOT / 'experiments/curriculum_gray3_stageB_B2D_paradigm3_progress/best_LPRNet_model.pth',
}

# ── Dumps ──────────────────────────────────────────────────────────
DUMPS = {
    'dump1': Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump'),
    'dump2': Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2'),
}

# ── Helpers ────────────────────────────────────────────────────────
def gray3_preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Apply gray3: convert to gray, stack to 3-channel."""
    gray = img_rgb.mean(axis=2, keepdims=True).astype(np.uint8)
    return np.concatenate([gray, gray, gray], axis=2)

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize same as training: center 127.5, scale 1/128."""
    x = x.astype(np.float32)
    x -= 127.5
    x *= 0.0078125
    x = np.transpose(x, (2, 0, 1))  # HWC → CHW
    return x[None, ...]  # add batch dim

def greedy_decode(logits, blank=BLANK):
    """Simple CTC greedy decode."""
    labels = []
    prev = blank
    for t in range(logits.shape[1]):
        c = int(np.argmax(logits[:, t]))
        if c != blank and c != prev:
            labels.append(c)
        prev = c
    return ''.join(CHARS[c] for c in labels)

# ── Main ───────────────────────────────────────────────────────────
all_results = {}

for dump_name, dump_dir in DUMPS.items():
    print(f"\n{'=' * 70}")
    print(f"  {dump_name}: {dump_dir}")
    print('=' * 70)
    
    # Read index.csv
    index_path = dump_dir / 'index.csv'
    if not index_path.exists():
        print(f"  SKIP: no index.csv")
        continue
    
    samples = []
    with open(index_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ocrin_rel = row['ocr_input_path'].split('/')[-1]
            ocrin_path = dump_dir / ocrin_rel
            if ocrin_path.exists():
                samples.append({
                    'id': row['sample_id'],
                    'gt': row['app_text'],
                    'ocrin_path': str(ocrin_path),
                })
    
    print(f"  Samples: {len(samples)}")
    
    # Load models and run inference
    for model_name, model_path in MODELS.items():
        print(f"\n  ─── {model_name} ───")
        
        if not model_path.exists():
            print(f"  SKIP: checkpoint not found at {model_path}")
            continue
        
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)
        net, cfg = build_lprnet_multihead_from_state_dict(
            state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
        )
        load_multihead_state_dict_compat(net, state, strict=False)
        net.to(DEVICE)
        net.eval()
        
        correct = 0
        total = 0
        details = []
        
        with torch.no_grad():
            for s in samples:
                # Load ocrin (already 94x24 RGB)
                img_pil = Image.open(s['ocrin_path']).convert('RGB')
                img_np = np.array(img_pil, dtype=np.uint8)
                
                # gray3
                img_gray3 = gray3_preprocess(img_np)
                
                # Normalize
                x = normalize(img_gray3)
                images = torch.from_numpy(x).to(DEVICE)
                
                # Inference
                raw = net(images)
                logits = _select_family_logits_from_dict(
                    raw, sample_families=['green8']
                ).detach().cpu().numpy()[0]
                
                # Decode
                pred = greedy_decode(logits)
                
                gt = s['gt'].strip()
                match = pred == gt
                if match:
                    correct += 1
                total += 1
                
                details.append({
                    'id': s['id'],
                    'gt': gt,
                    'pred': pred,
                    'match': match,
                })
                
                if not match:
                    print(f"    {s['id']:>3}: GT={gt:<10} PRED={pred:<10} ✗")
        
        acc = correct / max(total, 1) * 100
        print(f"\n  Accuracy: {correct}/{total} = {acc:.1f}%")
        
        key = f"{dump_name}_{model_name}"
        all_results[key] = {'correct': correct, 'total': total, 'acc': acc, 'details': details}

# ── Summary table ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"{'Dump':<10} {'Model':<6} {'Correct':>8} {'Total':>6} {'Acc':>7}")
print("-" * 40)
for key in sorted(all_results):
    r = all_results[key]
    dump_n, model_n = key.split('_', 1)
    print(f"{dump_n:<10} {model_n:<6} {r['correct']:>8} {r['total']:>6} {r['acc']:>6.1f}%")

# Save
out = ROOT / 'reports' / 'pose_eval' / 'board_ocr_dump_eval.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out}")
