#!/usr/bin/env python3
"""Unified board scan for all 6 ratio points (R00, R06, R10, R20, R35, R100).
Uses consistent methodology across all checkpoints.
"""
import csv, json, sys, os
from pathlib import Path
from collections import Counter
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

DUMP_DIRS = {
    'pos_ocr_dump': Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump'),
    'pos_ocr_dump_2': Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2'),
}

SEG_B_START = 10  # 0-indexed -> frame 11
SEG_B_END = 40    # 0-indexed -> frame 40

PROVINCE_CHARS = set('京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁')

# Training uses ocr_preproc=none + ocr_channel_order=bgr
# ocrin ppm is RGB, convert to BGR before feeding to model
def preprocess_bgr(img_rgb):
    """RGB → BGR (channel swap) matching training ocr_channel_order=bgr"""
    return img_rgb[:, :, ::-1].copy()  # RGB → BGR

def normalize(x):
    x = x.astype(np.float32)
    x -= 127.5
    x *= 0.0078125
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x

def greedy_decode(logits):
    labels = []
    prev = BLANK
    for t in range(logits.shape[1]):
        c = int(np.argmax(logits[:, t]))
        if c != BLANK and c != prev:
            labels.append(c)
        prev = c
    return ''.join(CHARS[c] for c in labels)

def compute_metrics(predictions, ground_truths):
    n = len(predictions)
    exact = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    total_gt_chars = sum(len(g) for g in ground_truths)
    total_pred_chars = sum(len(p) for p in predictions)
    correct_chars = 0
    len_err = 0
    prov_confusion = Counter()
    
    for p, g in zip(predictions, ground_truths):
        # Per-character accuracy using min length alignment
        min_len = min(len(p), len(g))
        correct_chars += sum(1 for i in range(min_len) if p[i] == g[i])
        # Extra characters in pred beyond GT are wrong, extra chars in GT beyond pred are wrong
        correct_chars += max(0, len(g) - len(p)) * 0  # GT excess beyond pred = missed
        
        if len(p) != len(g):
            len_err += 1
        
        if p != g and len(p) > 0 and len(g) > 0:
            pred_prov = p[0]
            gt_prov = g[0]
            if gt_prov in PROVINCE_CHARS and pred_prov != gt_prov:
                prov_confusion[pred_prov] += 1
    
    pp_exact = (exact / n * 100) if n > 0 else 0.0
    pp_char = (correct_chars / total_gt_chars * 100) if total_gt_chars > 0 else 0.0
    len_err_rate = (len_err / n * 100) if n > 0 else 0.0
    mean_len = np.mean([len(p) for p in predictions]) if predictions else 0.0
    
    return {
        'n': n,
        'exact': exact,
        'pp_exact': round(pp_exact, 1),
        'pp_char': round(pp_char, 1),
        'mean_len': round(float(mean_len), 2),
        'len_err_rate': round(len_err_rate, 1),
        'prov_confusion': dict(prov_confusion.most_common(10)),
    }

def scan_exp(exp_name, ckpt_names):
    exp_dir = ROOT / 'experiments' / exp_name
    
    # Load dump samples
    dump_data = {}
    for dump_key, dump_dir in DUMP_DIRS.items():
        index_path = dump_dir / 'index.csv'
        if not index_path.exists():
            continue
        with open(index_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            samples = []
            for row in reader:
                ocrin_rel = row['ocr_input_path'].split('/')[-1]
                ocrin_path = dump_dir / ocrin_rel
                if ocrin_path.exists():
                    samples.append({
                        'id': int(row.get('sample_id', 0)),
                        'gt': row.get('app_text', '').strip(),
                        'ocrin_path': str(ocrin_path),
                    })
        samples.sort(key=lambda s: s['id'])
        dump_data[dump_key] = samples
        dump_data[f'{dump_key}_seg_B'] = [s for s in samples if SEG_B_START <= s['id'] < SEG_B_END]
    
    result = {}
    for ckpt_name in ckpt_names:
        ckpt_path = exp_dir / ckpt_name
        if not ckpt_path.exists():
            continue
        
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        net, cfg = build_lprnet_multihead_from_state_dict(
            state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
        )
        load_multihead_state_dict_compat(net, state, strict=False)
        net.to(DEVICE)
        net.eval()
        
        ckpt_result = {}
        for key in list(dump_data.keys()):
            samples = dump_data[key]
            if not samples:
                continue
            predictions, ground_truths = [], []
            with torch.no_grad():
                for s in samples:
                    img = Image.open(s['ocrin_path']).convert('RGB')
                    img_np = np.array(img, dtype=np.uint8)
                    x = normalize(preprocess_bgr(img_np))
                    images = torch.from_numpy(x).to(DEVICE)
                    raw = net(images)
                    logits = _select_family_logits_from_dict(raw, sample_families=['green8']).detach().cpu().numpy()[0]
                    pred = greedy_decode(logits)
                    predictions.append(pred)
                    ground_truths.append(s['gt'])
            ckpt_result[key] = {
                'metrics': compute_metrics(predictions, ground_truths),
                'predictions': predictions,
            }
        result[ckpt_name] = ckpt_result
    return result

CKPT_NAMES = [
    'best_LPRNet_model.pth', 'Final_LPRNet_model.pth', 'last_LPRNet_model.pth',
    'LPRNet__iteration_2000.pth', 'LPRNet__iteration_4000.pth', 'LPRNet__iteration_6000.pth'
]

EXPS = {
    'R00_replace_only': 'a_ablation_replace_only_20260510',
    'R06_mix_rebuild': 'a_ablation_mix_rebuild_20260510',
    'R10': 'a_ratio_r10_20260510',
    'R20': 'a_ratio_r20_20260510',
    'R35': 'a_ratio_r35_20260510',
    'R100_real_only': 'a_ablation_real_only_20260510',
}

OUT_DIR = ROOT / 'experiments' / 'mix_source_audit_20260510' / 'unified_scan'
os.makedirs(OUT_DIR, exist_ok=True)

all_results = {}
for label, exp_name in EXPS.items():
    print(f"\n{'='*60}")
    print(f"  {label} ({exp_name})")
    print(f"{'='*60}")
    result = scan_exp(exp_name, CKPT_NAMES)
    all_results[label] = result
    
    # Determine board-optimal for this label
    best_checkpoint = None
    best_score = -1
    
    for ckpt, ckpt_data in result.items():
        seg_B = ckpt_data.get('pos_ocr_dump_seg_B', {}).get('metrics', {})
        static = ckpt_data.get('pos_ocr_dump_2', {}).get('metrics', {})
        
        # Print per-checkpoint summary
        seg_pp_char = seg_B.get('pp_char', 0)
        seg_len_err = seg_B.get('len_err_rate', 0)
        static_pp_exact = static.get('pp_exact', 0)
        static_pp_char = static.get('pp_char', 0)
        prov_bias = static.get('prov_confusion', {})
        bias_str = str(list(prov_bias.items())[:3]) if prov_bias else ''
        
        print(f"  {ckpt[:25]:<25} seg: pp_char={seg_pp_char:.1f}% len_err={seg_len_err:.1f}% static: pp_exact={static_pp_exact:.1f}% pp_char={static_pp_char:.1f}% {bias_str}")
        
        # Board-optimal scoring:
        # Static pp_exact is priority, then seg_B pp_char
        score = static_pp_exact * 1000 + seg_pp_char
        if score > best_score:
            best_score = score
            best_checkpoint = ckpt
    
    print(f"  >>> Board-optimal: {best_checkpoint}")
    
    # Save per-label
    out_path = OUT_DIR / f'{label}_scan.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

# Create consolidated summary for comparison
print("\n\n")
print("="*70)
print("  BOARD-OPTIMAL CHECKPOINT COMPARISON")
print("="*70)

# Collect board-optimal for each label
summary = {}
for label in EXPS:
    result = all_results[label]
    best_ckpt, best_score = None, -1
    for ckpt, ckpt_data in result.items():
        seg_B = ckpt_data.get('pos_ocr_dump_seg_B', {}).get('metrics', {})
        static = ckpt_data.get('pos_ocr_dump_2', {}).get('metrics', {})
        score = static.get('pp_exact', 0) * 1000 + seg_B.get('pp_char', 0)
        if score > best_score:
            best_score = score
            best_ckpt = ckpt
    summary[label] = result[best_ckpt] if best_ckpt else {}

header = f"{'Branch':<20} {'Optimal CKPT':<30} {'seg_B pp_char':>14} {'seg_B len_err':>14} {'static pp_exact':>16} {'static pp_char':>16}"
print(header)
print("-" * len(header))
for label in EXPS:
    s = summary.get(label, {})
    seg_B = s.get('pos_ocr_dump_seg_B', {}).get('metrics', {})
    static = s.get('pos_ocr_dump_2', {}).get('metrics', {})
    
    # Find which ckpt
    result = all_results[label]
    best_ckpt = None
    for ckpt, ckpt_data in result.items():
        if ckpt_data == s:
            best_ckpt = ckpt
            break
    
    print(f"{label:<20} {str(best_ckpt or 'N/A'):<30} {seg_B.get('pp_char','N/A'):>14} {seg_B.get('len_err_rate','N/A'):>14} {static.get('pp_exact','N/A'):>16} {static.get('pp_char','N/A'):>16}")

# Save consolidated summary
summary_out = OUT_DIR / 'board_optimal_summary.json'
with open(summary_out, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nSaved summary to {summary_out}")
