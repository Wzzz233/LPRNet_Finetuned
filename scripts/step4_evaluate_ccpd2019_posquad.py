#!/usr/bin/env python3
"""Evaluate LPRNet on CCPD2019 pose quad test set with per-subset breakdown."""

import csv, json, sys, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
from load_data import CHARS, CHARS_DICT, UnifiedManifestDataset
from train_LPRNet import build_lprnet, greedy_decode_logits, collate_wrapper

DATE_TAG = '20260508'
MANIFEST_DIR = ROOT / 'manifests_rebased' / f'blue_ccpd2019_tilt_db_challenge_posquad_{DATE_TAG}'
MODELS = {
    'blue_expert': ROOT / 'experiments/tilt_ocr_obbwarp_v7_from_v6_lenpos3_20260319/weights_stageC/Final_LPRNet_model.pth',
    'posquad_trained': ROOT / 'experiments/blue_ccpd2019_tilt_db_challenge_posquad_20260508/best_LPRNet_model.pth',
}

OCR_PARAMS = dict(
    ocr_crop_mode='obb_warp',
    ocr_resize_mode='letterbox',
    ocr_resize_kernel='nn',
    ocr_preproc='none',
    ocr_channel_order='bgr',
    ocr_quad_pad_ratio=0.0,
)

def evaluate_model(model, test_loader, device):
    model.eval()
    total = 0
    exact_correct = 0
    char_correct = 0
    char_total = 0
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images)  # [N, C, T]
            pred_texts = greedy_decode_logits(logits, CHARS_DICT)
            
            for pred_text, gt_text in zip(pred_texts, labels):
                total += 1
                char_total += len(gt_text)
                pred_clean = ''.join(c for c in pred_text if c != '-')
                if pred_clean == gt_text:
                    exact_correct += 1
                # Character-level accuracy
                for p, g in zip(pred_clean, gt_text):
                    if p == g:
                        char_correct += 1
                        
    return {
        'exact': exact_correct / max(total, 1),
        'exact_count': exact_correct,
        'total': total,
        'char_acc': char_correct / max(char_total, 1),
        'char_correct': char_correct,
        'char_total': char_total,
    }


# ── Per-subset eval ──────────────────────────────────────────────
print("Loading test manifest...", flush=True)
all_rows = []
with open(MANIFEST_DIR / 'test_posquad.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        all_rows.append(row)
print(f"  Total test: {len(all_rows)}", flush=True)

# Group by subset
from collections import defaultdict
by_subset = defaultdict(list)
for row in all_rows:
    source = row.get('source', '')
    # source = ccpd2019_ccpd_tilt, etc.
    subset = source.replace('ccpd2019_', '')
    by_subset[subset].append(row)

# Create per-subset manifests
subset_manifests = {}
for subset, rows in by_subset.items():
    path = Path(f'/tmp/test_{subset}.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for row in rows:
            w.writerow(row)
    subset_manifests[subset] = path
    print(f"  {subset}: {len(rows)} test samples", flush=True)

# Combined manifest
combined_path = Path('/tmp/test_all_combined.csv')
with open(combined_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader()
    for row in all_rows:
        w.writerow(row)

# Prepend all subset manifests + combined
subset_manifests['all'] = combined_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}", flush=True)

results_summary = {}

for model_name, model_path in MODELS.items():
    if not model_path.exists():
        print(f"\n  SKIP {model_name}: {model_path} not found", flush=True)
        continue
    
    print(f"\n{'=' * 60}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Path: {model_path}", flush=True)
    print(f"{'=' * 60}", flush=True)
    
    net = build_lprnet(lpr_max_len=8, head_mode='single', class_num=len(CHARS))
    state = torch.load(str(model_path), map_location='cpu')
    net.load_state_dict(state, strict=False)
    net.to(device)
    print(f"  Model loaded: {sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)
    
    results_summary[model_name] = {}
    
    for subset_name, manifest_path in subset_manifests.items():
        dataset = UnifiedManifestDataset(
            str(manifest_path),
            img_size=[94, 24],
            lpr_max_len=8,
            split_filter='test',
            data_mode='manifest',
            **OCR_PARAMS,
        )
        
        if len(dataset) == 0:
            print(f"  {subset_name}: 0 samples (skipping)", flush=True)
            continue
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=120, shuffle=False,
            num_workers=4, collate_fn=collate_wrapper,
        )
        
        metrics = evaluate_model(net, loader, device)
        results_summary[model_name][subset_name] = metrics
        
        print(f"  {subset_name:20s} exact={metrics['exact']*100:5.1f}%  "
              f"char={metrics['char_acc']*100:5.1f}%  "
              f"({metrics['exact_count']}/{metrics['total']})", flush=True)
    
    # Cleanup
    del net
    torch.cuda.empty_cache()

# ── Print comparison table ──────────────────────────────────────
print(f"\n\n{'=' * 70}", flush=True)
print(f"COMPARISON: Blue Expert vs Posquad-Trained", flush=True)
print(f"{'=' * 70}", flush=True)

if len(results_summary) == 2:
    old_name = 'blue_expert'
    new_name = 'posquad_trained'
    
    header = f"{'Subset':20s} {'Old Expert':>12s} {'Posquad':>12s} {'Diff':>8s}"
    print(header, flush=True)
    print('-' * len(header), flush=True)
    
    for subset in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge', 'all']:
        if subset not in results_summary[old_name]:
            continue
        old_m = results_summary[old_name][subset]
        new_m = results_summary[new_name][subset]
        diff = new_m['exact'] - old_m['exact']
        diff_str = f"+{diff*100:+.1f}%" if diff != 0 else f"{diff*100:+.1f}%"
        print(f"{subset:20s} {old_m['exact']*100:10.1f}%  {new_m['exact']*100:10.1f}%  {diff_str:>8s}", flush=True)
    
    print(f"\n  Detailed:", flush=True)
    for model_name in [old_name, new_name]:
        print(f"  {model_name}:", flush=True)
        for subset in ['ccpd_tilt', 'ccpd_db', 'ccpd_challenge', 'all']:
            if subset not in results_summary.get(model_name, {}):
                continue
            m = results_summary[model_name][subset]
            print(f"    {subset:20s} exact={m['exact']*100:5.1f}%  char={m['char_acc']*100:5.1f}%  "
                  f"({m['exact_count']}/{m['total']})", flush=True)

# Save to JSON
out_path = Path('/tmp/ccpd2019_posquad_eval_results.json')
json.dump(results_summary, out_path, ensure_ascii=False, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
