#!/usr/bin/env python3
"""
Evaluate trained embassy/police v2 models using the training pipeline's dataset loading
to ensure consistent preprocessing. Must set CHARS from correct keys file.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "training"))

# Override CHARS from keys file BEFORE importing dataset classes
def override_chars(keys_path):
    """Override global CHARS with keys from file."""
    import load_data as ld
    custom_chars = [l.strip() for l in open(keys_path) if l.strip()]
    custom_chars.append('-')  # CTC blank
    ld.CHARS.clear()
    ld.CHARS.extend(custom_chars)
    ld.CHARS_DICT.clear()
    ld.CHARS_DICT.update({c: i for i, c in enumerate(custom_chars)})
    return custom_chars, len(custom_chars)  # keys list, class_num

from LPRNet import build_lprnet
from training.train_LPRNet import UnifiedManifestDataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--family", required=True, choices=["embassy", "police"])
    ap.add_argument("--date-tag", default="20260601")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()

def decode_ctc(logits, keys, blank_idx):
    preds = logits.argmax(axis=0)
    result = []
    prev = -1
    for p in preds:
        if p != blank_idx and p != prev:
            result.append(keys[p] if p < len(keys) else '?')
        prev = p
    return ''.join(result), preds

def compute_metrics(gts, preds, keys, family):
    total = len(gts)
    exact_ok = sum(1 for g, p in zip(gts, preds) if g == p)
    len_errors = sum(1 for g, p in zip(gts, preds) if len(g) != len(p))
    illegal_chars = sum(1 for p in preds for ch in p if ch not in keys)
    
    per_pos = defaultdict(lambda: {"ok": 0, "total": 0})
    prov_conf = Counter()
    
    for gt, pred in zip(gts, preds):
        max_len = max(len(gt), len(pred))
        for pos in range(max_len):
            per_pos[pos]["total"] += 1
            if pos < len(gt) and pos < len(pred) and gt[pos] == pred[pos]:
                per_pos[pos]["ok"] += 1
        
        if family == "police" and len(gt) >= 7 and len(pred) >= 7:
            if gt[0] != pred[0]:
                prov_conf[f"{gt[0]}→{pred[0]}"] += 1
    
    metrics = {
        "total": total,
        "exact_ok": exact_ok,
        "exact_acc": exact_ok / total if total > 0 else 0,
        "len_errors": len_errors,
        "illegal_chars": illegal_chars,
    }
    for pos in sorted(per_pos.keys()):
        p = per_pos[pos]
        metrics[f"pos{pos}_acc"] = p["ok"] / p["total"] if p["total"] > 0 else 0
    
    return metrics, prov_conf

def main():
    args = parse_args()
    PROJECT_ROOT = Path("/home/wzzz/LPRNet")
    keys_path = str(PROJECT_ROOT / "keys" / f"{args.family}_keys.txt")
    
    # Override CHARS
    keys_list, class_num = override_chars(keys_path)
    blank_idx = class_num - 1
    
    print(f"Keys: {len(keys_list)} chars, class_num={class_num}, blank_idx={blank_idx}")
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_lprnet(lpr_max_len=8, phase=False, class_num=class_num, dropout_rate=0)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"Model loaded: {args.checkpoint}")
    
    # Eval splits
    splits = [
        ("val_clean", f"manifests_rebased/special_split_v2_{args.date_tag}/val_clean_{args.family}.csv"),
        ("val_hard", f"manifests_rebased/special_split_v2_{args.date_tag}/val_hard_{args.family}.csv"),
        ("v1_val", f"manifests_rebased/special_split_20260526/val_{args.family}_only.csv"),
    ]
    
    results = {}
    all_errors = []
    
    for split_name, manifest_rel in splits:
        manifest_path = PROJECT_ROOT / manifest_rel
        if not manifest_path.exists():
            print(f"[{split_name}] manifest not found, skipping")
            continue
        
        print(f"\n=== {split_name} ===")
        
        # Use training's dataset loader for consistent preprocessing
        dataset = UnifiedManifestDataset(
            manifest_path=str(manifest_path),
            img_size=[94, 24],
            lpr_max_len=8,
            split_filter=split_name if 'v1' not in split_name else 'val',
        )
        print(f"  Dataset: {len(dataset)} samples")
        
        gts, preds = [], []
        for i in range(len(dataset)):
            img, label_ids, _, _ = dataset[i]
            # Decode label from IDs
            gt = ''.join(keys_list[li] for li in label_ids if li < len(keys_list))
            
            with torch.no_grad():
                # img is numpy array (C,H,W), add batch dim
                img_t = torch.from_numpy(img).unsqueeze(0).to(device)
                logits = model(img_t).cpu().numpy()[0]
            text, _ = decode_ctc(logits, keys_list, blank_idx)
            
            gts.append(gt)
            preds.append(text)
            if gt != text:
                all_errors.append({"gt": gt, "pred": text, "idx": i})
        
        metrics, prov_conf = compute_metrics(gts, preds, keys_list, args.family)
        results[split_name] = metrics
        
        print(f"  Exact: {metrics['exact_ok']}/{metrics['total']} = {metrics['exact_acc']*100:.2f}%")
        print(f"  Len errors: {metrics['len_errors']}")
        print(f"  Illegal chars: {metrics['illegal_chars']}")
        for pos in sorted([k for k in metrics if k.startswith('pos')]):
            print(f"  {pos}: {metrics[pos]*100:.2f}%")
        if prov_conf:
            for conf, cnt in prov_conf.most_common(20):
                print(f"  Province: {conf}: {cnt}")
        
        # Save errors
        out_dir = args.output or os.path.dirname(args.checkpoint.rstrip('/'))
        with open(os.path.join(out_dir, f"{split_name}_errors.csv"), "w") as f:
            f.write("gt,pred\n")
            for e in all_errors[:200]:
                f.write(f"{e['gt']},{e['pred']}\n")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    for split_name, metrics in results.items():
        print(f"  {split_name}: {metrics['exact_acc']*100:.2f}%")
    
    # Save results
    out_dir = args.output or os.path.dirname(args.checkpoint.rstrip('/'))
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "family": args.family,
            "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        }, f, indent=2)
    print(f"Results: {os.path.join(out_dir, 'eval_results.json')}")

if __name__ == "__main__":
    main()
