#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

from data.load_data import UnifiedManifestDataset, CHARS
from model.LPRNet import build_lprnet_multihead
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits
from eval_lpr_detailed import decode_logits


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def tier_from_row(row):
    for key in ("img_rel_path", "img_path"):
        v = row.get(key) or ""
        m = re.search(r"/images/(?:train|val|test)/(simple|hard|extreme)/", v)
        if m:
            return m.group(1)
    return None


def bucket_from_row(row, mode):
    family = (row.get("family") or "").strip()
    if family != "green8":
        return None
    if mode == "real":
        if (row.get("source") or "").strip() == "real":
            return "real"
        return None
    if mode == "tier3":
        if (row.get("dataset_name") or "").strip() != "green_edgefit_tier3_full_v3_su_conservative":
            return None
        return tier_from_row(row)
    return None


def evaluate_subset(model, device, manifest_path, mode, batch_size=300, num_workers=4):
    ds = UnifiedManifestDataset(
        manifest_path=manifest_path,
        img_size=[94, 24],
        lpr_max_len=8,
        split_filter='test',
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    selected = []
    buckets = []
    for i, row in enumerate(ds.records):
        b = bucket_from_row(row, mode)
        if b is not None:
            selected.append(i)
            buckets.append(b)

    subset = Subset(ds, selected)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    stats = defaultdict(lambda: defaultdict(lambda: {"sample_count": 0, "exact_correct": 0, "first_correct": 0}))
    seen = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            images = images.to(device)
            sample_families = list(families)
            logits = forward_family_logits(model, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
            for i, (pred_ids, gt_ids) in enumerate(zip(decoded, targets)):
                row = ds.records[selected[seen + i]]
                bucket = buckets[seen + i]
                gt = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                pred = ''.join(CHARS[int(c)] for c in pred_ids)
                province = gt[0] if gt else '?'
                cell = stats[bucket][province]
                cell['sample_count'] += 1
                cell['exact_correct'] += int(pred == gt)
                cell['first_correct'] += int(bool(gt) and bool(pred) and gt[0] == pred[0])
            seen += len(targets)

    out = {}
    for bucket, provs in stats.items():
        out[bucket] = {}
        for prov, cell in sorted(provs.items()):
            out[bucket][prov] = {
                'sample_count': cell['sample_count'],
                'exact_plate_acc': safe_div(cell['exact_correct'], cell['sample_count']),
                'first_char_acc': safe_div(cell['first_correct'], cell['sample_count']),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--real_manifest', required=True)
    ap.add_argument('--tier3_manifest', required=True)
    ap.add_argument('--out_json', required=True)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0, enhanced_green_head='expD')
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    real = evaluate_subset(model, device, args.real_manifest, 'real')
    tier3 = evaluate_subset(model, device, args.tier3_manifest, 'tier3')

    report = {
        'model': args.model,
        'real_manifest': args.real_manifest,
        'tier3_manifest': args.tier3_manifest,
        'real': real.get('real', {}),
        'simple': tier3.get('simple', {}),
        'hard': tier3.get('hard', {}),
        'extreme': tier3.get('extreme', {}),
    }
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
