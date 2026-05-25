#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

from data.load_data import UnifiedManifestDataset, CHARS
from eval_lpr_detailed import decode_logits
from model.LPRNet import build_lprnet_multihead
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def eval_split(model_path, manifest, split, batch_size, num_workers):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full = UnifiedManifestDataset(
        manifest_path=manifest,
        img_size=[94, 24],
        lpr_max_len=8,
        split_filter=split,
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    idx = [i for i, row in enumerate(full.records) if (row.get('family') or '').strip() == 'green8']
    ds = Subset(full, idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0, enhanced_green_head='expD')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    exact = 0
    total = 0
    first_correct = 0
    province_rows = defaultdict(lambda: {'sample_count': 0, 'exact_plate_correct': 0, 'first_char_correct': 0})
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            images = images.to(device)
            sample_families = list(families)
            logits = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
            for pred_ids, gt_ids in zip(decoded, targets):
                pred = ''.join(CHARS[int(c)] for c in pred_ids)
                gt = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                total += 1
                exact += int(pred == gt)
                if gt:
                    row = province_rows[gt[0]]
                    row['sample_count'] += 1
                    row['exact_plate_correct'] += int(pred == gt)
                    row['first_char_correct'] += int(bool(pred) and pred[0] == gt[0])
                    first_correct += int(bool(pred) and pred[0] == gt[0])

    province_breakdown = {
        k: {
            'sample_count': v['sample_count'],
            'exact_plate_acc': safe_div(v['exact_plate_correct'], v['sample_count']),
            'first_char_acc': safe_div(v['first_char_correct'], v['sample_count']),
        }
        for k, v in sorted(province_rows.items())
    }
    under60 = sorted([k for k, v in province_breakdown.items() if v['exact_plate_acc'] < 0.60])
    return {
        'split': split,
        'sample_count': total,
        'exact_plate_acc': safe_div(exact, total),
        'first_char_acc': safe_div(first_correct, total),
        'province_breakdown': province_breakdown,
        'under60_exact_provinces': under60,
        'under60_exact_count': len(under60),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--batch_size', type=int, default=300)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    report = {
        'model': args.model,
        'manifest': args.manifest,
        'splits': {}
    }
    for split in ['val', 'test']:
        report['splits'][split] = eval_split(args.model, args.manifest, split, args.batch_size, args.num_workers)

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
