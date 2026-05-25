#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from collections import Counter, defaultdict

from data.load_data import CHARS
from eval_lpr_detailed import decode_logits
from model.LPRNet import build_lprnet_multihead
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits
from data.load_data import UnifiedManifestDataset

import torch
from torch.utils.data import DataLoader, Subset


def classify_error(gt, pred):
    if pred == gt:
        return 'exact'
    if len(pred) != len(gt):
        if pred and gt and pred[0] != gt[0]:
            return 'length_plus_province'
        return 'length'
    first_bad = bool(gt) and (not pred or pred[0] != gt[0])
    pos2_bad = len(gt) > 1 and (len(pred) <= 1 or pred[1] != gt[1])
    tail_bad = any((len(pred) <= i or pred[i] != gt[i]) for i in range(2, len(gt)))
    if first_bad and not pos2_bad and not tail_bad:
        return 'province_only'
    if first_bad and (pos2_bad or tail_bad):
        return 'province_plus_other'
    if (not first_bad) and pos2_bad and not tail_bad:
        return 'pos2_only'
    if (not first_bad) and (not pos2_bad) and tail_bad:
        return 'tail_only'
    if (not first_bad) and pos2_bad and tail_bad:
        return 'pos2_plus_tail'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--provinces', default='浙,苏,沪,粤,津,湘')
    args = ap.parse_args()

    target_provs = [x.strip() for x in args.provinces.split(',') if x.strip()]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full = UnifiedManifestDataset(
        manifest_path=args.manifest,
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
    idx = [i for i, row in enumerate(full.records) if (row.get('family') or '').strip() == 'green8']
    ds = Subset(full, idx)
    loader = DataLoader(ds, batch_size=300, shuffle=False, num_workers=4, collate_fn=collate_fn)

    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0, enhanced_green_head='expD')
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device)
    net.eval()

    rows = []
    rec_idx = 0
    base_records = [full.records[i] for i in idx]
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
                row = base_records[rec_idx]
                rec_idx += 1
                prov = gt[:1] if gt else ''
                if prov not in target_provs:
                    continue
                rows.append({
                    'img_path': row.get('img_path'),
                    'dataset_name': row.get('dataset_name'),
                    'split': row.get('split'),
                    'source': row.get('source'),
                    'text': gt,
                    'pred': pred,
                    'province': prov,
                    'error_type': classify_error(gt, pred),
                    'exact': pred == gt,
                    'first_ok': bool(pred) and bool(gt) and pred[0] == gt[0],
                    'pos2_ok': len(gt) > 1 and len(pred) > 1 and pred[1] == gt[1],
                    'tail_acc': 0.0 if len(gt) <= 2 else sum(1 for i in range(2, len(gt)) if len(pred) > i and pred[i] == gt[i]) / max(1, len(gt)-2),
                })

    out = {}
    for p in target_provs:
        sub = [r for r in rows if r['province'] == p]
        errs = [r for r in sub if not r['exact']]
        by_err = Counter(r['error_type'] for r in errs)
        by_src = defaultdict(Counter)
        for r in sub:
            by_src[r['source']]['sample_count'] += 1
            by_src[r['source']]['exact_correct'] += int(r['exact'])
            by_src[r['source']]['first_correct'] += int(r['first_ok'])
        out[p] = {
            'sample_count': len(sub),
            'exact_acc': round(sum(1 for r in sub if r['exact']) / len(sub), 4) if sub else 0.0,
            'first_acc': round(sum(1 for r in sub if r['first_ok']) / len(sub), 4) if sub else 0.0,
            'pos2_acc': round(sum(1 for r in sub if r['pos2_ok']) / max(1, sum(1 for r in sub if len(r['text']) > 1)), 4) if sub else 0.0,
            'mean_tail_acc': round(sum(r['tail_acc'] for r in sub) / len(sub), 4) if sub else 0.0,
            'error_type_breakdown': dict(by_err.most_common()),
            'by_source': {
                s: {
                    'sample_count': c['sample_count'],
                    'exact_acc': round(c['exact_correct'] / c['sample_count'], 4) if c['sample_count'] else 0.0,
                    'first_acc': round(c['first_correct'] / c['sample_count'], 4) if c['sample_count'] else 0.0,
                } for s, c in by_src.items()
            },
            'top_errors': errs[:20],
        }

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
