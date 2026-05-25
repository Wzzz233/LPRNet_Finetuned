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

ALL_PROVINCES = ['京','津','沪','渝','冀','晋','蒙','辽','吉','黑','苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新']


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def evaluate(model_path, manifest_path, split, dataset_name, batch_size, num_workers):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full = UnifiedManifestDataset(
        manifest_path=manifest_path,
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
    idx = [i for i, row in enumerate(full.records) if (row.get('family') or '').strip() == 'green8' and (row.get('dataset_name') or '') == dataset_name]
    ds = Subset(full, idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0, enhanced_green_head='expD')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    total = 0
    exact = 0
    first = 0
    rows = defaultdict(lambda: {'sample_count': 0, 'exact_plate_correct': 0, 'first_char_correct': 0})
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
                prov = gt[:1]
                total += 1
                exact += int(pred == gt)
                first += int(bool(pred) and pred[0] == gt[0])
                rows[prov]['sample_count'] += 1
                rows[prov]['exact_plate_correct'] += int(pred == gt)
                rows[prov]['first_char_correct'] += int(bool(pred) and pred[0] == gt[0])

    province_breakdown = {}
    for p in ALL_PROVINCES:
        v = rows.get(p, {'sample_count': 0, 'exact_plate_correct': 0, 'first_char_correct': 0})
        province_breakdown[p] = {
            'sample_count': v['sample_count'],
            'exact_plate_acc': safe_div(v['exact_plate_correct'], v['sample_count']),
            'first_char_acc': safe_div(v['first_char_correct'], v['sample_count']),
        }
    return {
        'split': split,
        'dataset_name': dataset_name,
        'sample_count': total,
        'exact_plate_acc': safe_div(exact, total),
        'first_char_acc': safe_div(first, total),
        'province_breakdown': province_breakdown,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--batch_size', type=int, default=300)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, 'r', encoding='utf-8')))
    target_sets = []
    for split in ['val', 'test']:
        names = sorted({r.get('dataset_name') for r in rows if r.get('split') == split and r.get('family') == 'green8'})
        for name in names:
            target_sets.append((split, name))

    report = {
        'model': args.model,
        'manifest': args.manifest,
        'datasets': {}
    }
    for split, name in target_sets:
        report['datasets'][f'{split}:{name}'] = evaluate(args.model, args.manifest, split, name, args.batch_size, args.num_workers)

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
