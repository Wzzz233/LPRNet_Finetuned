#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import UnifiedManifestDataset, CHARS
from eval_lpr_detailed import decode_logits
from LPRNet_multihead import build_lprnet_multihead, build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def build_model(model_path, device):
    state = torch.load(model_path, map_location=device)
    net, _cfg = build_lprnet_multihead_from_state_dict(
        state,
        lpr_max_len=8,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0,
    )
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net


def normalize_source(row):
    src = row.get('source', '')
    if src == 'real':
        return 'real'
    return 'synthetic'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--batch_size', type=int, default=300)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = build_model(args.model, device)

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
    keep = [i for i, row in enumerate(full.records) if row.get('family') == 'green8' and row.get('img_path') and Path(row.get('img_path')).exists()]
    ds = Subset(full, keep)
    records = [full.records[i] for i in keep]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    stats = defaultdict(lambda: {'count': 0, 'exact': 0, 'first': 0, 'province': defaultdict(lambda: {'count': 0, 'exact': 0, 'first': 0})})
    seen = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            batch_records = records[seen:seen + len(families)]
            start = 0
            targets = []
            for length in lengths:
                gt_ids = labels[start:start + length].numpy().tolist()
                targets.append(''.join(CHARS[int(c)] for c in gt_ids))
                start += length
            images = images.to(device)
            sample_families = list(families)
            logits = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
            preds = [''.join(CHARS[int(c)] for c in pred_ids) for pred_ids in decoded]
            for row, gt, pred in zip(batch_records, targets, preds):
                bucket = normalize_source(row)
                st = stats[bucket]
                st['count'] += 1
                st['exact'] += int(pred == gt)
                st['first'] += int(bool(pred) and bool(gt) and pred[0] == gt[0])
                prov = gt[0] if gt else ''
                pst = st['province'][prov]
                pst['count'] += 1
                pst['exact'] += int(pred == gt)
                pst['first'] += int(bool(pred) and bool(gt) and pred[0] == gt[0])
            seen += len(families)

    out = {'model': args.model, 'manifest': args.manifest, 'decode_mode': 'family_aware_beam', 'buckets': {}}
    for bucket, st in stats.items():
        province_metrics = {}
        macro_exact_sum = 0.0
        macro_first_sum = 0.0
        province_count = 0
        for prov, pst in sorted(st['province'].items()):
            exact = safe_div(pst['exact'], pst['count'])
            first = safe_div(pst['first'], pst['count'])
            province_metrics[prov] = {'sample_count': pst['count'], 'exact_plate_acc': exact, 'first_char_acc': first}
            macro_exact_sum += exact
            macro_first_sum += first
            province_count += 1
        out['buckets'][bucket] = {
            'sample_count': st['count'],
            'exact_plate_acc': safe_div(st['exact'], st['count']),
            'first_char_acc': safe_div(st['first'], st['count']),
            'province_macro_exact_acc': safe_div(macro_exact_sum, province_count),
            'province_macro_first_char_acc': safe_div(macro_first_sum, province_count),
            'province_metrics': province_metrics,
        }

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
