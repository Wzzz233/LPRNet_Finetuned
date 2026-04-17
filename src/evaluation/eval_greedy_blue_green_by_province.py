#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import UnifiedManifestDataset, CHARS
from LPRNet_multihead import build_lprnet_multihead, build_lprnet_multihead_from_state_dict
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def greedy_decode_logits(prebs):
    preb_labels = []
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = []
        for j in range(preb.shape[1]):
            preb_label.append(int(np.argmax(preb[:, j], axis=0)))
        no_repeat_blank_label = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    return preb_labels


def build_model(model_path, device):
    state = torch.load(model_path, map_location=device)
    net, _cfg = build_lprnet_multihead_from_state_dict(
        state,
        lpr_max_len=8,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0,
    )
    net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


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
    idx = [i for i, row in enumerate(full.records) if row.get('img_path') and Path(row.get('img_path')).exists()]
    ds = Subset(full, idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    family_rows = {}
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                gt_ids = labels[start:start + length].numpy().tolist()
                gt = ''.join(CHARS[int(c)] for c in gt_ids)
                targets.append(gt)
                start += length
            images = images.to(device)
            sample_families = list(families)
            logits = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded_ids = greedy_decode_logits(logits)
            preds = [''.join(CHARS[int(c)] for c in pred_ids) for pred_ids in decoded_ids]

            for fam, gt, pred in zip(sample_families, targets, preds):
                fam_row = family_rows.setdefault(fam, {'sample_count': 0, 'exact_correct': 0, 'first_correct': 0, 'province': {}})
                fam_row['sample_count'] += 1
                fam_row['exact_correct'] += int(pred == gt)
                if gt:
                    fam_row['first_correct'] += int(bool(pred) and pred[0] == gt[0])
                    prov_row = fam_row['province'].setdefault(gt[0], {'sample_count': 0, 'exact_correct': 0, 'first_correct': 0})
                    prov_row['sample_count'] += 1
                    prov_row['exact_correct'] += int(pred == gt)
                    prov_row['first_correct'] += int(bool(pred) and pred[0] == gt[0])

    out = {
        'model': args.model,
        'manifest': args.manifest,
        'decode_mode': 'greedy',
        'families': {}
    }
    for fam, row in sorted(family_rows.items()):
        province_metrics = {}
        for prov, prow in sorted(row['province'].items()):
            province_metrics[prov] = {
                'sample_count': prow['sample_count'],
                'exact_plate_acc': safe_div(prow['exact_correct'], prow['sample_count']),
                'first_char_acc': safe_div(prow['first_correct'], prow['sample_count']),
            }
        out['families'][fam] = {
            'sample_count': row['sample_count'],
            'exact_plate_acc': safe_div(row['exact_correct'], row['sample_count']),
            'first_char_acc': safe_div(row['first_correct'], row['sample_count']),
            'province_metrics': province_metrics,
        }

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
