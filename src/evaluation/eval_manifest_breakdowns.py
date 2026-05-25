#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict, Counter
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
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
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


def province_from_text(text: str) -> str:
    return text[0] if text else ''


def err_type(gt: str, pred: str) -> str:
    if pred == gt:
        return 'exact'
    if not pred:
        return 'empty'
    if len(pred) != len(gt):
        return 'length_mismatch'
    if gt and pred and gt[0] != pred[0]:
        return 'province_confusion'
    diffs = sum(1 for a, b in zip(gt, pred) if a != b)
    if diffs == 1:
        return 'single_char'
    if diffs == 2:
        return 'two_char'
    return 'multi_char'


def bucket_key(row, mode):
    src = row.get('source', 'unknown')
    fam = row.get('family', '')
    if mode == 'source_family':
        return f'{src}__{fam}'
    if mode == 'ccpd2020_province':
        if src != 'ccpd2020' or fam != 'green8':
            return None
        return province_from_text(row.get('text', ''))
    if mode == 'crpd_kind':
        if not src.startswith('crpd_'):
            return None
        return src
    if mode == 'green_synth_compare':
        if src not in ('green_exact_quad', 'green_edgefit_simple'):
            return None
        return src
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--mode', required=True, choices=['source_family', 'ccpd2020_province', 'crpd_kind', 'green_synth_compare'])
    ap.add_argument('--batch_size', type=int, default=240)
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
        ocr_preproc='gray3',
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    keep = [i for i, row in enumerate(full.records) if row.get('img_path') and Path(row.get('img_path')).exists()]
    ds = Subset(full, keep)
    records = [full.records[i] for i in keep]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    stats = defaultdict(lambda: {
        'count': 0, 'exact': 0, 'first': 0, 'char_ok': 0, 'char_total': 0,
        'family_counter': Counter(), 'source_counter': Counter(), 'province_counter': Counter(),
        'error_type_counter': Counter(), 'bad_cases': []
    })

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
            for row, fam, gt, pred in zip(batch_records, sample_families, targets, preds):
                key = bucket_key(row, args.mode)
                if key is None:
                    continue
                st = stats[key]
                st['count'] += 1
                st['exact'] += int(pred == gt)
                st['first'] += int(bool(pred) and bool(gt) and pred[0] == gt[0])
                m = min(len(pred), len(gt))
                st['char_ok'] += sum(1 for i in range(m) if pred[i] == gt[i])
                st['char_total'] += len(gt)
                st['family_counter'][fam] += 1
                st['source_counter'][row.get('source', 'unknown')] += 1
                prov = province_from_text(gt)
                st['province_counter'][prov] += 1
                et = err_type(gt, pred)
                st['error_type_counter'][et] += 1
                if len(st['bad_cases']) < 20 and pred != gt:
                    st['bad_cases'].append({
                        'img_path': row.get('img_path', ''),
                        'gt': gt,
                        'pred': pred,
                        'family': fam,
                        'source': row.get('source', ''),
                        'error_type': et,
                    })
            seen += len(families)

    out = {'model': args.model, 'manifest': args.manifest, 'mode': args.mode, 'buckets': {}}
    for key, st in sorted(stats.items()):
        out['buckets'][key] = {
            'sample_count': st['count'],
            'exact_plate_acc': safe_div(st['exact'], st['count']),
            'first_char_acc': safe_div(st['first'], st['count']),
            'char_acc': safe_div(st['char_ok'], st['char_total']),
            'family_counter': dict(st['family_counter']),
            'source_counter': dict(st['source_counter']),
            'province_counter': dict(st['province_counter']),
            'error_type_counter': dict(st['error_type_counter']),
            'bad_cases': st['bad_cases'],
        }

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
