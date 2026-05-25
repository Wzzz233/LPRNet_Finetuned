#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.load_data import CCPDBoardDataLoader, CHARS
from eval_lpr_detailed import decode_logits
from model.LPRNet import build_lprnet_multihead
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits


def classify_error(gt, pred):
    if pred == gt:
        return 'correct'
    if not pred:
        return 'empty'
    if len(pred) != len(gt):
        return 'length'
    diffs = [i for i, (g, p) in enumerate(zip(gt, pred)) if g != p]
    if not diffs:
        return 'other'
    if diffs == [0]:
        return 'province_only'
    if diffs == [1]:
        return 'pos2_only'
    if all(i >= 2 for i in diffs):
        return 'tail_only'
    if 0 in diffs and len(diffs) > 1:
        return 'province_plus_other'
    if 1 in diffs and len(diffs) > 1 and 0 not in diffs:
        return 'pos2_plus_tail'
    return 'mixed'


def eval_manifest(model_path, txt, image_root, out_details=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds = CCPDBoardDataLoader(
        [image_root],
        imgSize=[94,24],
        lpr_max_len=8,
        txt_file=txt,
        ocr_channel_order='bgr',
        ocr_crop_mode='obb_warp',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate_fn)
    net = build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0, enhanced_green_head='expD')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    rows = []
    idx0 = 0
    with torch.no_grad():
        for images, labels, lengths, _families in loader:
            start = 0
            targets = []
            fams = []
            for length in lengths:
                tgt = labels[start:start+length].numpy()
                targets.append(tgt)
                fams.append('green8')
                start += length
            images = images.to(device)
            logits = forward_family_logits(net, images, sample_families=fams).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=fams)
            for j, (pred_ids, gt_ids) in enumerate(zip(decoded, targets)):
                pred = ''.join(CHARS[int(c)] for c in pred_ids)
                gt = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                img_path = ds.img_paths[idx0 + j]
                rows.append({
                    'image_path': img_path,
                    'gt': gt,
                    'pred': pred,
                    'province': gt[:1],
                    'error_type': classify_error(gt, pred),
                    'exact': pred == gt,
                    'first_ok': bool(pred) and bool(gt) and pred[0] == gt[0],
                })
            idx0 += len(decoded)
    if out_details:
        Path(out_details).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    return rows


def summarize(rows, provinces):
    byp = {}
    for p in provinces:
        sub = [r for r in rows if r['province'] == p]
        errs = [r for r in sub if not r['exact']]
        counter = Counter(r['error_type'] for r in errs)
        byp[p] = {
            'sample_count': len(sub),
            'exact_acc': round(sum(1 for r in sub if r['exact']) / len(sub), 4) if sub else 0.0,
            'first_acc': round(sum(1 for r in sub if r['first_ok']) / len(sub), 4) if sub else 0.0,
            'error_type_breakdown': dict(counter.most_common()),
            'top_errors': errs[:12],
        }
    return byp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--real_txt', required=True)
    ap.add_argument('--real_root', required=True)
    ap.add_argument('--synth_txt', required=True)
    ap.add_argument('--synth_root', required=True)
    ap.add_argument('--provinces', required=True)
    ap.add_argument('--out_json', required=True)
    args = ap.parse_args()
    provinces = [x.strip() for x in args.provinces.split(',') if x.strip()]
    real_rows = eval_manifest(args.model, args.real_txt, args.real_root)
    synth_rows = eval_manifest(args.model, args.synth_txt, args.synth_root)
    report = {
        'provinces': provinces,
        'real_summary': summarize(real_rows, provinces),
        'synth_summary': summarize(synth_rows, provinces),
    }
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
