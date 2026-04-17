#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import UnifiedManifestDataset, CHARS
from eval_lpr_detailed import decode_logits
from LPRNet_multihead import build_lprnet_multihead, build_lprnet_multihead_from_state_dict
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def load_model(model_path, device):
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
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--ocr_preproc', default='none', choices=['none', 'raw', 'gray', 'gray3', 'bin'], help='OCR preprocess mode for evaluation')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full = UnifiedManifestDataset(
        manifest_path=args.manifest,
        img_size=[94, 24],
        lpr_max_len=8,
        split_filter='test',
        ocr_channel_order='bgr',
        ocr_crop_mode='board_dump',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc=args.ocr_preproc,
        ocr_min_occ_ratio=1.0,
        ocr_quad_pad_ratio=0.0,
    )
    idx = [i for i, row in enumerate(full.records) if row.get('img_path') and Path(row.get('img_path')).exists()]
    ds = Subset(full, idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    net = load_model(args.model, device)

    preds = []
    gts = []
    rel_paths = []
    with torch.no_grad():
        base_idx = 0
        for images, labels, lengths, families in loader:
            batch_records = [full.records[idx[i + base_idx]] for i in range(len(lengths))]
            base_idx += len(lengths)
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            images = images.to(device)
            sample_families = list(families)
            logits = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
            for pred_ids, gt_ids, rec in zip(decoded, targets, batch_records):
                pred = ''.join(CHARS[int(c)] for c in pred_ids)
                gt = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                preds.append(pred)
                gts.append(gt)
                rel_paths.append(rec.get('img_rel_path') or rec.get('img_path'))

    total = len(preds)
    exact = sum(int(p == g) for p, g in zip(preds, gts))
    first = sum(int(bool(p) and bool(g) and p[0] == g[0]) for p, g in zip(preds, gts))
    pred_counter = Counter(preds)
    major_pred, major_count = ('', 0)
    if pred_counter:
        major_pred, major_count = pred_counter.most_common(1)[0]
    gt_counter = Counter(gts)
    gt_text, gt_count = ('', 0)
    if gt_counter:
        gt_text, gt_count = gt_counter.most_common(1)[0]

    report = {
        'model': args.model,
        'manifest': args.manifest,
        'sample_count': total,
        'gt_text': gt_text,
        'same_track_exact_acc': safe_div(exact, total),
        'same_track_first_char_acc': safe_div(first, total),
        'same_track_unique_prediction_count': len(pred_counter),
        'major_prediction': major_pred,
        'major_prediction_ratio': safe_div(major_count, total),
        'gt_prediction_ratio': safe_div(pred_counter.get(gt_text, 0), total),
        'anhui_prefix_rate': safe_div(sum(1 for p in preds if p.startswith('皖')), total),
        'empty_pred_rate': safe_div(sum(1 for p in preds if not p), total),
        'template_collapse_rate': safe_div(major_count, total),
        'prediction_counter': dict(pred_counter.most_common()),
        'samples': [
            {
                'img_rel_path': rel,
                'gt': gt,
                'pred': pred,
                'exact': pred == gt,
                'first_char_ok': bool(pred) and bool(gt) and pred[0] == gt[0],
            }
            for rel, gt, pred in zip(rel_paths, gts, preds)
        ],
    }
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
