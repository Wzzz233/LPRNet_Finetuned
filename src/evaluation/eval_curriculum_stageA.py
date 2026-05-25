#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
from train_LPRNet import _select_family_logits_from_dict
from firstchar_fusion import extract_province_logits, fuse_first_char


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--batch_size', type=int, default=300)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--province-fusion-mode', default='none', choices=['none', 'replace_all', 'replace_if_confident', 'replace_if_not_cjk'])
    ap.add_argument('--province-conf-threshold', type=float, default=0.55)
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
        gray3_prob=1.0,
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    idx = [i for i, row in enumerate(full.records) if row.get('img_path') and Path(row.get('img_path')).exists()]
    ds = Subset(full, idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    family_rows = {}
    base_family_rows = {}
    fusion_reasons = {}
    fusion_changed_count = 0
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
            raw_outputs = net(images)
            logits = _select_family_logits_from_dict(raw_outputs, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
            preds = [''.join(CHARS[int(c)] for c in pred_ids) for pred_ids in decoded]
            province_prob = None
            if args.province_fusion_mode != 'none':
                province_logits = extract_province_logits(raw_outputs, sample_families)
                if province_logits is None:
                    raise RuntimeError(f'--province-fusion-mode={args.province_fusion_mode} but model has no usable province logits')
                province_prob = F.softmax(province_logits, dim=1).detach().cpu().numpy()

            for i, (fam, gt, base_pred) in enumerate(zip(sample_families, targets, preds)):
                pred = base_pred
                if fam == 'green8' and province_prob is not None:
                    province_idx = int(province_prob[i].argmax())
                    province_conf = float(province_prob[i][province_idx])
                    province_char = CHARS[province_idx] if province_idx < len(CHARS) else ''
                    pred, changed, reason = fuse_first_char(base_pred, province_char, province_conf, args.province_fusion_mode, args.province_conf_threshold)
                    fusion_reasons[reason] = fusion_reasons.get(reason, 0) + 1
                    fusion_changed_count += int(changed)
                fam_row = family_rows.setdefault(fam, {'sample_count': 0, 'exact_correct': 0, 'first_correct': 0, 'province': {}})
                fam_row['sample_count'] += 1
                fam_row['exact_correct'] += int(pred == gt)
                base_fam_row = base_family_rows.setdefault(fam, {'sample_count': 0, 'exact_correct': 0, 'first_correct': 0, 'province': {}})
                base_fam_row['sample_count'] += 1
                base_fam_row['exact_correct'] += int(base_pred == gt)
                if gt:
                    fam_row['first_correct'] += int(bool(pred) and pred[0] == gt[0])
                    prov_row = fam_row['province'].setdefault(gt[0], {'sample_count': 0, 'exact_correct': 0, 'first_correct': 0})
                    prov_row['sample_count'] += 1
                    prov_row['exact_correct'] += int(pred == gt)
                    prov_row['first_correct'] += int(bool(pred) and pred[0] == gt[0])

                    base_fam_row['first_correct'] += int(bool(base_pred) and base_pred[0] == gt[0])
                    base_prov_row = base_fam_row['province'].setdefault(gt[0], {'sample_count': 0, 'exact_correct': 0, 'first_correct': 0})
                    base_prov_row['sample_count'] += 1
                    base_prov_row['exact_correct'] += int(base_pred == gt)
                    base_prov_row['first_correct'] += int(bool(base_pred) and base_pred[0] == gt[0])

    def summarize_family_rows(rows, base_rows):
        out_families = {}
        for fam, row in sorted(rows.items()):
            province_metrics = {}
            weak_first = []
            weak_exact = []
            for prov, prow in sorted(row['province'].items()):
                exact = safe_div(prow['exact_correct'], prow['sample_count'])
                first = safe_div(prow['first_correct'], prow['sample_count'])
                base_prow = base_rows[fam]['province'][prov]
                province_metrics[prov] = {
                    'sample_count': prow['sample_count'],
                    'exact_plate_acc': exact,
                    'first_char_acc': first,
                    'base_exact_plate_acc': safe_div(base_prow['exact_correct'], base_prow['sample_count']),
                    'base_first_char_acc': safe_div(base_prow['first_correct'], base_prow['sample_count']),
                }
                if first < 0.60:
                    weak_first.append({'province': prov, 'sample_count': prow['sample_count'], 'first_char_acc': first, 'exact_plate_acc': exact})
                if exact < 0.60:
                    weak_exact.append({'province': prov, 'sample_count': prow['sample_count'], 'exact_plate_acc': exact, 'first_char_acc': first})
            weak_first.sort(key=lambda x: (x['first_char_acc'], x['sample_count']))
            weak_exact.sort(key=lambda x: (x['exact_plate_acc'], x['sample_count']))
            out_families[fam] = {
                'sample_count': row['sample_count'],
                'exact_plate_acc': safe_div(row['exact_correct'], row['sample_count']),
                'first_char_acc': safe_div(row['first_correct'], row['sample_count']),
                'base_exact_plate_acc': safe_div(base_rows[fam]['exact_correct'], base_rows[fam]['sample_count']),
                'base_first_char_acc': safe_div(base_rows[fam]['first_correct'], base_rows[fam]['sample_count']),
                'province_metrics': province_metrics,
                'weak_first_char_under_0_60': weak_first,
                'weak_exact_under_0_60': weak_exact,
            }
        return out_families

    out = {
        'model': args.model,
        'manifest': args.manifest,
        'decode_mode': 'family_aware_beam',
        'province_fusion_mode': args.province_fusion_mode,
        'province_conf_threshold': args.province_conf_threshold,
        'fusion_changed_count': fusion_changed_count,
        'fusion_reasons': fusion_reasons,
        'families': summarize_family_rows(family_rows, base_family_rows),
    }

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
