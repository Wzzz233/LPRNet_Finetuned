#!/usr/bin/env python3
import argparse
import json
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import UnifiedManifestDataset, CHARS
from eval_lpr_detailed import decode_logits
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits, _select_family_logits_from_dict
from firstchar_fusion import extract_province_logits, fuse_first_char


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
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--province-model', default='', help='optional secondary model used only for province logits during province fusion')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--batch_size', type=int, default=300)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--ocr_preproc', default='none', choices=['none', 'raw', 'gray', 'gray3', 'bin'], help='OCR preprocess mode for evaluation')
    ap.add_argument('--province-fusion-mode', default='none', choices=['none', 'replace_all', 'replace_if_confident', 'replace_if_not_cjk'])
    ap.add_argument('--province-conf-threshold', type=float, default=0.55)
    args = ap.parse_args()

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
        ocr_preproc=args.ocr_preproc,
        ocr_min_occ_ratio=0.90,
        ocr_quad_pad_ratio=0.0,
    )
    idx = [
        i for i, row in enumerate(full.records)
        if (row.get('family') or '').strip() == 'green8' and row.get('img_path') and Path(row.get('img_path')).exists()
    ]
    if len(idx) != sum(1 for row in full.records if (row.get('family') or '').strip() == 'green8'):
        print(f'[Info] skip missing-image green8 eval rows: keep {len(idx)}')
    ds = Subset(full, idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    net = load_model(args.model, device)
    province_model_path = ''
    province_net = None
    if args.province_fusion_mode != 'none':
        province_model_path = args.province_model or args.model
        if Path(province_model_path).resolve() == Path(args.model).resolve():
            province_net = net
        else:
            province_net = load_model(province_model_path, device)

    exact = 0
    total = 0
    first_correct = 0
    base_exact = 0
    base_first_correct = 0
    province_rows = {}
    base_province_rows = {}
    pos2_correct = 0
    pos2_total = 0
    pos3_correct = 0
    pos3_total = 0
    base_pos2_correct = 0
    base_pos3_correct = 0
    fusion_reasons = Counter()
    fusion_changed_count = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            images = images.to(device)
            sample_families = list(families)
            raw_outputs = net(images)
            logits = _select_family_logits_from_dict(raw_outputs, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
            province_prob = None
            if args.province_fusion_mode != 'none':
                province_raw_outputs = raw_outputs if province_net is net else province_net(images)
                province_logits = extract_province_logits(province_raw_outputs, sample_families)
                if province_logits is None:
                    raise RuntimeError(
                        f'--province-fusion-mode={args.province_fusion_mode} but model has no usable province logits '
                        f'(province model: {province_model_path})'
                    )
                province_prob = F.softmax(province_logits, dim=1).detach().cpu().numpy()
            for i, (pred_ids, gt_ids) in enumerate(zip(decoded, targets)):
                base_pred = ''.join(CHARS[int(c)] for c in pred_ids)
                pred = base_pred
                gt = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                if province_prob is not None:
                    province_idx = int(province_prob[i].argmax())
                    province_conf = float(province_prob[i][province_idx])
                    province_char = CHARS[province_idx] if province_idx < len(CHARS) else ''
                    pred, changed, reason = fuse_first_char(base_pred, province_char, province_conf, args.province_fusion_mode, args.province_conf_threshold)
                    fusion_reasons[reason] += 1
                    fusion_changed_count += int(changed)
                total += 1
                exact += int(pred == gt)
                base_exact += int(base_pred == gt)
                if gt:
                    row = province_rows.setdefault(gt[0], {'sample_count': 0, 'exact_plate_correct': 0, 'first_char_correct': 0})
                    row['sample_count'] += 1
                    row['exact_plate_correct'] += int(pred == gt)
                    row['first_char_correct'] += int(bool(pred) and pred[0] == gt[0])
                    first_correct += int(bool(pred) and pred[0] == gt[0])

                    base_row = base_province_rows.setdefault(gt[0], {'sample_count': 0, 'exact_plate_correct': 0, 'first_char_correct': 0})
                    base_row['sample_count'] += 1
                    base_row['exact_plate_correct'] += int(base_pred == gt)
                    base_row['first_char_correct'] += int(bool(base_pred) and base_pred[0] == gt[0])
                    base_first_correct += int(bool(base_pred) and base_pred[0] == gt[0])
                if len(gt) > 1:
                    pos2_total += 1
                    pos2_correct += int(len(pred) > 1 and pred[1] == gt[1])
                    base_pos2_correct += int(len(base_pred) > 1 and base_pred[1] == gt[1])
                for pos in range(2, len(gt)):
                    pos3_total += 1
                    pos3_correct += int(len(pred) > pos and pred[pos] == gt[pos])
                    base_pos3_correct += int(len(base_pred) > pos and base_pred[pos] == gt[pos])

    province_macro_exact = 0.0
    province_macro_first = 0.0
    if province_rows:
        province_macro_exact = sum(safe_div(v['exact_plate_correct'], v['sample_count']) for v in province_rows.values()) / len(province_rows)
        province_macro_first = sum(safe_div(v['first_char_correct'], v['sample_count']) for v in province_rows.values()) / len(province_rows)

    base_province_macro_exact = 0.0
    base_province_macro_first = 0.0
    if base_province_rows:
        base_province_macro_exact = sum(safe_div(v['exact_plate_correct'], v['sample_count']) for v in base_province_rows.values()) / len(base_province_rows)
        base_province_macro_first = sum(safe_div(v['first_char_correct'], v['sample_count']) for v in base_province_rows.values()) / len(base_province_rows)

    major_province = None
    major_ratio = 0.0
    major_exact = 0.0
    non_major_exact = 0.0
    if province_rows:
        max_count = max(v['sample_count'] for v in province_rows.values())
        majors = sorted([k for k, v in province_rows.items() if v['sample_count'] == max_count])
        major_province = '|'.join(majors)
        major_total = sum(province_rows[k]['sample_count'] for k in majors)
        major_correct = sum(province_rows[k]['exact_plate_correct'] for k in majors)
        non_total = sum(v['sample_count'] for k, v in province_rows.items() if k not in majors)
        non_correct = sum(v['exact_plate_correct'] for k, v in province_rows.items() if k not in majors)
        major_ratio = safe_div(major_total, total)
        major_exact = safe_div(major_correct, major_total)
        non_major_exact = safe_div(non_correct, non_total)

    report = {
        'model': args.model,
        'province_model': province_model_path or args.model,
        'family': 'green8',
        'sample_count': total,
        'province_fusion_mode': args.province_fusion_mode,
        'province_conf_threshold': args.province_conf_threshold,
        'exact_plate_acc': safe_div(exact, total),
        'first_char_acc': safe_div(first_correct, total),
        'province_macro_exact_acc': province_macro_exact,
        'province_macro_first_char_acc': province_macro_first,
        'major_province': major_province,
        'major_province_ratio': major_ratio,
        'major_province_exact_acc': major_exact,
        'non_major_province_exact_acc': non_major_exact,
        'pos2_alpha_acc': safe_div(pos2_correct, pos2_total),
        'pos3plus_alnum_acc': safe_div(pos3_correct, pos3_total),
        'base_exact_plate_acc': safe_div(base_exact, total),
        'base_first_char_acc': safe_div(base_first_correct, total),
        'base_province_macro_exact_acc': base_province_macro_exact,
        'base_province_macro_first_char_acc': base_province_macro_first,
        'base_pos2_alpha_acc': safe_div(base_pos2_correct, pos2_total),
        'base_pos3plus_alnum_acc': safe_div(base_pos3_correct, pos3_total),
        'fusion_changed_count': fusion_changed_count,
        'fusion_reasons': dict(fusion_reasons),
        'province_breakdown': {
            k: {
                'sample_count': v['sample_count'],
                'exact_plate_acc': safe_div(v['exact_plate_correct'], v['sample_count']),
                'first_char_acc': safe_div(v['first_char_correct'], v['sample_count']),
                'base_exact_plate_acc': safe_div(base_province_rows[k]['exact_plate_correct'], base_province_rows[k]['sample_count']),
                'base_first_char_acc': safe_div(base_province_rows[k]['first_char_correct'], base_province_rows[k]['sample_count']),
            }
            for k, v in sorted(province_rows.items())
        }
    }
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
