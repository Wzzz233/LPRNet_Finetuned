#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'evaluation'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import UnifiedManifestDataset, CHARS
from eval_lpr_detailed import decode_logits
from LPRNet_multihead import build_lprnet_multihead_from_state_dict
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits

MANIFEST_FIELDS = [
    'img_path', 'img_rel_path', 'dataset_name', 'split', 'text', 'plate_len', 'family', 'sub_type', 'source',
    'is_real', 'need_tilt_aug', 'preprocess_group', 'has_bbox', 'has_quad', 'can_parse_ccpd_geom', 'can_perspective',
    'bbox_source', 'quad_source', 'ocr_channel_order', 'ocr_crop_mode', 'ocr_resize_mode', 'ocr_resize_kernel',
    'ocr_preproc', 'ocr_min_occ_ratio', 'ocr_quad_pad_ratio'
]


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def is_cjk(tok):
    if not tok:
        return False
    cp = ord(tok[0])
    return 0x4E00 <= cp <= 0x9FFF


def fuse_first_char(base_text, pos0_char, pos0_conf, mode, threshold):
    if not base_text or not pos0_char:
        return base_text, False, 'empty'
    if mode == 'replace_all':
        if base_text[0] == pos0_char:
            return base_text, False, 'same'
        return pos0_char + base_text[1:], True, 'replace_all'
    if mode == 'replace_if_not_cjk':
        if is_cjk(base_text[0]):
            return base_text, False, 'base_is_cjk'
        if base_text[0] == pos0_char:
            return base_text, False, 'same'
        return pos0_char + base_text[1:], True, 'replace_if_not_cjk'
    if mode == 'replace_if_confident':
        if pos0_conf < threshold:
            return base_text, False, 'below_threshold'
        if base_text[0] == pos0_char:
            return base_text, False, 'same'
        return pos0_char + base_text[1:], True, 'replace_if_confident'
    return base_text, False, 'fusion_disabled'


def text_to_ids(text):
    out = []
    for ch in text:
        try:
            out.append(CHARS.index(ch))
        except ValueError:
            return None
    return out


def extract_pos0_logits(raw_dict, families):
    if 'pos0' in raw_dict and raw_dict['pos0'] is not None:
        return raw_dict['pos0']
    selected = []
    for i, family in enumerate(families):
        key = f'pos0_{family}'
        if key not in raw_dict or raw_dict[key] is None:
            return None
        selected.append(raw_dict[key][i:i + 1])
    return torch.cat(selected, dim=0) if selected else None


def build_temp_manifest(input_csv: Path, temp_manifest: Path):
    rows = []
    meta_rows = []
    with input_csv.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            img_path = row.get('local_ocrin_path') or row.get('ocr_input_path') or row.get('img_path')
            gt = row.get('gt_text', '').strip()
            if not img_path or not gt:
                continue
            img_path = str(Path(img_path))
            if not Path(img_path).exists():
                continue
            rows.append({
                'img_path': img_path,
                'img_rel_path': img_path,
                'dataset_name': 'dump_replay',
                'split': 'test',
                'text': gt,
                'plate_len': len(gt),
                'family': 'green8',
                'sub_type': 'green',
                'source': 'dump_replay',
                'is_real': 1,
                'need_tilt_aug': 0,
                'preprocess_group': 'dump_replay',
                'has_bbox': 0,
                'has_quad': 0,
                'can_parse_ccpd_geom': 0,
                'can_perspective': 0,
                'bbox_source': 'none',
                'quad_source': 'none',
                'ocr_channel_order': 'bgr',
                'ocr_crop_mode': 'obb_warp',
                'ocr_resize_mode': 'letterbox',
                'ocr_resize_kernel': 'nn',
                'ocr_preproc': 'none',
                'ocr_min_occ_ratio': 0.9,
                'ocr_quad_pad_ratio': 0.0,
            })
            meta = dict(row)
            meta['_row_index'] = i
            meta['_img_path'] = img_path
            meta_rows.append(meta)
    temp_manifest.parent.mkdir(parents=True, exist_ok=True)
    with temp_manifest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return meta_rows


def load_model(model_path: Path, device):
    state = torch.load(str(model_path), map_location=device)
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
    ap = argparse.ArgumentParser(description='Replay board dump OCRIN images with family-aware decode and summarize by GT text.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--input-csv', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--batch-size', type=int, default=300)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--decode-mode', default='family_aware_beam', choices=['greedy', 'green_ctc_beam', 'family_aware_beam'])
    ap.add_argument('--beam-size', type=int, default=20)
    ap.add_argument('--beam-topk', type=int, default=12)
    ap.add_argument('--pos0-fusion-mode', default='none', choices=['none', 'replace_all', 'replace_if_confident', 'replace_if_not_cjk'])
    ap.add_argument('--pos0-conf-threshold', type=float, default=0.55)
    args = ap.parse_args()

    model_path = Path(args.model)
    input_csv = Path(args.input_csv)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    temp_manifest = out_csv.parent / (out_csv.stem + '.manifest.csv')

    meta_rows = build_temp_manifest(input_csv, temp_manifest)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds_full = UnifiedManifestDataset(
        manifest_path=str(temp_manifest),
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
    idx = [i for i, row in enumerate(ds_full.records) if Path(row.get('img_path', '')).exists()]
    ds = Subset(ds_full, idx)
    meta_rows = [meta_rows[i] for i in idx]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    net = load_model(model_path, device)

    total = 0
    exact = 0
    first_char = 0
    base_exact = 0
    base_first_char = 0
    detail_rows = []
    per_gt = defaultdict(lambda: {'sample_count': 0, 'exact_correct': 0, 'first_char_correct': 0, 'pred_counter': Counter()})
    per_gt_base = defaultdict(lambda: {'sample_count': 0, 'exact_correct': 0, 'first_char_correct': 0, 'pred_counter': Counter()})
    fusion_reasons = Counter()
    changed_count = 0
    meta_cursor = 0
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
            logits = forward_family_logits(net, images, sample_families=sample_families).detach().cpu().numpy()
            decoded = decode_logits(logits, args.decode_mode, args.beam_size, args.beam_topk, sample_families=sample_families)
            pos0_prob = None
            if args.pos0_fusion_mode != 'none':
                pos0_logits = extract_pos0_logits(raw_outputs, sample_families)
                if pos0_logits is None:
                    raise RuntimeError(f'--pos0-fusion-mode={args.pos0_fusion_mode} but model has no usable pos0 logits for families={sorted(set(sample_families))}')
                pos0_prob = F.softmax(pos0_logits, dim=1).detach().cpu().numpy()
            for batch_idx, (pred_ids, gt_ids) in enumerate(zip(decoded, targets)):
                base_pred = ''.join(CHARS[int(c)] for c in pred_ids)
                gt = ''.join(CHARS[int(c)] for c in gt_ids.tolist())
                meta = meta_rows[meta_cursor]
                meta_cursor += 1
                pred = base_pred
                pos0_char = ''
                pos0_conf = 0.0
                fusion_reason = 'fusion_disabled'
                fusion_changed = 0
                if pos0_prob is not None:
                    pos0_idx = int(np.argmax(pos0_prob[batch_idx]))
                    pos0_conf = float(pos0_prob[batch_idx][pos0_idx])
                    pos0_char = CHARS[pos0_idx] if pos0_idx < len(CHARS) else ''
                    pred, changed, fusion_reason = fuse_first_char(base_pred, pos0_char, pos0_conf, args.pos0_fusion_mode, args.pos0_conf_threshold)
                    if changed and text_to_ids(pred) is None:
                        pred = base_pred
                        changed = False
                        fusion_reason = 'invalid_fused_text'
                    fusion_changed = int(changed)
                    changed_count += fusion_changed
                    fusion_reasons[fusion_reason] += 1
                is_exact = int(pred == gt)
                is_first = int(bool(pred) and bool(gt) and pred[0] == gt[0])
                base_is_exact = int(base_pred == gt)
                base_is_first = int(bool(base_pred) and bool(gt) and base_pred[0] == gt[0])
                total += 1
                exact += is_exact
                first_char += is_first
                base_exact += base_is_exact
                base_first_char += base_is_first
                p = per_gt[gt]
                p['sample_count'] += 1
                p['exact_correct'] += is_exact
                p['first_char_correct'] += is_first
                p['pred_counter'][pred] += 1
                p_base = per_gt_base[gt]
                p_base['sample_count'] += 1
                p_base['exact_correct'] += base_is_exact
                p_base['first_char_correct'] += base_is_first
                p_base['pred_counter'][base_pred] += 1
                detail = dict(meta)
                detail['pred_text'] = pred
                detail['base_pred_text'] = base_pred
                detail['gt_text'] = gt
                detail['exact_match'] = is_exact
                detail['first_char_match'] = is_first
                detail['base_exact_match'] = base_is_exact
                detail['base_first_char_match'] = base_is_first
                detail['pos0_char'] = pos0_char
                detail['pos0_conf'] = round(pos0_conf, 6)
                detail['fusion_reason'] = fusion_reason
                detail['fusion_changed'] = fusion_changed
                detail_rows.append(detail)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(detail_rows[0].keys()) if detail_rows else ['gt_text', 'pred_text', 'exact_match', 'first_char_match']
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    report = {
        'model': str(model_path),
        'input_csv': str(input_csv),
        'decode_mode': args.decode_mode,
        'pos0_fusion_mode': args.pos0_fusion_mode,
        'pos0_conf_threshold': args.pos0_conf_threshold,
        'sample_count': total,
        'exact_plate_acc': safe_div(exact, total),
        'first_char_acc': safe_div(first_char, total),
        'base_exact_plate_acc': safe_div(base_exact, total),
        'base_first_char_acc': safe_div(base_first_char, total),
        'fusion_changed_count': changed_count,
        'fusion_reasons': dict(fusion_reasons),
        'per_gt_text': {},
    }
    for gt, stats in sorted(per_gt.items()):
        top_preds = [{'pred_text': pred, 'count': cnt} for pred, cnt in stats['pred_counter'].most_common(5)]
        base_stats = per_gt_base[gt]
        base_top_preds = [{'pred_text': pred, 'count': cnt} for pred, cnt in base_stats['pred_counter'].most_common(5)]
        report['per_gt_text'][gt] = {
            'sample_count': stats['sample_count'],
            'exact_plate_acc': safe_div(stats['exact_correct'], stats['sample_count']),
            'first_char_acc': safe_div(stats['first_char_correct'], stats['sample_count']),
            'base_exact_plate_acc': safe_div(base_stats['exact_correct'], base_stats['sample_count']),
            'base_first_char_acc': safe_div(base_stats['first_char_correct'], base_stats['sample_count']),
            'top_predictions': top_preds,
            'base_top_predictions': base_top_preds,
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
