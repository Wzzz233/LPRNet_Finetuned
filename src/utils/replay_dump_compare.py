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
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits
from firstchar_fusion import extract_pos0_logits as extract_pos0_logits_helper, extract_province_logits as extract_province_logits_helper, fuse_first_char as fuse_first_char_helper
from train_tiny_province_net import TinyProvinceNet

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
    return fuse_first_char_helper(base_text, pos0_char, pos0_conf, mode, threshold)


def text_to_ids(text):
    out = []
    for ch in text:
        try:
            out.append(CHARS.index(ch))
        except ValueError:
            return None
    return out


def extract_pos0_logits(raw_dict, families):
    return extract_pos0_logits_helper(raw_dict, families)


def extract_province_logits(raw_dict, families):
    return extract_province_logits_helper(raw_dict, families)


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
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device)
    net.eval()
    return net


def load_tiny_province_model(model_path: Path, device):
    state = torch.load(str(model_path), map_location=device)
    in_channels = int(state['features.0.weight'].shape[1])
    net = TinyProvinceNet(in_channels=in_channels)
    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()
    return net, in_channels


def crop_left_patch_bgr(img_bgr, patch_ratio=0.42, min_width=24):
    if img_bgr is None or img_bgr.size == 0:
        return None
    _h, w = img_bgr.shape[:2]
    patch_w = max(min_width, int(round(w * patch_ratio)))
    patch_w = min(max(1, w), patch_w)
    patch = img_bgr[:, :patch_w]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (94, 24), interpolation=cv2.INTER_NEAREST)


def crop_full_for_a4c(img_bgr, target_h=64, target_w=171, gray3=False):
    if img_bgr is None or img_bgr.size == 0:
        return None
    out = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if gray3:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out


def load_aux_patch_batch(meta_batch, device, mode, patch_ratio):
    patches = []
    valid = []
    target_shape = (24, 94, 3)
    if mode == 'full_crop_gray3':
        target_shape = (64, 171, 3)
    for meta in meta_batch:
        aux_src = meta.get('local_aux_fullcrop_path') or meta.get('aux_fullcrop_path')
        src = aux_src or meta.get('local_crop_path') or meta.get('crop_path')
        if mode == 'ocrin_full':
            src = meta.get('local_ocrin_path') or meta.get('ocr_input_path') or src
        img = cv2.imread(str(src), cv2.IMREAD_COLOR) if src else None
        if mode == 'full_crop_gray3':
            patch = crop_full_for_a4c(img, target_h=64, target_w=171, gray3=True)
        else:
            patch = crop_left_patch_bgr(img, patch_ratio=patch_ratio)
        if patch is None:
            patches.append(np.zeros(target_shape, dtype=np.uint8))
            valid.append(False)
        else:
            patches.append(patch)
            valid.append(True)
    arr = np.stack(patches, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
    return tensor, valid


def main():
    ap = argparse.ArgumentParser(description='Replay board dump OCRIN images with family-aware decode and summarize by GT text.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--province-model', default='', help='optional secondary model used only for province logits during province fusion')
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
    ap.add_argument('--province-fusion-mode', default='none', choices=['none', 'replace_all', 'replace_if_confident', 'replace_if_not_cjk'])
    ap.add_argument('--province-conf-threshold', type=float, default=0.55)
    ap.add_argument('--independent-province-model', default='', help='standalone TinyProvinceNet checkpoint used for province fusion')
    ap.add_argument('--independent-province-input', default='crop_left_patch', choices=['crop_left_patch', 'ocrin_full', 'full_crop_gray3'])
    ap.add_argument('--independent-province-patch-ratio', type=float, default=0.42)
    args = ap.parse_args()
    if args.pos0_fusion_mode != 'none' and args.province_fusion_mode != 'none':
        raise RuntimeError('Only one of --pos0-fusion-mode / --province-fusion-mode may be enabled at a time')

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
    province_model_path = ''
    province_net = None
    tiny_province_net = None
    tiny_province_in_channels = 3
    if args.province_fusion_mode != 'none':
        if args.independent_province_model:
            province_model_path = args.independent_province_model
            tiny_province_net, tiny_province_in_channels = load_tiny_province_model(Path(province_model_path), device)
        else:
            province_model_path = args.province_model or args.model
            if Path(province_model_path).resolve() == model_path.resolve():
                province_net = net
            else:
                province_net = load_model(Path(province_model_path), device)

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
    fusion_type = 'disabled'
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
            province_prob = None
            fusion_type = 'disabled'
            if args.pos0_fusion_mode != 'none':
                pos0_logits = extract_pos0_logits(raw_outputs, sample_families)
                if pos0_logits is None:
                    raise RuntimeError(f'--pos0-fusion-mode={args.pos0_fusion_mode} but model has no usable pos0 logits for families={sorted(set(sample_families))}')
                pos0_prob = F.softmax(pos0_logits, dim=1).detach().cpu().numpy()
                fusion_type = 'pos0'
            elif args.province_fusion_mode != 'none':
                if tiny_province_net is not None:
                    meta_batch = meta_rows[meta_cursor:meta_cursor + len(sample_families)]
                    aux_images, valid_patch = load_aux_patch_batch(
                        meta_batch,
                        device,
                        args.independent_province_input,
                        args.independent_province_patch_ratio,
                    )
                    if tiny_province_in_channels == 1:
                        aux_images = aux_images[:, 0:1, :, :] * 0.1140 + aux_images[:, 1:2, :, :] * 0.5870 + aux_images[:, 2:3, :, :] * 0.2990
                    province_logits = tiny_province_net(aux_images)
                    province_prob = F.softmax(province_logits, dim=1).detach().cpu().numpy()
                    for patch_ok, batch_idx in zip(valid_patch, range(len(sample_families))):
                        if not patch_ok:
                            province_prob[batch_idx] = 0.0
                else:
                    province_raw_outputs = raw_outputs if province_net is net else province_net(images)
                    province_logits = extract_province_logits(province_raw_outputs, sample_families)
                    if province_logits is None:
                        raise RuntimeError(
                            f'--province-fusion-mode={args.province_fusion_mode} but model has no usable province logits '
                            f'for families={sorted(set(sample_families))} (province model: {province_model_path})'
                        )
                    province_prob = F.softmax(province_logits, dim=1).detach().cpu().numpy()
                fusion_type = 'province'
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
                elif province_prob is not None:
                    pos0_idx = int(np.argmax(province_prob[batch_idx]))
                    pos0_conf = float(province_prob[batch_idx][pos0_idx])
                    pos0_char = CHARS[pos0_idx] if pos0_idx < len(CHARS) else ''
                    pred, changed, fusion_reason = fuse_first_char(base_pred, pos0_char, pos0_conf, args.province_fusion_mode, args.province_conf_threshold)
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
                detail['aux_char'] = pos0_char
                detail['aux_conf'] = round(pos0_conf, 6)
                detail['fusion_type'] = fusion_type
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
        'province_model': province_model_path or str(model_path),
        'input_csv': str(input_csv),
        'decode_mode': args.decode_mode,
        'pos0_fusion_mode': args.pos0_fusion_mode,
        'pos0_conf_threshold': args.pos0_conf_threshold,
        'province_fusion_mode': args.province_fusion_mode,
        'province_conf_threshold': args.province_conf_threshold,
        'fusion_type': fusion_type,
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
