#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
for _p in (str(_SRC_DIR), str(_SRC_DIR / 'training'), str(_SRC_DIR / 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_data import CHARS, CHARS_DICT, prepare_board_ocr_input_from_quad_bgr888
from eval_lpr_detailed import decode_logits
from LPRNet import build_lprnet
from LPRNet_multihead import build_lprnet_multihead, build_lprnet_multihead_from_state_dict
from test_LPRNet import collate_fn
from train_LPRNet import forward_family_logits
from quad_refiner.decode import decode_corner_heatmaps
from quad_refiner.geometry import build_patch_box_from_quad, gate_refined_quad, map_quad_from_patch
from quad_refiner.model import QuadHeatmapRefiner

BOARD_PARAMS = dict(
    width=94,
    height=24,
    resize_mode='letterbox',
    resize_kernel='nn',
    preproc='none',
    channel_order='bgr',
    quad_pad_ratio=0.0,
)

DEFAULT_GATE_PARAMS = dict(
    min_corner_conf=0.20,
    min_area_ratio=0.65,
    max_area_ratio=1.45,
    max_center_shift_ratio=0.20,
    max_corner_shift_ratio=0.18,
    max_edge_ratio_ratio=2.2,
)

FAMILY_GATE_PARAMS = {
    'green8': dict(DEFAULT_GATE_PARAMS, max_corner_shift_ratio=0.26),
}

CCPD_PROVINCES = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
]

CCPD_ADS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9',
]


def decode_ccpd_text_from_sample_id(sample_id: str) -> str:
    stem = str(sample_id).split(':', 1)[-1]
    parts = stem.split('-')
    if len(parts) < 5:
        return ''
    try:
        codes = [int(item) for item in parts[4].split('_')]
    except Exception:
        return ''
    if len(codes) not in (7, 8):
        return ''
    if not 0 <= codes[0] < len(CCPD_PROVINCES):
        return ''
    out = [CCPD_PROVINCES[codes[0]]]
    for code in codes[1:]:
        if not 0 <= code < len(CCPD_ADS):
            return ''
        out.append(CCPD_ADS[code])
    return ''.join(out)


class BoardTripletDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(row['img_path'])
        if image is None:
            raise RuntimeError(f"failed to read image: {row['img_path']}")
        quad = np.asarray(row['quad'], dtype=np.float32)
        prepared, occ, warped, _, _ = prepare_board_ocr_input_from_quad_bgr888(
            image,
            quad,
            BOARD_PARAMS['width'],
            BOARD_PARAMS['height'],
            BOARD_PARAMS['resize_mode'],
            BOARD_PARAMS['resize_kernel'],
            BOARD_PARAMS['preproc'],
            BOARD_PARAMS['channel_order'],
            quad_pad_ratio=BOARD_PARAMS['quad_pad_ratio'],
        )
        arr = prepared.astype(np.float32)
        arr -= 127.5
        arr *= 0.0078125
        arr = np.transpose(arr, (2, 0, 1))
        label = [CHARS_DICT[c] for c in row['text']]
        return arr, label, len(label), row['family']


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def load_expert_model(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    is_multihead = any(k.startswith('containers.') for k in state)
    if is_multihead:
        net, _cfg = build_lprnet_multihead_from_state_dict(
            state,
            lpr_max_len=8,
            phase=False,
            class_num=len(CHARS),
            dropout_rate=0,
        )
        net.load_state_dict(state, strict=False)
    else:
        net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
        net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net, is_multihead


def load_refiner(path, device):
    model = QuadHeatmapRefiner(pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    model.eval()
    return model


def refine_quad(image, coarse_quad, model, device, input_width=256, input_height=128, pad_x=0.20, pad_y=0.25, gate_params=None):
    img_h, img_w = image.shape[:2]
    patch_box = build_patch_box_from_quad(coarse_quad, img_w=img_w, img_h=img_h, pad_x=pad_x, pad_y=pad_y)
    patch = image[patch_box.y1:patch_box.y2 + 1, patch_box.x1:patch_box.x2 + 1]
    patch = cv2.resize(patch, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(patch.transpose(2, 0, 1)).float()[None] / 255.0
    with torch.no_grad():
        out = model(x.to(device))
    heatmaps = torch.sigmoid(out['heatmaps'])[0].cpu().numpy()
    pred_patch, confs = decode_corner_heatmaps(heatmaps, in_w=input_width, in_h=input_height)
    pred_quad = map_quad_from_patch(pred_patch, patch_box, in_w=input_width, in_h=input_height)
    gate_kwargs = dict(DEFAULT_GATE_PARAMS)
    if gate_params:
        gate_kwargs.update(gate_params)
    gate = gate_refined_quad(coarse_quad, pred_quad, confs, patch_diag=math.hypot(input_width, input_height), **gate_kwargs)
    final_quad = pred_quad if gate.accepted else coarse_quad
    return final_quad, gate, pred_quad, confs


def make_triplet_rows(records, family, refiner_ckpt, limit, split='val', gate_params=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    refiner = load_refiner(refiner_ckpt, device)
    chosen = []
    for rec in records:
        if rec.get('family') != family:
            continue
        if split and rec.get('split') != split:
            continue
        chosen.append(rec)
        if limit and len(chosen) >= limit:
            break
    rows = []
    preview = []
    for rec in chosen:
        image = cv2.imread(rec['image_path'])
        if image is None:
            continue
        gt_quad = np.asarray(rec['gt_quad'], dtype=np.float32)
        coarse_quad = np.asarray(rec.get('coarse_quad', rec['gt_quad']), dtype=np.float32)
        refined_quad, gate, pred_quad, confs = refine_quad(image, coarse_quad, refiner, device, gate_params=gate_params)
        text = rec.get('text') or decode_ccpd_text_from_sample_id(rec['sample_id'])
        base = {
            'sample_id': rec['sample_id'],
            'img_path': rec['image_path'],
            'text': text,
            'family': family,
            'source_name': rec.get('source_name', ''),
            'split': rec.get('split', ''),
        }
        rows.extend([
            dict(base, path_type='gt', quad=np.asarray(gt_quad, dtype=float).tolist()),
            dict(base, path_type='coarse', quad=np.asarray(coarse_quad, dtype=float).tolist()),
            dict(base, path_type='refined', quad=np.asarray(refined_quad, dtype=float).tolist()),
        ])
        if len(preview) < 20:
            preview.append({
                'sample_id': rec['sample_id'],
                'text': text,
                'family': family,
                'source_name': rec.get('source_name', ''),
                'gate_accepted': bool(gate.accepted),
                'gate_reason': gate.reason,
                'coarse_quad': np.asarray(coarse_quad, dtype=float).round(2).tolist(),
                'refined_quad': np.asarray(refined_quad, dtype=float).round(2).tolist(),
                'pred_quad': np.asarray(pred_quad, dtype=float).round(2).tolist(),
                'corner_conf': [float(x) for x in confs],
            })
    return rows, preview


def evaluate_rows(rows, model_path, family_filter, batch_size=200, num_workers=4):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net, is_multihead = load_expert_model(model_path, device)
    ds = BoardTripletDataset(rows)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    metrics = defaultdict(lambda: {'sample_count': 0, 'exact': 0, 'first': 0, 'by_source': defaultdict(lambda: {'sample_count': 0, 'exact': 0, 'first': 0}), 'bad_cases': []})
    row_idx = 0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            batch_rows = rows[row_idx:row_idx + len(families)]
            row_idx += len(families)
            targets = []
            start = 0
            for length in lengths:
                gt_ids = labels[start:start + length].numpy().tolist()
                gt = ''.join(CHARS[int(c)] for c in gt_ids)
                targets.append(gt)
                start += length
            if is_multihead:
                images_t = images.to(device)
                sample_families = list(families)
                logits = forward_family_logits(net, images_t, sample_families=sample_families).detach().cpu().numpy()
                decoded = decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=sample_families)
                preds = [''.join(CHARS[int(c)] for c in pred_ids) for pred_ids in decoded]
            else:
                images_t = images.to(device)
                logits = net(images_t).cpu().detach().numpy()
                decoded = decode_logits(logits, 'greedy', 20, 12, sample_families=list(families))
                preds = [''.join(CHARS[int(c)] for c in pred_ids) for pred_ids in decoded]
            for row, gt, pred in zip(batch_rows, targets, preds):
                bucket = metrics[row['path_type']]
                bucket['sample_count'] += 1
                bucket['exact'] += int(pred == gt)
                bucket['first'] += int(bool(pred) and bool(gt) and pred[0] == gt[0])
                src = row.get('source_name', '')
                bucket['by_source'][src]['sample_count'] += 1
                bucket['by_source'][src]['exact'] += int(pred == gt)
                bucket['by_source'][src]['first'] += int(bool(pred) and bool(gt) and pred[0] == gt[0])
                if len(bucket['bad_cases']) < 10 and pred != gt:
                    bucket['bad_cases'].append({'sample_id': row['sample_id'], 'source_name': src, 'gt': gt, 'pred': pred})
    out = {'model': model_path, 'family': family_filter, 'decode_mode': 'family_aware_beam' if is_multihead else 'greedy', 'paths': {}}
    for path_type, st in metrics.items():
        out['paths'][path_type] = {
            'sample_count': st['sample_count'],
            'exact_plate_acc': safe_div(st['exact'], st['sample_count']),
            'first_char_acc': safe_div(st['first'], st['sample_count']),
            'bad_cases': st['bad_cases'],
            'by_source': {
                src: {
                    'sample_count': s['sample_count'],
                    'exact_plate_acc': safe_div(s['exact'], s['sample_count']),
                    'first_char_acc': safe_div(s['first'], s['sample_count']),
                }
                for src, s in sorted(st['by_source'].items())
            }
        }
    return out


def save_manifest(rows, path):
    fieldnames = ['sample_id', 'img_path', 'text', 'family', 'source_name', 'split', 'path_type', 'quad']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row['quad'] = json.dumps(row['quad'], ensure_ascii=False)
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description='Generate GT/coarse/refined board-consistent triplets and evaluate OCR experts.')
    ap.add_argument('--records-jsonl', required=True)
    ap.add_argument('--refiner-ckpt', required=True)
    ap.add_argument('--blue-model', required=True)
    ap.add_argument('--green-model', required=True)
    ap.add_argument('--blue-limit', type=int, default=1000)
    ap.add_argument('--green-limit', type=int, default=1000)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.records_jsonl).open('r', encoding='utf-8')]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blue_rows, blue_preview = make_triplet_rows(
        records,
        'normal7',
        args.refiner_ckpt,
        args.blue_limit,
        split=args.split,
        gate_params=FAMILY_GATE_PARAMS.get('normal7'),
    )
    green_rows, green_preview = make_triplet_rows(
        records,
        'green8',
        args.refiner_ckpt,
        args.green_limit,
        split=args.split,
        gate_params=FAMILY_GATE_PARAMS.get('green8'),
    )

    save_manifest(blue_rows, out_dir / 'blue_triplets.csv')
    save_manifest(green_rows, out_dir / 'green_triplets.csv')

    blue_eval = evaluate_rows(blue_rows, args.blue_model, 'normal7')
    green_eval = evaluate_rows(green_rows, args.green_model, 'green8')

    (out_dir / 'blue_preview.json').write_text(json.dumps(blue_preview, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'green_preview.json').write_text(json.dumps(green_preview, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'blue_eval.json').write_text(json.dumps(blue_eval, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'green_eval.json').write_text(json.dumps(green_eval, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {
        'board_params': BOARD_PARAMS,
        'records_jsonl': args.records_jsonl,
        'refiner_ckpt': args.refiner_ckpt,
        'blue_model': args.blue_model,
        'green_model': args.green_model,
        'split': args.split,
        'blue_samples': len(blue_rows) // 3,
        'green_samples': len(green_rows) // 3,
        'blue_eval': blue_eval,
        'green_eval': green_eval,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out_dir': str(out_dir), 'blue_samples': len(blue_rows) // 3, 'green_samples': len(green_rows) // 3}, ensure_ascii=False))


if __name__ == '__main__':
    main()
